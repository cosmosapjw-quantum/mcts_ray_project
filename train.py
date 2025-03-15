# train.py - Optimized Self-Play Manager
import os
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import multiprocessing
import threading

from model import SmallResNet
from inference.inference_server import InferenceServer
from utils.state_utils import TicTacToeState
from mcts.search import parallel_mcts, BatchMCTSWorker
from config import NUM_SIMULATIONS, NUM_WORKERS, SIMULATIONS_PER_WORKER, VERBOSE

# Training parameters
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100000
CHECKPOINT_DIR = 'checkpoints'
TEMPERATURE_SCHEDULE = {
    0: 1.0,     # First 100 moves use temperature 1.0
    10: 0.8,    # After 10 moves, temperature drops to 0.8
    20: 0.5,    # After 20 moves, temperature drops to 0.5
}

class ReplayBuffer:
    """Prioritized experience replay buffer"""
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, experience, priority=1.0):
        """Add an experience to the buffer with priority"""
        with self.lock:
            self.buffer.append(experience)
            self.priorities.append(priority)
        
    def sample(self, batch_size):
        """Sample a batch of experiences based on priority"""
        with self.lock:
            if len(self.buffer) < batch_size:
                batch_size = len(self.buffer)
                
            # Convert priorities to probabilities
            probs = np.array(self.priorities) / sum(self.priorities)
            
            # Sample indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            
            states, policies, values = [], [], []
            
            for idx in indices:
                state, policy, value = self.buffer[idx]
                
                # Make sure everything has the right shape
                if state.shape != torch.Size([9]):
                    print(f"Warning: unexpected state shape: {state.shape}, expected [9]")
                    continue
                    
                if policy.shape != torch.Size([9]):
                    print(f"Warning: unexpected policy shape: {policy.shape}, expected [9]")
                    continue
                
                states.append(state)
                policies.append(policy)
                values.append(value)
            
            # Safety check
            if not states:
                # Empty batch after filtering, return None
                return None
                
            return (
                torch.stack(states),
                torch.stack(policies),
                torch.tensor(values, dtype=torch.float)
            )
        
    def __len__(self):
        return len(self.buffer)

class SelfPlayManager:
    """Manager for self-play, training and evaluation with parallel game generation"""
    
    def __init__(self):
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init()
            
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SmallResNet().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Initialize inference server actor with smaller cache and batch settings
        self.inference_actor = InferenceServer.remote(
            batch_wait=0.002,  # Very short wait time
            cache_size=5000,   # Smaller cache to force more network usage
            max_batch_size=64  # Larger batch size to improve GPU utilization
        )
        
        # Create MCTS batch workers for reuse
        self.batch_workers = [
            BatchMCTSWorker.remote(self.inference_actor, batch_size=32)
            for _ in range(NUM_WORKERS)
        ]
        
        # Initialize logging
        self.writer = SummaryWriter()
        
        # Game counter and metrics
        self.game_count = 0
        self.game_start_times = {}
        self.win_rates = {1: 0, -1: 0, 0: 0}  # Player 1, Player -1, Draw
        
    def generate_game(self, game_id):
        """Generate a single self-play game"""
        # Track game start time
        self.game_start_times[game_id] = time.time()
        
        state = TicTacToeState()
        memory = []
        move_count = 0
        
        # Play until game is finished
        while not state.is_terminal():
            # Determine temperature based on move count
            temperature = 1.0
            for move_threshold, temp in sorted(TEMPERATURE_SCHEDULE.items()):
                if move_count >= move_threshold:
                    temperature = temp
            
            # Use parallel MCTS with one of our batch workers
            # We alternate workers for better load distribution
            worker_idx = game_id % len(self.batch_workers)
            worker = self.batch_workers[worker_idx]
            
            # Run search on the selected worker
            root_future = worker.search.remote(
                state, 
                NUM_SIMULATIONS * SIMULATIONS_PER_WORKER // NUM_WORKERS,
                add_noise=(move_count < 10)  # Only add noise in the first 10 moves
            )
            
            # Get result
            root = ray.get(root_future)
            
            # Extract visit counts to create policy
            visits = np.array([child.visits for child in root.children])
            actions = [child.action for child in root.children]
            
            # Apply temperature and select action
            if temperature == 0:
                # Deterministic selection
                best_idx = np.argmax(visits)
                action = actions[best_idx]
            else:
                # Apply temperature
                if temperature == 1.0:
                    probs = visits / np.sum(visits)
                else:
                    # Apply temperature scaling
                    visits_temp = visits ** (1.0 / temperature)
                    probs = visits_temp / np.sum(visits_temp)
                
                # Sample action based on distribution
                action = np.random.choice(actions, p=probs)
            
            # Create full policy vector - ALWAYS size 9 for TicTacToe
            policy = np.zeros(9)  # Fixed size for all policies
            for i, a in enumerate(actions):
                policy[a] = visits[i] / np.sum(visits)
            
            # Convert to tensors
            state_tensor = torch.tensor(state.board, dtype=torch.float).to(self.device)
            policy_tensor = torch.tensor(policy, dtype=torch.float).to(self.device)
            
            # Debug the shapes
            if VERBOSE and self.game_count == 0 and move_count == 0:
                print(f"State tensor shape: {state_tensor.shape}, Policy tensor shape: {policy_tensor.shape}")
                
            # Store experience
            memory.append((state_tensor, policy_tensor, state.current_player))
            
            # Apply action
            state = state.apply_action(action)
            move_count += 1
            
            if VERBOSE and move_count % 5 == 0:
                print(f"Game {game_id}, Move {move_count}")
        
        # Game finished
        outcome = state.winner
        game_time = time.time() - self.game_start_times[game_id]
        
        if VERBOSE:
            print(f"Game {game_id} finished after {move_count} moves with outcome: {outcome}, time: {game_time:.1f}s")
        
        # Update win rate statistics
        with threading.Lock():
            self.win_rates[outcome] = self.win_rates.get(outcome, 0) + 1
        
        # Add experiences to replay buffer
        for state_tensor, policy_tensor, player in memory:
            # Set target value based on game outcome
            if outcome == 0:  # Draw
                target_value = 0.0
            else:
                target_value = 1.0 if outcome == player else -1.0
            
            # Higher priority for decisive game outcomes
            priority = 2.0 if outcome != 0 else 1.0
            
            # Add to buffer
            self.replay_buffer.add((state_tensor, policy_tensor, target_value), priority)
            
        return outcome, move_count
    
    def generate_games_parallel(self, num_games):
        """Generate multiple games in parallel"""
        game_ids = list(range(self.game_count, self.game_count + num_games))
        
        # Launch games in parallel
        game_futures = [self.generate_game(game_id) for game_id in game_ids]
        
        # Get results
        results = []
        for future in game_futures:
            results.append(future)
            
        self.game_count += num_games
        return results
        
    def train_batch(self):
        """Train on a batch of experiences"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None
            
        # Sample batch
        batch = self.replay_buffer.sample(BATCH_SIZE)
        if batch is None:
            return None
            
        states, policies, values = batch
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_policies, pred_values = self.model(states)
        
        # Calculate loss
        policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / policies.size(0)
        value_loss = nn.MSELoss()(pred_values.squeeze(), values)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        return total_loss.item()
        
    def update_inference_server(self):
        """Update model on inference server"""
        # Convert model weights to NumPy arrays to avoid CUDA serialization issues
        state_dict = {}
        for key, value in self.model.state_dict().items():
            state_dict[key] = value.cpu().numpy()
            
        # Update server
        ray.get(self.inference_actor.update_model.remote(state_dict))
        
    def save_checkpoint(self, name="model"):
        """Save model checkpoint"""
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'game_count': self.game_count
        }, path)
        
        if VERBOSE:
            print(f"Saved checkpoint: {path}")
            
    def load_checkpoint(self, name="model"):
        """Load model checkpoint"""
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.game_count = checkpoint['game_count']
            
            # Update inference server
            self.update_inference_server()
            
            if VERBOSE:
                print(f"Loaded checkpoint: {path}")
                
    def train(self, num_games=1000):
        """Main training loop"""
        # Start tracking time
        start_time = time.time()
        train_start_time = time.time()
        
        # Determine how many games to run in parallel
        # We want to balance parallelism without overloading the system
        num_cores = multiprocessing.cpu_count()
        parallel_games = min(8, max(1, num_cores // 3))
        
        if VERBOSE:
            print(f"Running {parallel_games} games in parallel")
        
        # Main training loop
        games_completed = 0
        while games_completed < num_games:
            # Generate a batch of games in parallel
            batch_size = min(parallel_games, num_games - games_completed)
            
            # Generate games
            for _ in range(batch_size):
                self.generate_game(self.game_count)
                self.game_count += 1
                games_completed += 1
            
            # Train on replay buffer
            loss = self.train_batch()
            if loss is not None:
                self.writer.add_scalar('Training/Loss', loss, self.game_count)
            
            # Update inference server periodically
            if self.game_count % 5 == 0:
                self.update_inference_server()
                
            # Save checkpoint periodically
            if self.game_count % 50 == 0:
                self.save_checkpoint(f"model_{self.game_count}")
                
            # Print stats periodically
            if self.game_count % 10 == 0 or games_completed == num_games:
                elapsed = time.time() - start_time
                total_elapsed = time.time() - train_start_time
                games_per_hour = games_completed / (total_elapsed / 3600)
                buffer_size = len(self.replay_buffer)
                
                # Calculate win rates
                total_games = sum(self.win_rates.values())
                if total_games > 0:
                    win_rate_p1 = self.win_rates.get(1, 0) / total_games * 100
                    win_rate_p2 = self.win_rates.get(-1, 0) / total_games * 100
                    draw_rate = self.win_rates.get(0, 0) / total_games * 100
                else:
                    win_rate_p1, win_rate_p2, draw_rate = 0, 0, 0
                
                print(f"Game {self.game_count}: " +
                      f"buffer={buffer_size}, " +
                      f"loss={loss:.4f if loss else 'N/A'}, " +
                      f"rate={games_per_hour:.1f} games/hr, " +
                      f"P1 wins: {win_rate_p1:.1f}%, " +
                      f"P2 wins: {win_rate_p2:.1f}%, " +
                      f"Draws: {draw_rate:.1f}%")
                
                # Reset tracking
                start_time = time.time()
                
        # Save final model
        self.save_checkpoint("model_final")
        self.writer.close()

if __name__ == "__main__":
    # Create manager and run training
    manager = SelfPlayManager()
    try:
        manager.train(num_games=1000)
    finally:
        # Ensure clean shutdown
        ray.shutdown()