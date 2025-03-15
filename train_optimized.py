# train_optimized.py
"""
Optimized AlphaZero-style training with batch processing
for high GPU utilization and fast training
"""
import os
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import threading
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from model import SmallResNet
from utils.state_utils import TicTacToeState
from inference.batch_inference_server import BatchInferenceServer
from mcts.node import Node

# Training parameters
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128  # Reduced for earlier training
REPLAY_BUFFER_SIZE = 100000
CHECKPOINT_DIR = 'checkpoints'
NUM_SIMULATIONS = 200  # MCTS simulations per move
MCTS_BATCH_SIZE = 16   # Batch size for MCTS evaluations
NUM_WORKERS = 1        # Number of parallel MCTS workers
TEMPERATURE_SCHEDULE = {
    0: 1.0,     # First moves use temperature 1.0
    10: 0.8,    # After 10 moves, temperature drops to 0.8
    20: 0.5,    # After 20 moves, temperature drops to 0.5
}

# MCTS helper functions
def select_node(node, exploration_weight=1.0):
    """Select a leaf node from the tree"""
    path = [node]
    while node.children:
        # Find best child according to UCB formula
        best_score = float('-inf')
        best_child = None
        for child in node.children:
            # PUCT formula
            if child.visits == 0:
                score = float('inf')  # Prioritize unexplored
            else:
                q_value = child.value / child.visits
                u_value = exploration_weight * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        node = best_child
        path.append(node)
    return node, path

def expand_node(node, priors, add_noise=False):
    """Expand a node with actions and their priors"""
    actions = node.state.get_legal_actions()
    if not actions:
        return  # Terminal node
    
    # Add Dirichlet noise for root exploration
    if add_noise and node.parent is None:
        noise = np.random.dirichlet([0.3] * len(actions))
        for i, action in enumerate(actions):
            noisy_prior = 0.75 * priors[action] + 0.25 * noise[i]
            child_state = node.state.apply_action(action)
            child = Node(child_state, node)
            child.prior = noisy_prior
            child.action = action
            node.children.append(child)
    else:
        for action in actions:
            child_state = node.state.apply_action(action)
            child = Node(child_state, node)
            child.prior = priors[action]
            child.action = action
            node.children.append(child)

def backpropagate(path, value):
    """Update statistics in the path"""
    for node in reversed(path):
        node.visits += 1
        node.value += value
        value = -value  # Flip for opponent's perspective

# MCTS search with batching
def mcts_search(root_state, inference_server, num_simulations, batch_size=MCTS_BATCH_SIZE, verbose=False):
    """Perform MCTS search with batch evaluation"""
    # Create root node
    root = Node(root_state)
    
    # Get initial policy and value
    policy, value = ray.get(inference_server.infer.remote(root_state))
    
    # Expand root node
    expand_node(root, policy, add_noise=True)
    root.value = value
    root.visits = 1
    
    # Run simulations
    remaining_sims = num_simulations - 1  # -1 for root expansion
    
    while remaining_sims > 0:
        # Collect leaves for evaluation
        leaves = []
        paths = []
        terminal_leaves = []
        terminal_paths = []
        
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, remaining_sims)
        
        # Select leaves until batch is full
        while len(leaves) + len(terminal_leaves) < current_batch_size:
            leaf, path = select_node(root)
            
            if leaf.state.is_terminal():
                terminal_leaves.append(leaf)
                terminal_paths.append(path)
            else:
                leaves.append(leaf)
                paths.append(path)
                
            # If we've collected enough leaves, or if there are no more unexpanded nodes
            if len(leaves) + len(terminal_leaves) >= current_batch_size:
                break
        
        # Process terminal states immediately
        for leaf, path in zip(terminal_leaves, terminal_paths):
            value = leaf.state.winner if leaf.state.winner is not None else 0
            backpropagate(path, value)
            remaining_sims -= 1
        
        # Process non-terminal leaves with batch inference
        if leaves:
            # Get leaf states
            states = [leaf.state for leaf in leaves]
            if verbose and len(states) > 1:
                print(f"  Batch size: {len(states)}")
            
            # Batch inference
            results = ray.get(inference_server.batch_infer.remote(states))
            
            # Process results
            for leaf, path, result in zip(leaves, paths, results):
                policy, value = result
                expand_node(leaf, policy)
                backpropagate(path, value)
                remaining_sims -= 1
    
    return root

# Replay buffer for experience
class ReplayBuffer:
    """Thread-safe experience replay buffer"""
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
    """Manager for self-play, training and evaluation with optimized batch processing"""
    
    def __init__(self):
        # Initialize Ray if needed
        if ray.is_initialized():
            ray.shutdown()
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
        
        # Initialize inference server actor
        self.inference_actor = BatchInferenceServer.remote(
            batch_wait=0.002,   # Very short wait time
            cache_size=5000,    # Smaller cache to force more network usage
            max_batch_size=64   # Larger batch size to improve GPU utilization
        )
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir='runs/optimized')
        
        # Game counter and metrics
        self.game_count = 0
        self.win_rates = {1: 0, -1: 0, 0: 0}  # Player 1, Player -1, Draw
        
    def generate_game(self, game_id):
        """Generate a single self-play game"""
        start_time = time.time()
        
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
            
            # Perform MCTS with batching
            verbose = (self.game_count < 3 or self.game_count % 10 == 0)  # Show batch info for first few games and occasionally
            root = mcts_search(state, self.inference_actor, NUM_SIMULATIONS, verbose=verbose)
            
            # Convert visit counts to policy
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
            
            # Create full policy vector
            policy = np.zeros(9)  # Fixed size for TicTacToe
            for i, a in enumerate(actions):
                policy[a] = visits[i] / np.sum(visits)
            
            # Convert to tensors
            state_tensor = torch.tensor(state.board, dtype=torch.float).to(self.device)
            policy_tensor = torch.tensor(policy, dtype=torch.float).to(self.device)
            
            # Store experience
            memory.append((state_tensor, policy_tensor, state.current_player))
            
            # Apply action
            state = state.apply_action(action)
            move_count += 1
        
        # Game finished
        outcome = state.winner
        game_time = time.time() - start_time
        
        print(f"Game {game_id}: {move_count} moves, outcome={outcome}, time={game_time:.1f}s, buffer={len(self.replay_buffer)}")
        
        # Update win rate statistics
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
        
    def train_batch(self):
        """Train on a batch of experiences"""
        MIN_BUFFER_SIZE = max(BATCH_SIZE // 2, 32)  # Start training with fewer samples  # Wait for a good amount of samples
        if len(self.replay_buffer) < MIN_BUFFER_SIZE:
            return None
            
        # Sample batch
        batch = self.replay_buffer.sample(BATCH_SIZE)
        if batch is None:
            return None
            
        states, policies, values = batch
        values = values.to(self.device)
        
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
            
            print(f"Loaded checkpoint: {path}")
                
    def train(self, num_games=100):
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        """Main training loop"""
        # Start tracking time
        start_time = time.time()
        train_start_time = time.time()
        
        # Main training loop
        games_completed = 0
        while games_completed < num_games:
            # Generate a game
            self.generate_game(self.game_count)
            self.game_count += 1
            games_completed += 1
            
            # Train on replay buffer
            loss = self.train_batch()
            if loss is not None:
                # Update learning rate scheduler
                scheduler.step(loss)
                # Update learning rate scheduler
                scheduler.step(loss)
                self.writer.add_scalar('Training/Loss', loss, self.game_count)
            
            # Update inference server periodically
            if self.game_count % 5 == 0:
                self.update_inference_server()
                
            # Save checkpoint periodically
            if self.game_count % 20 == 0:
                self.save_checkpoint(f"model_{self.game_count}")
                
            # Print stats periodically
            # Get batch stats from inference server if available
            if self.game_count % 5 == 0 or games_completed == num_games:
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
                
                print(f"\nTraining Summary (Game {self.game_count}):")
                print(f"  Buffer size: {buffer_size}")
                if loss is not None:
                    loss_str = f"{loss:.4f}"
                else:
                    loss_str = "N/A"
                print(f"  Loss: {loss_str}, LR: {scheduler.get_last_lr()[0]:.2e}")
                print(f"  Rate: {games_per_hour:.1f} games/hr")
                print(f"  Win rates: P1={win_rate_p1:.1f}%, P2={win_rate_p2:.1f}%, Draw={draw_rate:.1f}%")
                
                # Reset tracking time
                start_time = time.time()
                
        # Save final model
        self.save_checkpoint("model_final")
        self.writer.close()
        
        # Print final stats
        print("\nTraining completed!")
        print(f"Total games: {num_games}")
        print(f"Time: {(time.time() - train_start_time) / 60:.1f} minutes")
        print(f"Games per hour: {games_completed / ((time.time() - train_start_time) / 3600):.1f}")

if __name__ == "__main__":
    # Create manager and run training
    manager = SelfPlayManager()
    try:
        manager.train(num_games=100)
    finally:
        # Ensure clean shutdown
        ray.shutdown()