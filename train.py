# train.py - Simplified Self-Play Manager
import os
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from model import SmallResNet
from inference.inference_server import InferenceServer
from utils.state_utils import TicTacToeState
from mcts.search import parallel_mcts
from config import NUM_SIMULATIONS, NUM_WORKERS, SIMULATIONS_PER_WORKER, VERBOSE

# Training parameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100000
CHECKPOINT_DIR = 'checkpoints'

class ReplayBuffer:
    """Simple experience replay buffer"""
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        indices = np.random.choice(len(self.buffer), batch_size)
        states, policies, values = [], [], []
        
        for idx in indices:
            state, policy, value = self.buffer[idx]
            states.append(state)
            policies.append(policy)
            values.append(value)
            
        return (
            torch.stack(states),
            torch.stack(policies),
            torch.tensor(values, dtype=torch.float)
        )
        
    def __len__(self):
        return len(self.buffer)

class SelfPlayManager:
    """Manager for self-play, training and evaluation"""
    
    def __init__(self):
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init()
            
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SmallResNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Initialize inference server actor
        self.inference_actor = InferenceServer.remote(batch_wait=0.02)
        
        # Initialize logging
        self.writer = SummaryWriter()
        
        # Game counter
        self.game_count = 0
        
    def generate_game(self):
        """Generate a single self-play game"""
        state = TicTacToeState()
        memory = []
        move_count = 0
        
        # Play until game is finished
        while not state.is_terminal():
            # Use parallel MCTS to get action and policy
            action, policy = parallel_mcts(
                state, 
                self.inference_actor, 
                NUM_SIMULATIONS * SIMULATIONS_PER_WORKER,
                NUM_WORKERS, 
                temperature=1.0 if move_count < 10 else 0.5
            )
            
            # Convert to tensors
            state_tensor = torch.tensor(state.board, dtype=torch.float).to(self.device)
            policy_tensor = torch.tensor(policy, dtype=torch.float).to(self.device)
            
            # Store experience
            memory.append((state_tensor, policy_tensor, state.current_player))
            
            # Apply action
            state = state.apply_action(action)
            move_count += 1
            
            if VERBOSE and move_count % 5 == 0:
                print(f"Move {move_count}")
        
        # Game finished
        outcome = state.winner
        
        if VERBOSE:
            print(f"Game finished after {move_count} moves with outcome: {outcome}")
        
        # Add experiences to replay buffer
        for state_tensor, policy_tensor, player in memory:
            # Set target value based on game outcome
            if outcome == 0:  # Draw
                target_value = 0.0
            else:
                target_value = 1.0 if outcome == player else -1.0
                
            # Add to buffer
            self.replay_buffer.add((state_tensor, policy_tensor, target_value))
            
        return outcome, move_count
        
    def train_batch(self):
        """Train on a batch of experiences"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None
            
        # Sample batch
        states, policies, values = self.replay_buffer.sample(BATCH_SIZE)
        
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
        games_trained = 0
        
        for i in range(num_games):
            # Generate a self-play game
            outcome, moves = self.generate_game()
            self.game_count += 1
            games_trained += 1
            
            # Log game stats
            self.writer.add_scalar('Game/Outcome', outcome, self.game_count)
            self.writer.add_scalar('Game/Length', moves, self.game_count)
            
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
            if self.game_count % 10 == 0:
                elapsed = time.time() - start_time
                games_per_hour = games_trained / (elapsed / 3600)
                buffer_size = len(self.replay_buffer)
                
                print(f"Game {self.game_count}: " +
                      f"outcome={outcome}, moves={moves}, " +
                      f"buffer={buffer_size}, " +
                      f"loss={loss:.4f if loss else 'N/A'}, " +
                      f"rate={games_per_hour:.1f} games/hr")
                
                # Reset tracking
                start_time = time.time()
                games_trained = 0
                
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