# train/async_training.py
"""
Asynchronous training pipeline that separates self-play and model training
into different processes that operate concurrently.
"""

import os
import ray
import time
import torch
import threading
import multiprocessing
import numpy as np
import logging
from queue import Queue
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AsyncTraining")

@ray.remote
class ExperienceCollector:
    """
    Actor that generates self-play games and collects experiences.
    """
    
    def __init__(self, 
                 inference_server, 
                 num_simulations=800,
                 num_collectors=2,
                 batch_size=32,
                 exploration_weight=1.4,
                 temperature_schedule=None,
                 experience_buffer_size=10000):
        """
        Initialize the experience collector.
        
        Args:
            inference_server: Ray actor for neural network inference
            num_simulations: Number of MCTS simulations per move
            num_collectors: Number of leaf collector threads
            batch_size: Batch size for leaf evaluation
            exploration_weight: Exploration constant for PUCT
            temperature_schedule: Schedule for temperature parameter
            experience_buffer_size: Size of local experience buffer
        """
        self.inference_server = inference_server
        self.num_simulations = num_simulations
        self.num_collectors = num_collectors
        self.batch_size = batch_size
        self.exploration_weight = exploration_weight
        self.temperature_schedule = temperature_schedule or {
            0: 1.0,
            15: 0.5,
            30: 0.25
        }
        
        # Local experience buffer
        self.experience_buffer = []
        self.experience_buffer_size = experience_buffer_size
        
        # Game statistics
        self.game_count = 0
        self.win_rates = {1: 0, -1: 0, 0: 0}
        
        # Import modules lazily to avoid issues with Ray serialization
        from mcts.leaf_parallel_mcts import leaf_parallel_search
        from utils.optimized_state import OptimizedTicTacToeState
        
        self.leaf_parallel_search = leaf_parallel_search
        self.OptimizedTicTacToeState = OptimizedTicTacToeState
        
        logger.info("ExperienceCollector initialized")
    
    def inference_function(self, state_or_batch):
        """Wrapper function for neural network inference"""
        if isinstance(state_or_batch, list):
            # Batch inference
            return self.inference_server.batch_infer.remote(state_or_batch)
        else:
            # Single state inference
            return self.inference_server.infer.remote(state_or_batch)
    
    def generate_game(self):
        """
        Generate a single self-play game and collect experiences.
        
        Returns:
            dict: Game statistics
        """
        start_time = time.time()
        
        # Initialize state
        state = self.OptimizedTicTacToeState()
        memory = []
        move_count = 0
        
        # Play until game is finished
        while not state.is_terminal():
            # Determine temperature based on move count
            from utils.mcts_utils import get_temperature
            temperature = get_temperature(move_count, self.temperature_schedule)
            
            # Perform search
            root, stats = self.leaf_parallel_search(
                root_state=state,
                inference_fn=self.inference_function,
                num_simulations=self.num_simulations,
                num_collectors=self.num_collectors,
                batch_size=self.batch_size,
                exploration_weight=self.exploration_weight,
                add_dirichlet_noise=True,
                collect_stats=False
            )
            
            # Extract visit counts for policy
            visits = np.array([child.visits for child in root.children])
            actions = [child.action for child in root.children]
            
            # Apply temperature and select action
            from utils.mcts_utils import apply_temperature, visits_to_policy
            action, _ = apply_temperature(visits, actions, temperature)
            
            # Create full policy vector from visits
            policy = visits_to_policy(visits, actions, board_size=9)
            
            # Store experience as tuple of (state_bytes, policy_array, player)
            memory.append((
                state.to_bytes(),
                policy,
                state.current_player
            ))
            
            # Apply action
            state = state.apply_action(action)
            move_count += 1
        
        # Game finished
        outcome = state.get_winner()
        game_time = time.time() - start_time
        
        # Update statistics
        self.game_count += 1
        self.win_rates[outcome] = self.win_rates.get(outcome, 0) + 1
        
        # Process experiences and add to buffer
        for state_bytes, policy, player in memory:
            # Set target value based on game outcome
            if outcome == 0:  # Draw
                target_value = 0.0
            else:
                target_value = 1.0 if outcome == player else -1.0
            
            # Serialize everything for Ray transfer
            experience = {
                "state": state_bytes,  # Efficient binary representation
                "policy": policy.tobytes(),  # Numpy array as bytes
                "value": target_value,
                "player": player,
                "priority": 2.0 if outcome != 0 else 1.0  # Higher priority for non-draws
            }
            
            # Add to buffer
            self.experience_buffer.append(experience)
            
            # Keep buffer size limited
            if len(self.experience_buffer) > self.experience_buffer_size:
                self.experience_buffer.pop(0)
        
        # Return game statistics
        return {
            "outcome": outcome,
            "moves": move_count,
            "time": game_time,
            "game_id": self.game_count
        }
    
    def get_experiences(self, count=None):
        """
        Get experiences from the buffer.
        
        Args:
            count: Number of experiences to return (None = all)
            
        Returns:
            list: List of experience dictionaries
        """
        if count is None or count >= len(self.experience_buffer):
            # Return all experiences and clear buffer
            experiences = list(self.experience_buffer)
            self.experience_buffer.clear()
        else:
            # Return requested number of experiences
            experiences = self.experience_buffer[:count]
            self.experience_buffer = self.experience_buffer[count:]
            
        return experiences
    
    def get_statistics(self):
        """Get game statistics"""
        total_games = sum(self.win_rates.values())
        if total_games > 0:
            win_rate_p1 = self.win_rates.get(1, 0) / total_games * 100
            win_rate_p2 = self.win_rates.get(-1, 0) / total_games * 100
            draw_rate = self.win_rates.get(0, 0) / total_games * 100
        else:
            win_rate_p1 = win_rate_p2 = draw_rate = 0
            
        return {
            "games_completed": self.game_count,
            "win_rate_p1": win_rate_p1,
            "win_rate_p2": win_rate_p2,
            "draw_rate": draw_rate,
            "buffer_size": len(self.experience_buffer)
        }

@ray.remote
class Trainer:
    """
    Actor that trains the neural network model using collected experiences.
    """
    
    def __init__(self,
                 learning_rate=3e-4,
                 weight_decay=1e-4,
                 batch_size=1024,
                 training_epochs=10,
                 replay_buffer_size=500000,
                 min_buffer_size=1024,
                 use_gpu=True):
        """
        Initialize the trainer.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            batch_size: Batch size for training
            training_epochs: Number of epochs to train per batch
            replay_buffer_size: Maximum size of replay buffer
            min_buffer_size: Minimum buffer size before training
            use_gpu: Whether to use GPU for training
        """
        # Training configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.replay_buffer_size = replay_buffer_size
        self.min_buffer_size = min_buffer_size
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.use_mixed_precision = use_gpu and self.device.type == "cuda"
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        
        # Initialize model
        from model import SmallResNet
        self.model = SmallResNet().to(self.device)
        
        # Enable automatic mixed precision for faster training
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
        
        # Initialize replay buffer
        # We'll implement a simple version here for clarity
        self.experiences = []
        self.priorities = []
        
        # Training metrics
        self.train_steps = 0
        self.total_loss = 0
        self.last_loss = 0
        
        logger.info("Trainer ready")
    
    def add_experiences(self, experiences):
        """
        Add experiences to the replay buffer.
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            int: Number of experiences added
        """
        count = 0
        
        for exp in experiences:
            try:
                # Deserialize state, policy, and value
                state_bytes = exp["state"]
                policy_bytes = exp["policy"]
                value = exp["value"]
                priority = exp["priority"]
                
                # Reconstruct state and policy
                from utils.optimized_state import OptimizedTicTacToeState
                state = OptimizedTicTacToeState.from_bytes(state_bytes)
                policy = np.frombuffer(policy_bytes, dtype=np.float32).reshape(9)
                
                # Convert state to tensor
                state_tensor = torch.tensor(state.encode(), dtype=torch.float32)
                policy_tensor = torch.tensor(policy, dtype=torch.float32)
                value_tensor = torch.tensor(value, dtype=torch.float32)
                
                # Add to replay buffer
                self.experiences.append((state_tensor, policy_tensor, value_tensor))
                self.priorities.append(priority)
                count += 1
                
                # Trim buffer if needed
                if len(self.experiences) > self.replay_buffer_size:
                    # Remove lowest priority experience
                    idx = np.argmin(self.priorities)
                    self.experiences.pop(idx)
                    self.priorities.pop(idx)
                    
            except Exception as e:
                logger.error(f"Error adding experience: {e}")
                
        return count
    
    def train_batch(self):
        """
        Train on a batch of experiences.
        
        Returns:
            float or None: Average loss for the batch, or None if no training occurred
        """
        if len(self.experiences) < self.min_buffer_size:
            return None
            
        import torch.nn as nn
        
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        
        # Train for multiple epochs on the same batch
        for epoch in range(self.training_epochs):
            # Sample batch based on priorities
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.experiences), 
                                     min(self.batch_size, len(self.experiences)), 
                                     p=probs, replace=False)
            
            # Get batch
            batch = [self.experiences[i] for i in indices]
            states, policies, values = zip(*batch)
            
            # Stack tensors
            states = torch.stack(states).to(self.device)
            policies = torch.stack(policies).to(self.device)
            values = torch.stack(values).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Use mixed precision if available
            if self.scaler:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    pred_policies, pred_values = self.model(states)
                    
                    # Calculate loss
                    policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / policies.size(0)
                    value_loss = nn.MSELoss()(pred_values.squeeze(), values)
                    loss = policy_loss + value_loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                pred_policies, pred_values = self.model(states)
                
                # Calculate loss
                policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / policies.size(0)
                value_loss = nn.MSELoss()(pred_values.squeeze(), values)
                loss = policy_loss + value_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / self.training_epochs if self.training_epochs > 0 else 0
        
        # Update learning rate scheduler
        if avg_loss > 0:
            self.scheduler.step(avg_loss)
        
        # Update training metrics
        self.train_steps += 1
        self.total_loss += total_loss
        self.last_loss = avg_loss
        
        return {
            "loss": avg_loss,
            "policy_loss": policy_loss_total / self.training_epochs,
            "value_loss": value_loss_total / self.training_epochs,
            "learning_rate": self.get_learning_rate()
        }
    
    def get_learning_rate(self):
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0]
    
    def get_model_state(self):
        """Get model state dict as numpy arrays for transmission"""
        state_dict = {}
        for key, value in self.model.state_dict().items():
            state_dict[key] = value.cpu().numpy()
        return state_dict
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        return True
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            # Load training steps
            if 'train_steps' in checkpoint:
                self.train_steps = checkpoint['train_steps']
                
            # Load scaler state if available
            if self.scaler and 'scaler' in checkpoint and checkpoint['scaler']:
                self.scaler.load_state_dict(checkpoint['scaler'])
                
            logger.info(f"Loaded checkpoint: {path}")
            return True
        
        logger.warning(f"Checkpoint not found: {path}")
        return False
    
    def get_stats(self):
        """Get training statistics"""
        return {
            "train_steps": self.train_steps,
            "last_loss": self.last_loss,
            "total_loss": self.total_loss,
            "learning_rate": self.get_learning_rate(),
            "buffer_size": len(self.experiences)
        }

class AsyncTrainingPipeline:
    """
    Asynchronous training pipeline that separates self-play and training.
    """
    
    def __init__(self,
                 num_collectors=2,
                 checkpoint_dir="checkpoints",
                 checkpoint_freq=20,
                 model_update_freq=5,
                 experience_collection_interval=0.5,
                 logging_interval=5.0,
                 use_gpu=True):
        """
        Initialize the asynchronous training pipeline.
        
        Args:
            num_collectors: Number of parallel experience collectors
            checkpoint_dir: Directory for model checkpoints
            checkpoint_freq: Frequency of checkpoints (in training steps)
            model_update_freq: Frequency of model updates (in training steps)
            experience_collection_interval: Time between experience collection (seconds)
            logging_interval: Time between logging (seconds)
            use_gpu: Whether to use GPU
        """
        self.num_collectors = num_collectors
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.model_update_freq = model_update_freq
        self.experience_collection_interval = experience_collection_interval
        self.logging_interval = logging_interval
        self.use_gpu = use_gpu
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Control flags
        self.shutdown_flag = threading.Event()
        self.collectors = []
        self.trainer = None
        self.inference_server = None
        
        # Statistics
        self.start_time = None
        self.game_count = 0
        self.experience_count = 0
        
        logger.info("AsyncTrainingPipeline initialized")
    
    def setup(self):
        """Set up the training pipeline components"""
        try:
            # Create inference server
            from inference.enhanced_batch_inference_server import EnhancedBatchInferenceServer
            
            self.inference_server = EnhancedBatchInferenceServer.remote(
                initial_batch_wait=0.001,
                cache_size=20000,
                max_batch_size=256,
                adaptive_batching=True,
                mixed_precision=self.use_gpu
            )
            
            # Create trainer
            self.trainer = Trainer.remote(
                learning_rate=3e-4,
                weight_decay=1e-4,
                batch_size=1024,
                training_epochs=10,
                replay_buffer_size=500000,
                min_buffer_size=1024,
                use_gpu=self.use_gpu
            )
            
            # Get initial model state
            model_state = ray.get(self.trainer.get_model_state.remote())
            
            # Update inference server with initial model
            ray.get(self.inference_server.update_model.remote(model_state))
            
            # Create experience collectors
            self.collectors = []
            for i in range(self.num_collectors):
                collector = ExperienceCollector.remote(
                    inference_server=self.inference_server,
                    num_simulations=800,
                    num_collectors=2,
                    batch_size=32,
                    exploration_weight=1.4
                )
                self.collectors.append(collector)
            
            logger.info(f"Set up {self.num_collectors} experience collectors")
            
            return True
        except Exception as e:
            logger.error(f"Error setting up pipeline: {e}")
            return False
    
    def start(self, load_checkpoint=None):
        """
        Start the asynchronous training pipeline.
        
        Args:
            load_checkpoint: Path to checkpoint to load (optional)
        """
        if not self.setup():
            logger.error("Failed to set up pipeline")
            return
            
        # Load checkpoint if provided
        if load_checkpoint:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{load_checkpoint}.pt")
            ray.get(self.trainer.load_checkpoint.remote(checkpoint_path))
            
            # Update inference server with loaded model
            model_state = ray.get(self.trainer.get_model_state.remote())
            ray.get(self.inference_server.update_model.remote(model_state))
            
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        # Start timing
        self.start_time = time.time()
        
        # Start worker threads
        self.experience_thread = threading.Thread(
            target=self._experience_collection_worker,
            daemon=True
        )
        self.logging_thread = threading.Thread(
            target=self._logging_worker,
            daemon=True
        )
        
        self.experience_thread.start()
        self.logging_thread.start()
        
        logger.info("Training pipeline started")
    
    def _experience_collection_worker(self):
        """Worker thread for collecting experiences"""
        while not self.shutdown_flag.is_set():
            try:
                # Collect experiences from all collectors
                for i, collector in enumerate(self.collectors):
                    # Generate game in background
                    collector.generate_game.remote()
                    
                    # Get available experiences
                    experiences = ray.get(collector.get_experiences.remote(100))
                    
                    if experiences:
                        # Add to trainer
                        count = ray.get(self.trainer.add_experiences.remote(experiences))
                        self.experience_count += count
                        
                        logger.debug(f"Added {count} experiences from collector {i}")
                    
                # Train on collected experiences
                train_result = ray.get(self.trainer.train_batch.remote())
                
                if train_result:
                    logger.debug(f"Training step completed: loss={train_result['loss']:.4f}")
                    
                    # Update model on inference server periodically
                    trainer_stats = ray.get(self.trainer.get_stats.remote())
                    if trainer_stats["train_steps"] % self.model_update_freq == 0:
                        model_state = ray.get(self.trainer.get_model_state.remote())
                        ray.get(self.inference_server.update_model.remote(model_state))
                        
                        logger.info(f"Updated model on inference server (step {trainer_stats['train_steps']})")
                    
                    # Save checkpoint periodically
                    if trainer_stats["train_steps"] % self.checkpoint_freq == 0:
                        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{trainer_stats['train_steps']}.pt")
                        ray.get(self.trainer.save_checkpoint.remote(checkpoint_path))
                        
                        # Also save as latest
                        latest_path = os.path.join(self.checkpoint_dir, "model_latest.pt")
                        ray.get(self.trainer.save_checkpoint.remote(latest_path))
                
                # Sleep before next collection
                time.sleep(self.experience_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in experience collection: {e}")
                import traceback
                traceback.print_exc()
                
                # Sleep to avoid error flooding
                time.sleep(1.0)
    
    def _logging_worker(self):
        """Worker thread for logging statistics"""
        while not self.shutdown_flag.is_set():
            try:
                # Collect statistics from all components
                collector_stats = [ray.get(collector.get_statistics.remote()) 
                                 for collector in self.collectors]
                trainer_stats = ray.get(self.trainer.get_stats.remote())
                
                # Calculate aggregate statistics
                total_games = sum(stats["games_completed"] for stats in collector_stats)
                self.game_count = total_games
                
                # Calculate win rates
                p1_rates = [stats["win_rate_p1"] for stats in collector_stats]
                p2_rates = [stats["win_rate_p2"] for stats in collector_stats]
                draw_rates = [stats["draw_rate"] for stats in collector_stats]
                
                avg_p1_rate = sum(p1_rates) / len(p1_rates) if p1_rates else 0
                avg_p2_rate = sum(p2_rates) / len(p2_rates) if p2_rates else 0
                avg_draw_rate = sum(draw_rates) / len(draw_rates) if draw_rates else 0
                
                # Calculate performance metrics
                elapsed = time.time() - self.start_time
                games_per_hour = total_games / (elapsed / 3600) if elapsed > 0 else 0
                
                # Log statistics
                logger.info("\nTraining Statistics:")
                logger.info(f"  Total games: {total_games}, Experiences: {self.experience_count}")
                logger.info(f"  Rate: {games_per_hour:.1f} games/hr")
                logger.info(f"  Win rates: P1={avg_p1_rate:.1f}%, P2={avg_p2_rate:.1f}%, Draw={avg_draw_rate:.1f}%")
                logger.info(f"  Training: Steps={trainer_stats['train_steps']}, Loss={trainer_stats['last_loss']:.4f}")
                logger.info(f"  Learning rate: {trainer_stats['learning_rate']:.2e}")
                logger.info(f"  Buffer size: {trainer_stats['buffer_size']}")
                
                # Sleep before next logging
                time.sleep(self.logging_interval)
                
            except Exception as e:
                logger.error(f"Error in logging: {e}")
                
                # Sleep to avoid error flooding
                time.sleep(1.0)
    
    def shutdown(self):
        """Shutdown the training pipeline"""
        logger.info("Shutting down training pipeline...")
        
        # Set shutdown flag
        self.shutdown_flag.set()
        
        # Wait for threads to exit
        if hasattr(self, 'experience_thread') and self.experience_thread.is_alive():
            self.experience_thread.join(timeout=1.0)
        
        if hasattr(self, 'logging_thread') and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=1.0)
        
        # Save final checkpoint
        if self.trainer:
            final_path = os.path.join(self.checkpoint_dir, "model_final.pt")
            ray.get(self.trainer.save_checkpoint.remote(final_path))
            
            # Also save as latest
            latest_path = os.path.join(self.checkpoint_dir, "model_latest.pt")
            ray.get(self.trainer.save_checkpoint.remote(latest_path))
        
        # Calculate final statistics
        total_time = time.time() - self.start_time if self.start_time else 0
        
        logger.info("\nTraining completed!")
        logger.info(f"Total games: {self.game_count}")
        logger.info(f"Total experiences: {self.experience_count}")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Games per hour: {self.game_count / (total_time / 3600):.1f}")
        
        logger.info("Pipeline shutdown complete")
        
# Example usage
def run_async_training(num_collectors=2, gpu=True, checkpoint=None, duration_hours=None):
    """
    Run asynchronous training for a specified duration.
    
    Args:
        num_collectors: Number of parallel experience collectors
        gpu: Whether to use GPU
        checkpoint: Checkpoint to load (optional)
        duration_hours: Training duration in hours (None = indefinite)
    """
    # Create pipeline
    pipeline = AsyncTrainingPipeline(
        num_collectors=num_collectors,
        use_gpu=gpu
    )
    
    try:
        # Start pipeline
        pipeline.start(load_checkpoint=checkpoint)
        
        if duration_hours:
            # Sleep for specified duration
            logger.info(f"Running for {duration_hours:.1f} hours")
            time.sleep(duration_hours * 3600)
        else:
            # Run indefinitely until interrupted
            logger.info("Running indefinitely (Ctrl+C to stop)")
            while True:
                time.sleep(60)
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Shutdown pipeline
        pipeline.shutdown()