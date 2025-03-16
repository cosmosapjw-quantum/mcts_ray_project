# train/enhanced_self_play.py
"""
Enhanced self-play manager with leaf parallelization and optimized components.
"""
import ray
import time
import logging
import numpy as np
import torch
from typing import Tuple, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedSelfPlay")

# Import optimized components
from utils.improved_ray_manager import RayActorManager, inference_server_health_check
from inference.enhanced_batch_inference_server import EnhancedBatchInferenceServer
from utils.optimized_state import OptimizedTicTacToeState
from mcts.leaf_parallel_mcts import leaf_parallel_search
from mcts.node import Node

# Import standard components that we'll keep
from model import SmallResNet
from utils.mcts_utils import apply_temperature, visits_to_policy, get_temperature
from utils.improved_ray_manager import create_manager_with_inference_server
from train.replay_buffer import ReplayBuffer  # Assuming this is already efficient
from train.trainer import Trainer  # Assuming this is already efficient

class EnhancedSelfPlayManager:
    """
    Enhanced manager for self-play game generation and model improvement.
    
    This implementation uses:
    - Leaf parallelization MCTS for efficient tree search
    - Optimized state representation for minimal overhead
    - Enhanced batch inference server for maximum GPU utilization
    - Proper Ray configuration and actor management
    """
    
    def __init__(self,
                 # Search configuration
                 num_simulations=800,
                 num_collectors=2,
                 batch_size=32,
                 exploration_weight=1.4,
                 temperature_schedule=None,
                 
                 # Server configuration
                 inference_server_batch_wait=0.001,
                 inference_server_cache_size=20000,
                 inference_server_max_batch_size=256,
                 
                 # Training configuration
                 learning_rate=3e-4,
                 weight_decay=1e-4,
                 replay_buffer_size=500000,
                 
                 # System configuration
                 use_gpu=True,
                 cpu_limit=None,
                 gpu_fraction=1.0,
                 use_mixed_precision=True,
                 verbose=False):
        """
        Initialize the enhanced self-play manager.
        
        Args:
            num_simulations: Number of MCTS simulations per move
            num_collectors: Number of leaf collector threads
            batch_size: Batch size for MCTS leaf evaluation
            exploration_weight: Exploration constant for PUCT
            temperature_schedule: Schedule for temperature reduction
            inference_server_batch_wait: Wait time for inference batching
            inference_server_cache_size: Size of inference cache
            inference_server_max_batch_size: Maximum batch size for inference
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            replay_buffer_size: Maximum size of replay buffer
            use_gpu: Whether to use GPU
            cpu_limit: Limit on CPU cores to use (None = auto-detect)
            gpu_fraction: Fraction of GPU to allocate
            use_mixed_precision: Whether to use mixed precision
            verbose: Whether to print detailed information
        """
        self.num_simulations = num_simulations
        self.num_collectors = num_collectors
        self.batch_size = batch_size
        self.exploration_weight = exploration_weight
        self.temperature_schedule = temperature_schedule or {
            0: 1.0,
            15: 0.5,
            30: 0.25
        }
        self.verbose = verbose
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        
        # Create Ray manager
        self.ray_manager, self.inference_server = create_manager_with_inference_server(
            use_gpu=use_gpu, 
            batch_wait=inference_server_batch_wait,
            cache_size=inference_server_cache_size, 
            max_batch_size=inference_server_max_batch_size,
            cpu_limit=cpu_limit,
            gpu_fraction=gpu_fraction,
            use_mixed_precision=use_mixed_precision
        )
        
        # Initialize model
        self.model = SmallResNet().to(self.device)
        
        # Enable automatic mixed precision for faster training
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            replay_buffer=self.replay_buffer,
            device=self.device,
            scaler=self.scaler,
            log_dir='runs/enhanced_self_play'
        )
        
        if not self.inference_server:
            raise RuntimeError("Failed to create inference server")
        
        # Update model on inference server
        self.update_inference_server()
        
        # Game counter and metrics
        self.game_count = 0
        self.win_rates = {1: 0, -1: 0, 0: 0}  # Player 1, Player -1, Draw
        
        logger.info("EnhancedSelfPlayManager initialized")
        logger.info(f"MCTS configuration: simulations={num_simulations}, " +
                   f"collectors={num_collectors}, batch_size={batch_size}")
    
    def inference_function(self, state_or_batch):
        """
        Wrapper function for neural network inference with proper Ray handling.
        
        Args:
            state_or_batch: Single state or batch of states
                
        Returns:
            For single state: (policy, value) tuple
            For batch: List of (policy, value) tuples
        """
        # Track inference attempt count
        attempt_count = getattr(self, '_inference_attempt_count', 0) + 1
        setattr(self, '_inference_attempt_count', attempt_count)
        
        try:
            # Limit nested attempts
            if attempt_count > 3:
                logger.warning("Too many nested inference attempts, using fallback")
                setattr(self, '_inference_attempt_count', 0)
                if isinstance(state_or_batch, list):
                    return [(np.ones(9)/9, 0.0) for _ in state_or_batch]
                else:
                    return (np.ones(9)/9, 0.0)
                    
            # If server not available, try to recreate
            if not self.inference_server:
                if not hasattr(self, '_last_recreation_attempt') or time.time() - self._last_recreation_attempt > 60.0:
                    logger.info("Inference server unavailable, attempting recreation")
                    self._recreate_inference_server()
                else:
                    logger.warning("Inference server unavailable, using fallback (retry later)")
                
                # Return uniform values as fallback
                if isinstance(state_or_batch, list):
                    return [(np.ones(9)/9, 0.0) for _ in state_or_batch]
                else:
                    return (np.ones(9)/9, 0.0)
            
            # Check if it's a batch or single state
            if isinstance(state_or_batch, list):
                # Batch inference with timeout and error handling
                try:
                    batch_result = ray.get(self.inference_server.batch_infer.remote(state_or_batch), timeout=10.0)
                    setattr(self, '_inference_attempt_count', 0)  # Reset counter on success
                    return batch_result
                except (ray.exceptions.GetTimeoutError, ray.exceptions.RayActorError) as e:
                    logger.warning(f"Batch inference timeout or actor failure: {e}. Using fallback.")
                    
                    # Try to recreate server and retry ONE more time
                    if self._recreate_inference_server():
                        try:
                            logger.info("Retrying batch inference after server recreation")
                            return self.inference_function(state_or_batch)  # Recursive retry
                        except Exception:
                            pass
                            
                    # Return uniform policies and neutral values as fallback
                    return [(np.ones(9)/9, 0.0) for _ in state_or_batch]
            else:
                # Single state inference with timeout and error handling
                try:
                    single_result = ray.get(self.inference_server.infer.remote(state_or_batch), timeout=5.0)
                    setattr(self, '_inference_attempt_count', 0)  # Reset counter on success
                    return single_result
                except (ray.exceptions.GetTimeoutError, ray.exceptions.RayActorError) as e:
                    logger.warning(f"Single inference timeout or actor failure: {e}. Using fallback.")
                    
                    # Try to recreate server and retry ONE more time
                    if self._recreate_inference_server():
                        try:
                            logger.info("Retrying single inference after server recreation")
                            return self.inference_function(state_or_batch)  # Recursive retry
                        except Exception:
                            pass
                    
                    # Return uniform policy and neutral value as fallback
                    return (np.ones(9)/9, 0.0)
        except Exception as e:
            logger.error(f"Unexpected error in inference function: {e}")
            setattr(self, '_inference_attempt_count', 0)  # Reset counter
            
            # Return fallback values
            if isinstance(state_or_batch, list):
                return [(np.ones(9)/9, 0.0) for _ in state_or_batch]
            else:
                return (np.ones(9)/9, 0.0)
    
    def perform_search(self, state, temperature=1.0):
        """
        Perform MCTS search using leaf parallelization with robust error handling.
        
        Args:
            state: Current game state
            temperature: Temperature parameter for action selection
            
        Returns:
            tuple: (action, policy_tensor, root) - Selected action, policy tensor, and root node
        """
        try:
            # Run leaf-parallel search
            search_start = time.time()
            
            root, stats = leaf_parallel_search(
                root_state=state,
                inference_fn=self.inference_function,  # Use the fixed inference function
                num_simulations=self.num_simulations,
                num_collectors=self.num_collectors,
                batch_size=self.batch_size,
                exploration_weight=self.exploration_weight,
                add_dirichlet_noise=True,
                collect_stats=self.verbose
            )
            
            search_time = time.time() - search_start
            
            if self.verbose:
                logger.info(f"Search completed in {search_time:.3f}s with {self.num_simulations} simulations " +
                        f"({self.num_simulations/search_time:.1f} sims/s)")
            
            # Extract visit counts
            visits = np.array([child.visits for child in root.children])
            actions = [child.action for child in root.children]
            
            # Apply temperature and select action
            action, probs = apply_temperature(visits, actions, temperature)
            
            # Create full policy vector
            policy = visits_to_policy(visits, actions, board_size=9)  # Fixed size for TicTacToe
            
            # Convert policy to tensor
            policy_tensor = torch.tensor(policy, dtype=torch.float).to(self.device)
            
            return action, policy_tensor, root
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            
            # Emergency fallback: use uniform random policy
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                raise ValueError("No legal actions available")
                
            # Random action
            action = np.random.choice(legal_actions)
            
            # Uniform policy
            policy = np.zeros(9)  # Fixed size for TicTacToe
            for a in legal_actions:
                policy[a] = 1.0 / len(legal_actions)
                
            # Convert to tensor
            policy_tensor = torch.tensor(policy, dtype=torch.float).to(self.device)
            
            return action, policy_tensor, None
    
    def update_inference_server(self):
        """Update model on inference server with robust error handling"""
        try:
            # Check if server is available
            if not self.inference_server:
                logger.warning("No inference server available to update")
                return self._recreate_inference_server()
                
            # Convert model weights to NumPy arrays to avoid CUDA serialization issues
            state_dict = {}
            for key, value in self.model.state_dict().items():
                state_dict[key] = value.cpu().numpy()
                
            # Update server with timeout
            try:
                ray.get(self.inference_server.update_model.remote(state_dict), timeout=10.0)
                logger.info("Updated model on inference server")
                return True
            except (ray.exceptions.GetTimeoutError, ray.exceptions.RayActorError) as e:
                logger.error(f"Failed to update inference server: {e}")
                # Try to recreate the server
                return self._recreate_inference_server()
        except Exception as e:
            logger.error(f"Unexpected error updating inference server: {e}")
            return False
            
    def _recreate_inference_server(self):
        """Recreate the inference server if it's dead or unresponsive"""
        try:
            logger.info("Attempting to recreate inference server")
            
            # Track attempt time
            self._last_recreation_attempt = time.time()
            
            # Kill old server if it exists (ignore errors)
            if self.inference_server:
                try:
                    ray.kill(self.inference_server)
                except Exception as e:
                    logger.debug(f"Error killing existing inference server: {e}")
                    
                # Wait to ensure it's gone
                time.sleep(2.0)
            
            # Create with maximum wait time
            from inference.enhanced_batch_inference_server import EnhancedBatchInferenceServer
            
            # Use longer timeouts for creation and model loading
            self.inference_server = EnhancedBatchInferenceServer.options(
                num_cpus=1.0,
                num_gpus=1.0 if self.device.type == 'cuda' else 0.0
            ).remote(
                initial_batch_wait=0.001,
                cache_size=20000,
                max_batch_size=256,
                adaptive_batching=True,
                mixed_precision=self.use_mixed_precision
            )
                
            # Give it time to initialize basic structure
            time.sleep(3.0)
                
            # Update model with extended timeout
            state_dict = {}
            for key, value in self.model.state_dict().items():
                state_dict[key] = value.cpu().numpy()
                
            # Use longer timeout (20 seconds)
            ray.get(self.inference_server.update_model.remote(state_dict), timeout=20.0)
            
            logger.info("Successfully recreated inference server")
            return True
        except Exception as e:
            logger.error(f"Failed to recreate inference server: {e}")
            # Set to None to avoid further usage attempts
            self.inference_server = None
            return False
    
    def generate_game(self, game_id):
        """
        Generate a single self-play game with enhanced performance.
        
        Args:
            game_id: Unique identifier for the game
            
        Returns:
            tuple: (outcome, move_count) - The game result and number of moves played
        """
        start_time = time.time()
        
        # Use optimized state implementation
        state = OptimizedTicTacToeState()
        memory = []
        move_count = 0
        
        # Play until game is finished
        while not state.is_terminal():
            # Determine temperature based on move count
            temperature = get_temperature(move_count, self.temperature_schedule)
            
            # Perform search
            action, policy_tensor, _ = self.perform_search(state, temperature)
            
            # Convert state to tensor
            state_tensor = torch.tensor(state.encode(), dtype=torch.float).to(self.device)
            
            # Store experience
            memory.append((state_tensor, policy_tensor, state.current_player))
            
            # Apply action
            state = state.apply_action(action)
            move_count += 1
            
            if self.verbose and move_count % 5 == 0:
                logger.info(f"Game {game_id}, Move {move_count}, Player {state.current_player}")
                logger.info(str(state))
        
        # Game finished
        outcome = state.get_winner()
        game_time = time.time() - start_time
        
        logger.info(f"Game {game_id}: {move_count} moves, outcome={outcome}, time={game_time:.1f}s, buffer={len(self.replay_buffer)}")
        
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
    
    def update_inference_server(self):
        """Update model on inference server"""
        try:
            # Convert model weights to NumPy arrays to avoid CUDA serialization issues
            state_dict = {}
            for key, value in self.model.state_dict().items():
                state_dict[key] = value.cpu().numpy()
                
            # Update server
            import ray
            ray.get(self.inference_server.update_model.remote(state_dict))
            
            logger.info("Updated model on inference server")
            return True
        except Exception as e:
            logger.error(f"Failed to update inference server: {e}")
            return False
    
    def save_checkpoint(self, name="model"):
        """
        Save model checkpoint with game metadata.
        
        Args:
            name: Checkpoint name
        """
        # Additional data to save
        additional_data = {
            'game_count': self.game_count,
            'win_rates': self.win_rates
        }
        
        # Use trainer to save checkpoint
        self.trainer.save_checkpoint(name, additional_data)
        logger.info(f"Saved checkpoint: {name}")
    
    def load_checkpoint(self, name="model"):
        """
        Load model checkpoint and metadata.
        
        Args:
            name: Checkpoint name
        """
        # Use trainer to load checkpoint
        additional_data = self.trainer.load_checkpoint(name)
        
        if additional_data:
            # Load game metadata
            if 'game_count' in additional_data:
                self.game_count = additional_data['game_count']
            
            if 'win_rates' in additional_data:
                self.win_rates = additional_data['win_rates']
            
            # Update inference server
            self.update_inference_server()
            
            logger.info(f"Loaded checkpoint: {name}")
            return True
        
        logger.warning(f"Failed to load checkpoint: {name}")
        return False
    
    def train(self, num_games=100):
        """
        Main training loop with enhanced performance.
        
        Args:
            num_games: Number of games to play
        """
        # Start tracking time
        start_time = time.time()
        
        # Main training loop
        games_completed = 0
        while games_completed < num_games:
            # Generate game
            self.generate_game(self.game_count)
            self.game_count += 1
            games_completed += 1
            
            # Train on replay buffer after each game
            loss = self.trainer.train_batch()
            
            # Update inference server periodically
            if (self.game_count) % 5 == 0:
                self.update_inference_server()
                
            # Save checkpoint periodically
            if (self.game_count) % 20 == 0:
                self.save_checkpoint(f"model_{self.game_count}")
                
            # Print stats periodically
            if (self.game_count) % 5 == 0 or games_completed == num_games:
                elapsed = time.time() - start_time
                games_per_hour = games_completed / (elapsed / 3600) if elapsed > 0 else 0
                buffer_size = len(self.replay_buffer)
                
                # Calculate win rates
                total_games = sum(self.win_rates.values())
                if total_games > 0:
                    win_rate_p1 = self.win_rates.get(1, 0) / total_games * 100
                    win_rate_p2 = self.win_rates.get(-1, 0) / total_games * 100
                    draw_rate = self.win_rates.get(0, 0) / total_games * 100
                else:
                    win_rate_p1 = win_rate_p2 = draw_rate = 0
                
                logger.info(f"\nTraining Summary (Game {self.game_count}):")
                logger.info(f"  Buffer size: {buffer_size}")
                loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                logger.info(f"  Loss: {loss_str}, LR: {self.trainer.get_learning_rate():.2e}")
                logger.info(f"  Rate: {games_per_hour:.1f} games/hr")
                logger.info(f"  Win rates: P1={win_rate_p1:.1f}%, P2={win_rate_p2:.1f}%, Draw={draw_rate:.1f}%")
        
        # Save final model
        self.save_checkpoint("model_final")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        logger.info("\nTraining completed!")
        logger.info(f"Total games: {num_games}")
        logger.info(f"Time: {total_time / 60:.1f} minutes")
        logger.info(f"Games per hour: {num_games / (total_time / 3600):.1f}")
    
    def cleanup(self):
        """Clean up resources with proper error handling"""
        try:
            # Close trainer
            if hasattr(self, 'trainer'):
                try:
                    self.trainer.close()
                except Exception as e:
                    logger.error(f"Error closing trainer: {e}")
            
            # Shutdown Ray manager
            if hasattr(self, 'ray_manager'):
                try:
                    self.ray_manager.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down Ray manager: {e}")
                    
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")