# train/self_play.py
"""
Self-play manager for AlphaZero-style training.
"""
import os
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import multiprocessing

from config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_SIMULATIONS,
    NUM_WORKERS, NUM_PARALLEL_GAMES, TEMPERATURE_SCHEDULE,
    RAY_OBJECT_STORE_MEMORY, RAY_HEAP_MEMORY,
    INFERENCE_SERVER_BATCH_WAIT, INFERENCE_SERVER_CACHE_SIZE, 
    INFERENCE_SERVER_MAX_BATCH_SIZE, MCTS_BATCH_SIZE
)
from model import SmallResNet
from utils.state_utils import TicTacToeState
from utils.mcts_utils import apply_temperature, visits_to_policy, get_temperature
from inference.batch_inference_server import BatchInferenceServer
# Import updated MCTS implementations
from mcts import (
    batched_mcts_search, parallel_mcts, 
    BatchMCTSWorker, mcts_with_timeout
)
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer

class SelfPlayManager:
    """Manager for self-play game generation and model improvement"""
    
    def __init__(self, use_parallel_mcts=True, enable_time_based_search=False, 
                max_search_time=None, verbose=False, max_workers=4):
        """
        Initialize the self-play manager
        
        Args:
            use_parallel_mcts: Whether to use root parallelization with multiple workers
            enable_time_based_search: Whether to use time-based search instead of simulation count
            max_search_time: Maximum search time in seconds (if time-based search is enabled)
            verbose: Whether to print detailed information during search
            max_workers: Maximum number of parallel workers (lower is more stable)
        """
        # Store MCTS configuration
        self.use_parallel_mcts = use_parallel_mcts
        self.enable_time_based_search = enable_time_based_search
        self.max_search_time = max_search_time or 1.0  # Default to 1 second
        self.verbose = verbose
        self.max_workers = max_workers
        
        # Initialize Ray with proper CPU and memory configuration
        if ray.is_initialized():
            ray.shutdown()
        
        # Use fewer CPUs for improved stability    
        num_cpus = min(multiprocessing.cpu_count() - 1, max_workers + 2)  # Reserve cores for system
        ray.init(
            num_cpus=num_cpus,
            object_store_memory=RAY_OBJECT_STORE_MEMORY // 2,  # Use less memory for stability
            _memory=RAY_HEAP_MEMORY // 2,  # Use less memory for stability
            include_dashboard=False  # Disable dashboard for stability
        )
            
        # Set device and enable mixed precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Enable automatic mixed precision for faster training
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == 'cuda') if self.device.type == 'cuda' else None
        
        # Initialize model
        self.model = SmallResNet().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            replay_buffer=self.replay_buffer,
            device=self.device,
            scaler=self.scaler,
            log_dir='runs/self_play'
        )
        
        # Initialize inference server actor with optimized parameters
        self.inference_actor = BatchInferenceServer.remote(
            batch_wait=INFERENCE_SERVER_BATCH_WAIT,
            cache_size=INFERENCE_SERVER_CACHE_SIZE,
            max_batch_size=INFERENCE_SERVER_MAX_BATCH_SIZE
        )
        
        # Game counter and metrics
        self.game_count = 0
        self.win_rates = {1: 0, -1: 0, 0: 0}  # Player 1, Player -1, Draw
        
        # Track failures to automatically switch methods if needed
        self.parallel_failures = 0
        self.max_failures = 3  # After this many failures, switch to batched
        
        search_method = "time-based" if self.enable_time_based_search else ("task-based parallel" if self.use_parallel_mcts else "batched")
        print(f"SelfPlayManager initialized with {search_method} MCTS")
        if self.enable_time_based_search:
            print(f"Using time-based search with {self.max_search_time}s per move")
        if self.use_parallel_mcts:
            print(f"Using maximum {self.max_workers} workers for parallelization")
        
    def perform_search(self, state, temperature):
        """
        Perform MCTS search using the configured search method
        
        Args:
            state: Current game state
            temperature: Temperature parameter for action selection
            
        Returns:
            tuple: (action, policy_tensor, root) - Selected action, policy tensor, and root node
        """
        try:
            # Choose search method based on configuration
            if self.enable_time_based_search:
                # Use time-based search
                root = mcts_with_timeout(
                    state, 
                    self.inference_actor, 
                    max_time_seconds=self.max_search_time,
                    batch_size=MCTS_BATCH_SIZE,
                    exploration_weight=1.4  # Can be added to config
                )
                
                # Extract visit counts
                visits = np.array([child.visits for child in root.children])
                actions = [child.action for child in root.children]
                
                # Apply temperature and select action
                action, probs = apply_temperature(visits, actions, temperature)
                
                # Create full policy vector
                policy = visits_to_policy(visits, actions, board_size=9)  # Fixed size for TicTacToe
                
            elif self.use_parallel_mcts:
                try:
                    # Use task-based parallel MCTS with limited workers
                    action, policy = parallel_mcts(
                        state,
                        self.inference_actor,
                        NUM_SIMULATIONS,
                        num_workers=self.max_workers,  # Use limited workers for stability
                        temperature=temperature,
                        return_action_probs=True
                    )
                    root = None  # Root not returned from parallel_mcts
                    
                    # Reset failure counter on success
                    self.parallel_failures = 0
                    
                except Exception as e:
                    print(f"Task-based parallel MCTS failed: {e}")
                    self.parallel_failures += 1
                    
                    # If we've had too many failures, switch to batched search permanently
                    if self.parallel_failures >= self.max_failures:
                        print(f"Too many parallel failures ({self.parallel_failures}), switching to batched search permanently")
                        self.use_parallel_mcts = False
                    
                    # Fall back to batched search for this move
                    root = batched_mcts_search(
                        state,
                        self.inference_actor,
                        NUM_SIMULATIONS,
                        batch_size=MCTS_BATCH_SIZE,
                        verbose=self.verbose
                    )
                    
                    # Extract visit counts
                    visits = np.array([child.visits for child in root.children])
                    actions = [child.action for child in root.children]
                    
                    # Apply temperature and select action
                    action, probs = apply_temperature(visits, actions, temperature)
                    
                    # Create full policy vector
                    policy = visits_to_policy(visits, actions, board_size=9)  # Fixed size for TicTacToe
                
            else:
                # Use batched MCTS with a single root
                root = batched_mcts_search(
                    state,
                    self.inference_actor,
                    NUM_SIMULATIONS,
                    batch_size=MCTS_BATCH_SIZE,
                    verbose=self.verbose
                )
                
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
            print(f"Search method failed completely: {e}")
            print("Falling back to random action selection")
            
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
        
    def generate_game(self, game_id):
        """
        Generate a single self-play game
        
        Args:
            game_id: Unique identifier for the game
            
        Returns:
            tuple: (outcome, move_count) - The game result and number of moves played
        """
        start_time = time.time()
        
        state = TicTacToeState()
        memory = []
        move_count = 0
        
        # Play until game is finished
        while not state.is_terminal():
            # Determine temperature based on move count
            temperature = get_temperature(move_count, TEMPERATURE_SCHEDULE)
            
            # Perform search
            action, policy_tensor, _ = self.perform_search(state, temperature)
            
            # Convert state to tensor
            state_tensor = torch.tensor(state.board, dtype=torch.float).to(self.device)
            
            # Store experience
            memory.append((state_tensor, policy_tensor, state.current_player))
            
            # Apply action
            state = state.apply_action(action)
            move_count += 1
        
        # Game finished
        outcome = state.get_winner()
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
        
    def generate_games_parallel(self, num_games):
        """
        Generate multiple games in parallel
        
        Args:
            num_games: Number of games to generate
            
        Returns:
            list: Results from all games
        """
        futures = []
        
        # Launch games in parallel
        for i in range(min(num_games, NUM_PARALLEL_GAMES)):
            game_id = self.game_count + i
            futures.append(self.generate_game(game_id))
        
        # Get results
        results = []
        for future in futures:
            results.append(future)
            
        return results
        
    def update_inference_server(self):
        """Update model on inference server"""
        # Convert model weights to NumPy arrays to avoid CUDA serialization issues
        state_dict = {}
        for key, value in self.model.state_dict().items():
            state_dict[key] = value.cpu().numpy()
            
        # Update server
        ray.get(self.inference_actor.update_model.remote(state_dict))
        
    def save_checkpoint(self, name="model"):
        """
        Save model checkpoint with game metadata
        
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
            
    def load_checkpoint(self, name="model"):
        """
        Load model checkpoint and metadata
        
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
                
    def train(self, num_games=100):
        """
        Main training loop with parallel game generation
        
        Args:
            num_games: Number of games to play
        """
        # Start tracking time
        start_time = time.time()
        train_start_time = time.time()
        
        # Main training loop
        games_completed = 0
        while games_completed < num_games:
            # Determine number of games to generate
            games_to_generate = min(NUM_PARALLEL_GAMES, num_games - games_completed)
            
            # Generate games in parallel
            for i in range(games_to_generate):
                self.generate_game(self.game_count + i)
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
                    loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                    print(f"  Loss: {loss_str}, LR: {self.trainer.get_learning_rate():.2e}")
                    print(f"  Rate: {games_per_hour:.1f} games/hr")
                    print(f"  Win rates: P1={win_rate_p1:.1f}%, P2={win_rate_p2:.1f}%, Draw={draw_rate:.1f}%")
                    
                    # Reset tracking time
                    start_time = time.time()
                
        # Save final model
        self.save_checkpoint("model_final")
        
        # Close trainer
        self.trainer.close()
        
        # Print final stats
        print("\nTraining completed!")
        print(f"Total games: {num_games}")
        print(f"Time: {(time.time() - train_start_time) / 60:.1f} minutes")
        print(f"Games per hour: {games_completed / ((time.time() - train_start_time) / 3600):.1f}")