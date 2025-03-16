# train/patch_self_play.py
"""
Enhanced patching for SelfPlayManager to enable efficient batch size optimization
without Ray dependencies and with detailed performance profiling.
"""

import sys
import os
import time
import torch
import numpy as np
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure that the project directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Performance timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Store timing info in the instance if it's a method
        if args and hasattr(args[0], '_timing_stats'):
            method_name = func.__name__
            execution_time = end_time - start_time
            
            if method_name not in args[0]._timing_stats:
                args[0]._timing_stats[method_name] = []
            
            args[0]._timing_stats[method_name].append(execution_time)
        
        return result
    return wrapper

# Mock BatchInferenceServer for testing without Ray
class MockBatchInferenceServer:
    """Mock version of BatchInferenceServer for testing without Ray"""
    
    def __init__(self, batch_wait=0.005, cache_size=1000, max_batch_size=64):
        self.batch_wait = batch_wait
        self.cache_size = cache_size
        self.max_batch_size = max_batch_size
        self.cache = {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Stats tracking
        self.total_requests = 0
        self.total_batch_requests = 0
        self.total_batches = 0
        self.total_cache_hits = 0
        self.batch_sizes = []
        self.inference_times = []
        
        logger.info(f"MockBatchInferenceServer created on {self.device}")
    
    def infer(self, state):
        """Handle single state inference"""
        self.total_requests += 1
        
        # Check cache
        board_key = str(state.board)
        if board_key in self.cache:
            self.total_cache_hits += 1
            return self.cache[board_key]
        
        # Ensure model is initialized
        if self.model is None:
            self._setup_model()
        
        # Prepare input
        board_array = np.array([state.board], dtype=np.float32)
        inputs = torch.tensor(board_array, dtype=torch.float32).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    policy, value = self.model(inputs)
            else:
                policy, value = self.model(inputs)
        
        # Process results
        policy_np = policy[0].cpu().numpy()
        value_np = value[0].cpu().numpy()
        result = (policy_np, value_np.item())
        
        # Update cache
        self.cache[board_key] = result
        
        # Trim cache if needed
        if len(self.cache) > self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        return result
    
    def batch_infer(self, states):
        """Handle batch inference for multiple states"""
        self.total_batch_requests += 1
        num_states = len(states)
        self.total_requests += num_states
        
        # Track batch size
        self.batch_sizes.append(num_states)
        self.total_batches += 1
        
        # Check cache for each state
        results = [None] * num_states
        uncached_indices = []
        uncached_states = []
        
        for i, state in enumerate(states):
            board_key = str(state.board)
            if board_key in self.cache:
                results[i] = self.cache[board_key]
                self.total_cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_states.append(state)
        
        # Return immediately if all states were cached
        if not uncached_states:
            return results
        
        # Ensure model is initialized
        if self.model is None:
            self._setup_model()
        
        # Prepare batch input
        inference_start = time.time()
        
        # Convert states to tensor
        boards_array = np.array([s.board for s in uncached_states], dtype=np.float32)
        inputs = torch.tensor(boards_array, dtype=torch.float32).to(self.device)
        
        # Perform batch inference
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    policy_batch, value_batch = self.model(inputs)
            else:
                policy_batch, value_batch = self.model(inputs)
        
        # Process results
        policy_np = policy_batch.cpu().numpy()
        value_np = value_batch.cpu().numpy()
        
        self.inference_times.append(time.time() - inference_start)
        
        # Update cache and fill results
        for i, (idx, state) in enumerate(zip(uncached_indices, uncached_states)):
            result = (policy_np[i], value_np[i][0])
            
            # Update cache
            board_key = str(state.board)
            self.cache[board_key] = result
            
            # Fill in result
            results[idx] = result
        
        # Trim cache if needed
        if len(self.cache) > self.cache_size:
            # Remove oldest items
            excess = len(self.cache) - self.cache_size
            for _ in range(excess):
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return results
    
    def update_model(self, state_dict):
        """Update the model weights"""
        # Ensure model is initialized
        if self.model is None:
            self._setup_model()
        
        # Convert NumPy arrays to tensors if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                new_state_dict[k] = torch.tensor(v)
            else:
                new_state_dict[k] = v
        
        # Load new weights
        self.model.load_state_dict(new_state_dict)
        
        # Clear cache after model update
        self.cache.clear()
        
        return True
    
    def _setup_model(self):
        """Initialize the model"""
        from model import SmallResNet
        
        self.model = SmallResNet()
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model initialized on {self.device}")
    
    def get_stats(self):
        """Get server statistics"""
        stats = {
            "total_requests": self.total_requests,
            "total_batch_requests": self.total_batch_requests,
            "total_batches": self.total_batches,
            "total_cache_hits": self.total_cache_hits,
            "cache_hit_rate": self.total_cache_hits / max(1, self.total_requests) * 100,
            "cache_size": len(self.cache)
        }
        
        if self.batch_sizes:
            stats["avg_batch_size"] = sum(self.batch_sizes) / len(self.batch_sizes)
        
        if self.inference_times:
            stats["avg_inference_time"] = sum(self.inference_times) / len(self.inference_times) * 1000  # ms
        
        return stats

# Enhanced MCTS implementations optimized for testing
def enhanced_batched_mcts_search(state, inference_server, num_simulations, batch_size=64, verbose=False):
    """
    Optimized batched MCTS search implementation for testing.
    
    This implementation includes detailed profiling and supports the MockBatchInferenceServer.
    """
    from mcts.node import Node
    from mcts.core import expand_node, backpropagate, select_node
    import time
    
    # Timing stats
    timing = {
        "selection": 0.0,
        "inference": 0.0,
        "expansion": 0.0,
        "backpropagation": 0.0
    }
    
    # Create root node
    root = Node(state)
    
    # Perform root evaluation
    selection_start = time.time()
    policy, value = inference_server.infer(state)
    timing["inference"] += time.time() - selection_start
    
    # Expand root with initial policy
    expansion_start = time.time()
    expand_node(root, policy, add_noise=True)
    timing["expansion"] += time.time() - expansion_start
    
    # Initialize root
    root.value = value
    root.visits = 1
    
    # Run simulations
    remaining_sims = num_simulations - 1  # -1 for root evaluation
    
    while remaining_sims > 0:
        # Collect leaves for evaluation
        leaves = []
        paths = []
        terminal_leaves = []
        terminal_paths = []
        
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, remaining_sims)
        
        # Select leaves until batch is full
        selection_start = time.time()
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
        timing["selection"] += time.time() - selection_start
        
        # Process terminal states
        backprop_start = time.time()
        for leaf, path in zip(terminal_leaves, terminal_paths):
            value = leaf.state.get_winner()
            backpropagate(path, value)
            remaining_sims -= 1
        timing["backpropagation"] += time.time() - backprop_start
        
        # Process non-terminal leaves with batch inference
        if leaves:
            # Get leaf states
            states = [leaf.state for leaf in leaves]
            
            # Perform batch inference
            inference_start = time.time()
            results = inference_server.batch_infer(states)
            timing["inference"] += time.time() - inference_start
            
            # Process results
            expansion_start = time.time()
            for leaf, path, result in zip(leaves, paths, results):
                policy, value = result
                expand_node(leaf, policy)
                timing["expansion"] += time.time() - expansion_start
                
                backprop_start = time.time()
                backpropagate(path, value)
                timing["backpropagation"] += time.time() - backprop_start
                
                remaining_sims -= 1
    
    if verbose:
        # Print timing breakdown
        total_time = sum(timing.values())
        print("MCTS Timing Breakdown:")
        for phase, time_spent in timing.items():
            percentage = (time_spent / total_time) * 100 if total_time > 0 else 0
            print(f"  {phase}: {time_spent:.3f}s ({percentage:.1f}%)")
    
    return root

def patch_self_play_manager():
    """
    Create an enhanced patched version of SelfPlayManager for optimization testing.
    
    This patch:
    1. Replaces Ray actors with local mock objects
    2. Adds detailed performance profiling
    3. Makes batch size and other parameters easily adjustable
    4. Exposes internal statistics for analysis
    """
    from train.enhanced_self_play import SelfPlayManager
    original_init = SelfPlayManager.__init__
    
    def patched_init(self, use_parallel_mcts=False, enable_time_based_search=False, 
                     max_search_time=None, verbose=False, max_workers=4):
        """Patched initialization that avoids Ray dependencies"""
        # Create an object to store timing statistics
        self._timing_stats = {}
        
        # Store configuration
        self.use_parallel_mcts = False  # Force disable parallel MCTS
        self.enable_time_based_search = enable_time_based_search
        self.max_search_time = max_search_time or 1.0
        self.verbose = verbose
        self.max_workers = 1  # Force single worker
        
        # Skip Ray initialization
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Enable mixed precision if on GPU
        self.scaler = None
        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler()
        
        # Load model and optimization modules
        from model import SmallResNet
        import torch.optim as optim
        from train.replay_buffer import ReplayBuffer
        from train.trainer import Trainer
        
        # Initialize model
        self.model = SmallResNet().to(self.device)
        
        # Set default hyperparameters (these can be overridden)
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.batch_size = 64  # MCTS batch size (default)
        
        # Configurable server parameters (can be set after initialization)
        self.inference_server_batch_wait = 0.001
        self.inference_server_cache_size = 10000
        self.inference_server_max_batch_size = 256
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize replay buffer
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
        
        # Create mock inference server instead of Ray actor
        self.inference_actor = MockBatchInferenceServer(
            batch_wait=self.inference_server_batch_wait,
            cache_size=self.inference_server_cache_size,
            max_batch_size=self.inference_server_max_batch_size
        )
        
        # Game counter and metrics
        self.game_count = 0
        self.win_rates = {1: 0, -1: 0, 0: 0}  # Player 1, Player -1, Draw
        
        logger.info("Patched SelfPlayManager initialized (no Ray dependencies)")
    
    # Apply the initialization patch
    SelfPlayManager.__init__ = patched_init
    
    # Patch the perform_search method
    original_perform_search = SelfPlayManager.perform_search
    
    @timing_decorator
    def patched_perform_search(self, state, temperature):
        """Patched search method that uses the enhanced MCTS implementation"""
        # Use the enhanced batched MCTS search
        root = enhanced_batched_mcts_search(
            state,
            self.inference_actor,
            800,  # Use a fixed simulation count for consistency
            batch_size=self.batch_size,  # Use the configurable batch size
            verbose=self.verbose
        )
        
        # Extract visit counts
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
                # Temperature scaling
                visits_temp = visits ** (1.0 / temperature)
                probs = visits_temp / np.sum(visits_temp)
            
            # Sample action
            try:
                idx = np.random.choice(len(actions), p=probs)
                action = actions[idx]
            except ValueError:
                # Fallback to deterministic
                best_idx = np.argmax(visits)
                action = actions[best_idx]
        
        # Create policy tensor
        policy = np.zeros(9)  # Fixed size for TicTacToe
        total_visits = np.sum(visits)
        if total_visits > 0:
            for i, a in enumerate(actions):
                policy[a] = visits[i] / total_visits
                
        policy_tensor = torch.tensor(policy, dtype=torch.float).to(self.device)
        
        return action, policy_tensor, root
    
    # Apply the search patch
    SelfPlayManager.perform_search = patched_perform_search
    
    # Add method to get performance statistics
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        stats = {
            "timing": self._timing_stats,
            "inference_server": self.inference_actor.get_stats() if hasattr(self.inference_actor, "get_stats") else {},
            "batch_size": self.batch_size,
            "device": str(self.device)
        }
        return stats
    
    # Add the new method
    SelfPlayManager.get_performance_stats = get_performance_stats
    
    # Add method to update server configuration
    def update_server_config(self, batch_size=None, cache_size=None, batch_wait=None, max_batch_size=None):
        """Update server configuration parameters"""
        if batch_size is not None:
            self.batch_size = batch_size
            
        if cache_size is not None:
            self.inference_server_cache_size = cache_size
            if hasattr(self.inference_actor, "cache_size"):
                self.inference_actor.cache_size = cache_size
                
        if batch_wait is not None:
            self.inference_server_batch_wait = batch_wait
            if hasattr(self.inference_actor, "batch_wait"):
                self.inference_actor.batch_wait = batch_wait
                
        if max_batch_size is not None:
            self.inference_server_max_batch_size = max_batch_size
            if hasattr(self.inference_actor, "max_batch_size"):
                self.inference_actor.max_batch_size = max_batch_size
                
        logger.info(f"Updated server config - batch_size: {self.batch_size}, "
                   f"cache_size: {self.inference_server_cache_size}, "
                   f"batch_wait: {self.inference_server_batch_wait}, "
                   f"max_batch_size: {self.inference_server_max_batch_size}")
    
    # Add the configuration update method
    SelfPlayManager.update_server_config = update_server_config
    
    logger.info("Patched SelfPlayManager with enhanced profiling and configuration")
    return SelfPlayManager

# Apply the patch when this module is imported
patched_manager = patch_self_play_manager()

# Standalone test function
def test_patched_manager():
    """Test the patched manager with a simple game"""
    from utils.state_utils import TicTacToeState
    
    # Create manager
    manager = patched_manager(verbose=True)
    
    # Set batch size for testing
    manager.update_server_config(batch_size=32)
    
    # Play a test game
    state = TicTacToeState()
    moves = 0
    
    print("Starting test game...")
    
    while not state.is_terminal():
        # Make a move
        action, _, _ = manager.perform_search(state, temperature=1.0)
        
        # Apply action
        state = state.apply_action(action)
        moves += 1
        
        print(f"Move {moves}:")
        print(state)
    
    print(f"Game finished after {moves} moves. Winner: {state.get_winner()}")
    
    # Print performance stats
    stats = manager.get_performance_stats()
    print("\nPerformance Statistics:")
    
    if "timing" in stats:
        avg_search_time = sum(stats["timing"].get("perform_search", [0])) / len(stats["timing"].get("perform_search", [1]))
        print(f"Average search time: {avg_search_time*1000:.2f} ms")
    
    if "inference_server" in stats:
        print("Inference Server Stats:")
        for key, value in stats["inference_server"].items():
            print(f"  {key}: {value}")
    
    return stats

if __name__ == "__main__":
    test_patched_manager()