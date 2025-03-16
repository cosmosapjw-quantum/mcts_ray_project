# inference/enhanced_batch_inference_server.py
"""
Enhanced implementation of the inference server with adaptive batching and improved GPU utilization
"""
import ray
import time
import queue
import threading
import numpy as np
import logging
from collections import OrderedDict, deque
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedBatchInferenceServer")

@ray.remote(num_gpus=1.0)  # Allocate full GPU
class EnhancedBatchInferenceServer:
    def __init__(self, 
                 initial_batch_wait=0.001,    # Reduced from original
                 cache_size=20000,            # Increased cache
                 max_batch_size=256,          # Larger batches
                 adaptive_batching=True,      # Enable adaptive
                 monitoring_interval=10.0,
                 mixed_precision=True):
        # Track creation time for health checks
        self.creation_time = time.time()
        
        # Initialize basic attributes
        self.initial_batch_wait = initial_batch_wait
        self.current_batch_wait = initial_batch_wait
        self.max_batch_size = max_batch_size
        self.adaptive_batching = adaptive_batching
        self.monitoring_interval = monitoring_interval
        self.mixed_precision = mixed_precision
        
        # Adaptive parameters
        self.min_batch_wait = 0.0005  # Minimum wait time (0.5ms)
        self.max_batch_wait = 0.005   # Maximum wait time (5ms)
        self.target_batch_ratio = 0.7  # Target batch fullness (70%)
        self.adaptation_rate = 0.1    # How quickly to adapt
        
        # Initialize request queues with higher priority levels (0-3)
        self.queues = [queue.Queue() for _ in range(4)]  # Multiple priority levels
        
        # Performance monitoring
        self.total_requests = 0
        self.total_batch_requests = 0
        self.total_batches = 0
        self.total_cache_hits = 0
        self.batch_sizes = deque(maxlen=100)  # Store most recent batch sizes
        self.inference_times = deque(maxlen=100)  # Store most recent inference times
        self.batch_waits = deque(maxlen=100)  # Store batch wait times for tuning
        self.queue_sizes = deque(maxlen=100)  # Track queue sizes
        self.last_stats_time = time.time()
        
        # Expanded monitoring
        self.gpu_utilization = deque(maxlen=100)  # Estimated GPU utilization
        self.batch_fullness = deque(maxlen=100)   # How full batches are (ratio)
        self.priority_distribution = [0, 0, 0, 0]  # Count by priority level
        
        # Cache with improved capacity
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()
        self.cache_hits_by_age = [0] * 10  # Hits by age decile
        
        # Control flags
        self.setup_complete = False
        self.shutdown_flag = False
        self.health_status = "initializing"
        
        # Worker threads
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        
        # Start threads
        self.worker_thread.start()
        self.monitoring_thread.start()
        
        logger.info("EnhancedBatchInferenceServer created - will initialize model on first use")
        logger.info(f"Configuration: max_batch_size={max_batch_size}, " +
                   f"initial_batch_wait={initial_batch_wait}, " +
                   f"adaptive_batching={adaptive_batching}, " +
                   f"mixed_precision={mixed_precision}")
        
        # Allow server to initialize asynchronously
        threading.Thread(target=self._delayed_setup, daemon=True).start()
    
    def _delayed_setup(self):
        """Initialize PyTorch model after a short delay"""
        try:
            time.sleep(1.0)  # Brief delay
            self._setup()
        except Exception as e:
            logger.error(f"Error in delayed setup: {e}")
    
    def _setup(self):
        """Initialize PyTorch model on demand with optimized settings"""
        if self.setup_complete:
            return
            
        # Import torch only when needed
        try:
            import torch
            from model import SmallResNet
            
            # Configure device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize model and move to device
            self.model = SmallResNet()
            self.model.to(self.device)
            self.model.eval()
            
            # Enable auto-tuned CUDA kernels
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                
                # Set up mixed precision if requested and supported
                self.use_amp = self.mixed_precision and hasattr(torch.cuda, 'amp')
                if self.use_amp:
                    logger.info("Using mixed precision (FP16) for inference")
                    self.amp_dtype = torch.float16
                else:
                    self.use_amp = False
                    logger.info("Using full precision (FP32) for inference")
            else:
                logger.warning("CUDA not available, using CPU for inference")
                self.use_amp = False
            
            self.setup_complete = True
            self.health_status = "ready"
            logger.info(f"Model initialized on {self.device}")
        except Exception as e:
            self.health_status = f"setup_failed: {str(e)}"
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def infer(self, state, priority=0):
        """Request inference for a single state"""
        # Ensure model is initialized
        if not self.setup_complete:
            self._setup()
            
        self.total_requests += 1
        self.priority_distribution[min(priority, 3)] += 1
        
        # Try cache first
        board_key = str(state.board if hasattr(state, 'board') else state)
        with self.cache_lock:
            if board_key in self.cache:
                # Move to end to mark as recently used
                result = self.cache[board_key]
                self.cache.move_to_end(board_key)
                self.total_cache_hits += 1
                
                # Update cache hit statistics
                cache_keys = list(self.cache.keys())
                pos = cache_keys.index(board_key)
                decile = min(9, int(10 * pos / len(self.cache)))
                self.cache_hits_by_age[decile] += 1
                
                return result
        
        # Not in cache, queue for inference
        result_queue = queue.Queue()
        self.queues[min(priority, 3)].put((state, result_queue))
        
        # Wait for result with timeout to prevent deadlock
        try:
            result = result_queue.get(timeout=5.0)  # 5 second timeout
            return result
        except queue.Empty:
            logger.warning("Inference request timed out, returning default values")
            # Return default values
            policy = np.ones(9) / 9  # Uniform policy for TicTacToe
            value = 0.0
            return (policy, value)
    
    def batch_infer(self, states, priority=0):
        """Direct batch inference method for multiple states at once"""
        # Ensure model is initialized
        if not self.setup_complete:
            self._setup()
            
        # Track as a batch request
        self.total_batch_requests += 1
        num_states = len(states)
        self.total_requests += num_states
        
        # Check cache first for each state
        results = [None] * num_states
        uncached_indices = []
        uncached_states = []
        
        with self.cache_lock:
            for i, state in enumerate(states):
                board_key = str(state.board if hasattr(state, 'board') else state)
                if board_key in self.cache:
                    results[i] = self.cache[board_key]
                    self.cache.move_to_end(board_key)
                    self.total_cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_states.append(state)
        
        # If all states were cached, return immediately
        if not uncached_states:
            return results
            
        # Process uncached states through neural network
        try:
            # Record batch size for stats
            self.batch_sizes.append(len(uncached_states))
            self.total_batches += 1
            
            # Perform inference
            inference_start = time.time()
            policy_batch, value_batch = self._perform_inference(uncached_states)
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # Update cache and fill results
            with self.cache_lock:
                for i, (idx, state) in enumerate(zip(uncached_indices, uncached_states)):
                    # Safety check for index
                    if i < len(policy_batch) and i < len(value_batch):
                        result = (policy_batch[i], value_batch[i][0])
                        
                        # Update cache
                        board_key = str(state.board if hasattr(state, 'board') else state)
                        self.cache[board_key] = result
                        
                        # Fill in result
                        results[idx] = result
                    else:
                        # Handle out of range indices (shouldn't happen but for safety)
                        default_policy = np.ones(9) / 9
                        default_value = 0.0
                        results[idx] = (default_policy, default_value)
                    
                # Trim cache if needed
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                    
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return default values for uncached states
            for idx in uncached_indices:
                if results[idx] is None:  # Only fill in if not already set
                    results[idx] = (np.ones(9)/9, 0.0)
                
        return results
    
    def _perform_inference(self, states):
        """
        Perform neural network inference with support for multiple game types.
        This version handles states from different game implementations.
        """
        import torch
        
        try:
            # First, determine if all states are from the same game type
            game_types = set()
            for state in states:
                game_types.add(state.game_name if hasattr(state, 'game_name') else type(state).__name__)
            
            # If mixed game types (unlikely), process in separate batches
            if len(game_types) > 1:
                logger.warning(f"Mixed game types in batch: {game_types}")
                
                # Group states by game type
                game_groups = {}
                for i, state in enumerate(states):
                    game_name = state.game_name if hasattr(state, 'game_name') else type(state).__name__
                    if game_name not in game_groups:
                        game_groups[game_name] = []
                    game_groups[game_name].append((i, state))
                
                # Process each group separately
                all_policies = [None] * len(states)
                all_values = [None] * len(states)
                
                for game_name, state_group in game_groups.items():
                    indices = [i for i, _ in state_group]
                    group_states = [s for _, s in state_group]
                    
                    # Process this group
                    group_policies, group_values = self._process_game_batch(group_states, game_name)
                    
                    # Fill in results
                    for idx, policy, value in zip(indices, group_policies, group_values):
                        all_policies[idx] = policy
                        all_values[idx] = value
                
                return all_policies, all_values
            
            # Standard case: all states are from the same game
            game_name = list(game_types)[0]
            return self._process_game_batch(states, game_name)
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
        
    def _process_game_batch(self, states, game_name):
        """
        Process a batch of states from the same game type.
        
        Args:
            states: List of game states of the same type
            game_name: Name of the game type
            
        Returns:
            tuple: (policies, values) as numpy arrays
        """
        import torch
        
        # Get policy size for this game (from registry if available)
        if hasattr(states[0], 'policy_size'):
            policy_size = states[0].policy_size
        else:
            try:
                from utils.game_registry import GameRegistry
                policy_size = GameRegistry.get_policy_size(game_name)
            except Exception as e:
                logger.warning(f"Could not determine policy size from registry: {e}")
                # Fallback: guess based on first state
                if hasattr(states[0], 'board') and hasattr(states[0].board, 'size'):
                    policy_size = states[0].board.size
                else:
                    # Last resort: assume TicTacToe
                    policy_size = 9
                    logger.warning(f"Using fallback policy size of {policy_size} for {game_name}")
        
        # Convert states to tensor format using standardized encoding method
        if hasattr(states[0], 'encode_for_inference'):
            # Use game-specific encoding
            try:
                encoded_states = [state.encode_for_inference() for state in states]
                batch_tensor = self._prepare_encoded_batch(encoded_states)
            except Exception as e:
                logger.error(f"Error encoding states: {e}, falling back to basic encoding")
                encoded_states = [state.encode() for state in states]
                batch_tensor = self._prepare_encoded_batch(encoded_states)
        else:
            # Use basic encoding
            try:
                encoded_states = [state.encode() for state in states]
                batch_tensor = self._prepare_encoded_batch(encoded_states)
            except Exception as e:
                logger.error(f"Error encoding states: {e}, using fallback")
                # Last resort: try to extract board
                batch_tensor = self._extract_board_fallback(states)
        
        # Move to correct device
        batch_tensor = batch_tensor.to(self.device)
        
        # Perform inference with proper error handling
        try:
            # Use mixed precision if enabled
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    with torch.no_grad():
                        policy_batch, value_batch = self.model(batch_tensor)
            else:
                with torch.no_grad():
                    policy_batch, value_batch = self.model(batch_tensor)
                    
            # Move results back to CPU and convert to numpy
            policy_batch = policy_batch.cpu().numpy()
            value_batch = value_batch.cpu().numpy()
            
            # Update GPU utilization estimate
            batch_size_ratio = len(states) / self.max_batch_size
            self.gpu_utilization.append(batch_size_ratio)
            self.batch_fullness.append(batch_size_ratio)
            
            # Mask illegal moves if needed
            if hasattr(states[0], 'get_action_mask'):
                for i, state in enumerate(states):
                    mask = state.get_action_mask()
                    
                    # Apply mask - set illegal move probabilities to 0
                    policy_batch[i] = policy_batch[i] * mask
                    
                    # Renormalize policy (if any legal moves remain)
                    policy_sum = policy_batch[i].sum()
                    if policy_sum > 0:
                        policy_batch[i] /= policy_sum
                    else:
                        # If all moves were masked or sum is 0, use uniform policy over legal moves
                        legal_actions = state.get_legal_actions()
                        if legal_actions:
                            uniform_prob = 1.0 / len(legal_actions)
                            for action in legal_actions:
                                policy_batch[i][action] = uniform_prob
            
            return policy_batch, value_batch
                
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            # Provide fallback results
            batch_size = len(states)
            fallback_policies = np.zeros((batch_size, policy_size))
            fallback_values = np.zeros((batch_size, 1))
            
            # Use uniform distribution over legal moves for each state
            for i, state in enumerate(states):
                legal_actions = state.get_legal_actions()
                if legal_actions:
                    uniform_prob = 1.0 / len(legal_actions)
                    for action in legal_actions:
                        fallback_policies[i][action] = uniform_prob
                else:
                    # If no legal actions, use uniform over all actions
                    fallback_policies[i] = np.ones(policy_size) / policy_size
            
            return fallback_policies, fallback_values

    def _prepare_encoded_batch(self, encoded_states):
        """
        Prepare a batch tensor from encoded states, handling different formats.
        
        Args:
            encoded_states: List of encoded states (numpy arrays)
            
        Returns:
            torch.Tensor: Batch tensor ready for model input
        """
        import torch
        import numpy as np
        
        # Check for consistent shapes
        shapes = set(tuple(state.shape) for state in encoded_states)
        
        if len(shapes) > 1:
            # States have inconsistent shapes - need to handle separately
            raise ValueError(f"Inconsistent shapes in batch: {shapes}")
        
        # Stack into a batch tensor
        try:
            # Try numpy stack first
            batch = np.stack(encoded_states)
            return torch.tensor(batch, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error stacking encoded states: {e}")
            
            # Fallback: convert individually
            tensors = []
            for state in encoded_states:
                tensors.append(torch.tensor(state, dtype=torch.float32))
            
            return torch.stack(tensors)

    def _extract_board_fallback(self, states):
        """
        Last resort fallback to extract board representations from states.
        
        Args:
            states: List of game states
            
        Returns:
            torch.Tensor: Batch tensor with basic board representation
        """
        import torch
        import numpy as np
        
        # Try to extract board attribute
        boards = []
        for state in states:
            if hasattr(state, 'board'):
                if isinstance(state.board, np.ndarray):
                    boards.append(state.board.astype(np.float32))
                else:
                    # Try to convert to numpy array
                    boards.append(np.array(state.board, dtype=np.float32))
            else:
                # Create a zero board as placeholder - will give meaningless results
                # but prevents complete failure
                boards.append(np.zeros(9, dtype=np.float32))  # Assuming 3x3 board
        
        # Stack boards and convert to tensor
        return torch.tensor(np.stack(boards), dtype=torch.float32)
    
    def _batch_worker(self):
        """Enhanced worker thread that processes the queue with adaptive batching"""
        while not self.shutdown_flag:
            # Only process once setup is complete
            if not self.setup_complete:
                time.sleep(0.1)
                self._setup()  # Try to set up if not done
                continue
                
            # Import torch only when needed
            import torch
                
            states, futures = [], []
            
            # Start timing for batch collection
            batch_start_time = time.time()
            batch_target_time = batch_start_time + self.current_batch_wait
            
            # Try to fill batch up to max_batch_size or until wait time expires
            while len(states) < self.max_batch_size:
                # Check if we've exceeded wait time
                current_time = time.time()
                if current_time >= batch_target_time and len(states) > 0:
                    break
                    
                # Determine remaining wait time
                remaining_wait = max(0, batch_target_time - current_time)
                
                # Process from highest priority queue to lowest with available items
                request_processed = False
                
                for priority in range(3, -1, -1):  # From 3 (highest) to 0 (lowest)
                    try:
                        if not self.queues[priority].empty():
                            state, future = self.queues[priority].get_nowait()
                            states.append(state)
                            futures.append(future)
                            request_processed = True
                            break  # Process one request at a time to check time again
                    except queue.Empty:
                        continue
                
                # If no request processed from any queue, wait a bit
                if not request_processed:
                    # If we already have some items, don't wait long
                    if len(states) > 0:
                        time.sleep(0.0001)  # 0.1ms
                    else:
                        time.sleep(0.001)  # 1ms
            
            # Record batch collection time
            collection_time = time.time() - batch_start_time
            self.batch_waits.append(collection_time)
            
            # If we have states to process
            if states:
                # Process batch through neural network
                batch_size = len(states)
                self.batch_sizes.append(batch_size)
                self.batch_fullness.append(batch_size / self.max_batch_size)
                
                try:
                    # Run inference
                    inference_start = time.time()
                    policy_batch, value_batch = self._perform_inference(states)
                    inference_time = time.time() - inference_start
                    self.inference_times.append(inference_time)
                    
                    # Return results through futures
                    for i, future in enumerate(futures):
                        if i < len(policy_batch) and i < len(value_batch):
                            future.put((policy_batch[i], value_batch[i][0]))
                        else:
                            # Safety check - shouldn't happen but just in case
                            future.put((np.ones(9)/9, 0.0))
                    
                    # Update adaptive batch wait if enabled
                    if self.adaptive_batching:
                        self._update_adaptive_batch_wait(batch_size)
                        
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    # Return default values on error
                    default_policy = np.ones(9) / 9  # Uniform policy
                    default_value = 0.0  # Neutral value
                    
                    for future in futures:
                        future.put((default_policy, default_value))
            
            # Track queue sizes for monitoring
            total_queued = sum(q.qsize() for q in self.queues)
            self.queue_sizes.append(total_queued)
    
    def _update_adaptive_batch_wait(self, current_batch_size):
        """Update batch wait time based on recent performance"""
        # Get batch fullness ratio
        batch_ratio = current_batch_size / self.max_batch_size
        
        # If batches are too small, increase wait time to collect larger batches
        if batch_ratio < self.target_batch_ratio:
            # Increase wait time, but not beyond max
            new_wait = min(
                self.max_batch_wait,
                self.current_batch_wait * (1 + self.adaptation_rate)
            )
        # If batches are consistently full, decrease wait time to reduce latency
        elif batch_ratio >= 0.95 and len(self.batch_fullness) > 5 and np.mean(list(self.batch_fullness)[-5:]) > 0.9:
            # Decrease wait time, but not below min
            new_wait = max(
                self.min_batch_wait,
                self.current_batch_wait * (1 - self.adaptation_rate)
            )
        else:
            # Keep current wait time
            new_wait = self.current_batch_wait
        
        # Update if changed significantly
        if abs(new_wait - self.current_batch_wait) / self.current_batch_wait > 0.05:
            logger.debug(f"Adjusting batch wait time: {self.current_batch_wait:.6f} → {new_wait:.6f} " +
                        f"(batch ratio: {batch_ratio:.2f})")
            self.current_batch_wait = new_wait
    
    def _monitoring_worker(self):
        """Monitor performance and health of the inference server"""
        while not self.shutdown_flag:
            time.sleep(self.monitoring_interval)
            
            try:
                # Check if we're processing requests
                current_time = time.time()
                if current_time - self.last_stats_time > self.monitoring_interval:
                    self._log_statistics()
                    self.last_stats_time = current_time
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
    
    def _log_statistics(self):
        """Log detailed performance statistics"""
        try:
            # Calculate basic statistics
            avg_batch = np.mean(self.batch_sizes) if self.batch_sizes else 0
            avg_time = np.mean(self.inference_times) if self.inference_times else 0
            cache_hit_rate = self.total_cache_hits / max(1, self.total_requests) * 100
            avg_batch_wait = np.mean(self.batch_waits) if self.batch_waits else 0
            avg_queue_size = np.mean(self.queue_sizes) if self.queue_sizes else 0
            
            # Calculate batch fullness
            batch_fullness = np.mean(self.batch_fullness) * 100 if self.batch_fullness else 0
            
            # Estimate GPU utilization (very rough)
            gpu_util = np.mean(self.gpu_utilization) * 100 if self.gpu_utilization else 0
            
            # Get current memory usage if on GPU
            gpu_mem_used = "N/A"
            try:
                if hasattr(self, 'device') and self.device.type == 'cuda':
                    import torch
                    gpu_mem_used = f"{torch.cuda.memory_allocated(self.device) / (1024**2):.1f}MB"
            except:
                pass
            
            # Log basic statistics
            logger.info(
                f"Stats: Requests={self.total_requests}, "
                f"Batches={self.total_batches}, Avg batch={avg_batch:.1f}, "
                f"Batch fullness={batch_fullness:.1f}%, "
                f"Avg inference time={avg_time*1000:.2f}ms, "
                f"Cache hit rate={cache_hit_rate:.1f}%, "
                f"Queue size={avg_queue_size:.1f}, "
                f"Batch wait={self.current_batch_wait*1000:.2f}ms, "
                f"Estimated GPU util={gpu_util:.1f}%, "
                f"GPU memory={gpu_mem_used}"
            )
            
            # Log priority distribution
            total_priority = sum(self.priority_distribution)
            if total_priority > 0:
                priority_pcts = [100 * count / total_priority for count in self.priority_distribution]
                logger.debug(f"Priority distribution: P0={priority_pcts[0]:.1f}%, " +
                            f"P1={priority_pcts[1]:.1f}%, P2={priority_pcts[2]:.1f}%, " +
                            f"P3={priority_pcts[3]:.1f}%")
            
        except Exception as e:
            logger.error(f"Error logging statistics: {e}")
    
    def update_model(self, state_dict):
        """Update model with new weights"""
        # Import torch only when needed
        import torch
        
        try:
            if not self.setup_complete:
                self._setup()
                
            # Convert state dict format for safe loading
            new_state_dict = {}
            for k, v in state_dict.items():
                if isinstance(v, np.ndarray):
                    new_state_dict[k] = torch.tensor(v)
                else:
                    new_state_dict[k] = v
            
            # Load new weights
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Clear cache after model update
            with self.cache_lock:
                self.cache.clear()
                
            logger.info("Model updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            self.health_status = f"model_update_failed: {str(e)}"
            return False
    
    def get_health_status(self):
        """Return current health status of the server"""
        return {
            "status": self.health_status,
            "uptime": time.time() - self.creation_time,
            "batch_size": np.mean(self.batch_sizes) if self.batch_sizes else 0,
            "inference_time_ms": np.mean(self.inference_times) * 1000 if self.inference_times else 0,
            "cache_size": len(self.cache),
            "queue_size": sum(q.qsize() for q in self.queues),
            "setup_complete": self.setup_complete
        }
    
    def shutdown(self):
        """Gracefully shutdown the server"""
        logger.info("Shutting down inference server...")
        self.shutdown_flag = True
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Inference server shutdown complete")