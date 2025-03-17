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
                 mixed_precision=True,
                 verbose=False):
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
        
        self.verbose = verbose
    
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
            
            # Optimize memory format for Ampere architecture
            if self.device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
                self.model = self.model.to(memory_format=torch.channels_last)
                logger.info("Using channels_last memory format for Ampere architecture")
                
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
    
    def infer(self, state, retry_count=3, timeout=10.0, priority=0):
        """
        Request inference for a single state with improved error handling and retries.
        
        Args:
            state: Game state to evaluate
            retry_count: Number of retries on timeout/failure
            timeout: Timeout in seconds for each attempt
            priority: Priority level (higher = more urgent)
        
        Returns:
            tuple: (policy, value) for the state
        """
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
        
        # Not in cache, attempt inference with retries
        attempts = 0
        last_error = None
        
        while attempts < retry_count:
            attempts += 1
            
            try:
                # Not in cache, queue for inference
                result_queue = queue.Queue()
                self.queues[min(priority, 3)].put((state, result_queue))
                
                # Wait for result with timeout to prevent deadlock
                result = result_queue.get(timeout=timeout)
                
                # Cache successful result
                with self.cache_lock:
                    self.cache[board_key] = result
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
                        
                return result
                
            except queue.Empty:
                last_error = "Timeout waiting for inference"
                # Increase timeout for next attempt
                timeout = timeout * 1.5
                logger.warning(f"Inference timeout (attempt {attempts}/{retry_count}), retrying with longer timeout")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Inference error (attempt {attempts}/{retry_count}): {e}")
                # Short delay before retry
                time.sleep(0.1 * attempts)  # Increasing delay with each attempt
        
        # All attempts failed, return default values
        logger.error(f"All inference attempts failed: {last_error}")
        # Return default values
        policy = np.ones(9) / 9  # Uniform policy for TicTacToe
        value = 0.0
        return (policy, value)

    def batch_infer(self, states, retry_count=2, timeout=15.0, priority=0):
        """
        Direct batch inference method with improved error handling and memory management.
        
        Args:
            states: List of states to evaluate
            retry_count: Number of retries on timeout/failure
            timeout: Timeout in seconds for the batch
            priority: Priority level
        
        Returns:
            list: List of (policy, value) tuples for each state
        """
        # Ensure model is initialized
        if not self.setup_complete:
            self._setup()
            
        # Track as a batch request
        self.total_batch_requests += 1
        num_states = len(states)
        self.total_requests += num_states
        
        # Proactively split very large batches to avoid GPU memory issues
        if num_states > 128:
            logger.info(f"Large batch of {num_states} states, splitting into smaller chunks")
            results = []
            chunk_size = min(64, self.max_batch_size // 2)  # Conservative chunking
            
            for i in range(0, num_states, chunk_size):
                chunk_end = min(i + chunk_size, num_states)
                batch_chunk = states[i:chunk_end]
                chunk_size = len(batch_chunk)
                
                try:
                    # Process each chunk with reduced timeout
                    chunk_timeout = max(5.0, timeout * (chunk_size / num_states))
                    chunk_results = self.batch_infer(batch_chunk, retry_count, chunk_timeout, priority)
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error processing batch chunk {i//chunk_size}: {e}")
                    # Return fallback results for this chunk
                    fallback_results = [(np.ones(9)/9, 0.0) for _ in range(chunk_size)]
                    results.extend(fallback_results)
                    
            return results
        
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
                    
                    # Update cache hit statistics
                    cache_keys = list(self.cache.keys())
                    pos = cache_keys.index(board_key)
                    decile = min(9, int(10 * pos / len(self.cache)))
                    self.cache_hits_by_age[decile] += 1
                else:
                    uncached_indices.append(i)
                    uncached_states.append(state)
        
        # If all states were cached, return immediately
        if not uncached_states:
            return results
        
        # Force CUDA memory cleanup before processing new batch
        try:
            if hasattr(self, 'device') and self.device.type == 'cuda':
                import torch
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Error cleaning CUDA cache: {e}")
        
        # Process uncached states through neural network with retries
        attempts = 0
        last_error = None
        
        while attempts < retry_count and uncached_states:
            attempts += 1
            
            try:
                # Record batch size for stats
                batch_size = len(uncached_states)
                self.batch_sizes.append(batch_size)
                self.total_batches += 1
                
                # Check for very small batches (possible stall recovery)
                if batch_size == 1:
                    logger.debug("Processing single-state batch")
                
                # Track memory usage before inference
                mem_before = None
                try:
                    if hasattr(self, 'device') and self.device.type == 'cuda':
                        import torch
                        mem_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                except Exception:
                    pass
                    
                # Perform inference with timeout monitoring
                inference_start = time.time()
                policy_batch, value_batch = self._perform_inference(uncached_states)
                inference_time = time.time() - inference_start
                
                # Track memory usage after inference
                try:
                    if hasattr(self, 'device') and self.device.type == 'cuda' and mem_before is not None:
                        import torch
                        mem_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                        mem_diff = mem_after - mem_before
                        if mem_diff > 100:  # More than 100MB increase
                            logger.warning(f"Large memory increase during inference: {mem_diff:.1f}MB for batch of {batch_size}")
                except Exception:
                    pass
                
                self.inference_times.append(inference_time)
                
                # Validate results shape
                if len(policy_batch) != batch_size or len(value_batch) != batch_size:
                    logger.warning(f"Result size mismatch: got {len(policy_batch)} policies and {len(value_batch)} values for {batch_size} states")
                    
                    # Ensure correct lengths by truncating or padding
                    if len(policy_batch) > batch_size:
                        policy_batch = policy_batch[:batch_size]
                    if len(value_batch) > batch_size:
                        value_batch = value_batch[:batch_size]
                        
                    # Pad with defaults if needed
                    while len(policy_batch) < batch_size:
                        policy_batch = np.append(policy_batch, [np.ones(9)/9], axis=0)
                    while len(value_batch) < batch_size:
                        value_batch = np.append(value_batch, [[0.0]], axis=0)
                
                # Update cache and fill results
                with self.cache_lock:
                    for i, (idx, state) in enumerate(zip(uncached_indices, uncached_states)):
                        # Safety check for index
                        if i < len(policy_batch) and i < len(value_batch):
                            result = (policy_batch[i], value_batch[i][0] if value_batch[i].size > 0 else 0.0)
                            
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
                        
                # All processed successfully
                break
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Batch inference error (attempt {attempts}/{retry_count}): {e}")
                
                # If error was severe, throttle retries and possibly split the batch
                severe_error = any(term in str(e).lower() for term in 
                                ["cuda", "memory", "gpu", "device", "timeout", "ray"])
                
                if severe_error and batch_size > 16:
                    logger.warning(f"Severe error with batch size {batch_size}, splitting in half for retry")
                    
                    # Split the batch and process each half separately
                    half = batch_size // 2
                    first_half_indices = uncached_indices[:half]
                    first_half_states = uncached_states[:half]
                    second_half_indices = uncached_indices[half:]
                    second_half_states = uncached_states[half:]
                    
                    # Process first half
                    try:
                        half_timeout = max(5.0, timeout / 2)
                        policy_batch_1, value_batch_1 = self._perform_inference(first_half_states)
                        
                        # Update results with first half
                        for i, (idx, state) in enumerate(zip(first_half_indices, first_half_states)):
                            if i < len(policy_batch_1) and i < len(value_batch_1):
                                result = (policy_batch_1[i], value_batch_1[i][0] if value_batch_1[i].size > 0 else 0.0)
                                results[idx] = result
                                
                                # Update cache
                                with self.cache_lock:
                                    board_key = str(state.board if hasattr(state, 'board') else state)
                                    self.cache[board_key] = result
                    except Exception as inner_e:
                        logger.error(f"Error processing first half batch: {inner_e}")
                        # Fill with defaults
                        for idx in first_half_indices:
                            results[idx] = (np.ones(9)/9, 0.0)
                    
                    # Process second half
                    try:
                        policy_batch_2, value_batch_2 = self._perform_inference(second_half_states)
                        
                        # Update results with second half
                        for i, (idx, state) in enumerate(zip(second_half_indices, second_half_states)):
                            if i < len(policy_batch_2) and i < len(value_batch_2):
                                result = (policy_batch_2[i], value_batch_2[i][0] if value_batch_2[i].size > 0 else 0.0)
                                results[idx] = result
                                
                                # Update cache
                                with self.cache_lock:
                                    board_key = str(state.board if hasattr(state, 'board') else state)
                                    self.cache[board_key] = result
                    except Exception as inner_e:
                        logger.error(f"Error processing second half batch: {inner_e}")
                        # Fill with defaults
                        for idx in second_half_indices:
                            results[idx] = (np.ones(9)/9, 0.0)
                    
                    # Consider batch processed after split handling
                    break
                
                else:
                    # For non-severe errors or small batches, just wait and retry
                    sleep_time = min(2.0, 0.5 * attempts)
                    logger.warning(f"Waiting {sleep_time:.1f}s before retry")
                    time.sleep(sleep_time)
                    
                    # Force CUDA memory cleanup
                    try:
                        if hasattr(self, 'device') and self.device.type == 'cuda':
                            import torch
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
        
        # Fill in any remaining uncached results with default values
        for idx in uncached_indices:
            if results[idx] is None:
                logger.warning(f"Using default values for state at index {idx} after all attempts failed")
                results[idx] = (np.ones(9)/9, 0.0)
        
        # Verify that all results are populated
        for i, result in enumerate(results):
            if result is None:
                logger.error(f"Result at index {i} is still None, using default")
                results[i] = (np.ones(9)/9, 0.0)
        
        return results
    
    def _perform_inference(self, states):
        """
        Perform neural network inference with proper error handling.
        """
        import torch
        
        try:
            # Force memory cleanup before large batches
            if len(states) > 32 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Perform inference with timeout tracking
            start_time = time.time()
            max_inference_time = 5.0  # 5 second timeout
            
            # Move to correct device with error handling
            batch_tensor = self._prepare_encoded_batch(states)
            batch_tensor = batch_tensor.to(self.device)
            
            # More aggressive batch splitting for high memory usage
            if len(states) > 32 and torch.cuda.is_available():
                # Check memory usage
                if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.7:  # Lower threshold
                    logger.warning(f"High memory usage, splitting batch of {len(states)}")
                    half_point = len(states) // 2
                    results_1 = self._perform_inference(states[:half_point])
                    results_2 = self._perform_inference(states[half_point:])
                    return [
                        np.concatenate([results_1[0], results_2[0]]),
                        np.concatenate([results_1[1], results_2[1]])
                    ]
            
            # Use mixed precision if enabled
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    with torch.no_grad():
                        policy_batch, value_batch = self.model(batch_tensor)
            else:
                with torch.no_grad():
                    policy_batch, value_batch = self.model(batch_tensor)
            
            # Check for timeout
            if time.time() - start_time > max_inference_time:
                logger.warning(f"Inference timeout after {time.time() - start_time:.1f}s")
                raise TimeoutError("Inference took too long")
                
            # Move results back to CPU and convert to numpy
            policy_batch = policy_batch.cpu().numpy()
            value_batch = value_batch.cpu().numpy()
            
            return policy_batch, value_batch
                
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            # Provide fallback results
            batch_size = len(states)
            policy_size = getattr(states[0], 'policy_size', 9)
            fallback_policies = np.zeros((batch_size, policy_size))
            fallback_values = np.zeros((batch_size, 1))
            
            # Use uniform distribution for policies
            for i, state in enumerate(states):
                actions = state.get_legal_actions()
                if actions:
                    for action in actions:
                        fallback_policies[i][action] = 1.0 / len(actions)
                else:
                    fallback_policies[i] = np.ones(policy_size) / policy_size
            
            return fallback_policies, fallback_values
        
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
            # Add debug logging to track queue size
            if self.verbose and self.eval_queue.qsize() > 0:
                logger.debug(f"Eval queue size: {self.eval_queue.qsize()}")
            
            # Wait for substantial batch or timeout
            wait_start = time.time()
            batch_target_time = wait_start + self.max_wait_time
            min_batch_size = max(1, min(self.min_batch_size, self.batch_size // 2))
            
            # Determine batch collection strategy based on queue size
            queue_size = self.eval_queue.qsize()
            if queue_size >= self.batch_size:
                # Queue has enough items for a full batch, collect it immediately
                target_collect_size = min(self.batch_size, queue_size)
                max_wait_time = 0.002  # Very short wait (2ms)
            elif queue_size >= min_batch_size:
                # Queue has enough for a minimum batch, wait a bit for more
                target_collect_size = min(self.batch_size, queue_size)
                max_wait_time = self.max_wait_time / 2
            else:
                # Queue has few items, wait longer for batch formation
                target_collect_size = self.batch_size
                max_wait_time = self.max_wait_time
            
            # Collect batch with timeout strategy
            batch = []
            batch_too_small = True
            
            # Try to collect first item
            try:
                item = self.eval_queue.get(timeout=max_wait_time)
                batch.append(item)
                batch_too_small = len(batch) < min_batch_size
            except queue.Empty:
                # No items available, sleep briefly and continue
                time.sleep(0.001)
                self.wait_times.append(time.time() - wait_start)
                continue
            
            # Now try to collect remaining items up to target size
            collection_start = time.time()
            collection_timeout = min(max_wait_time, batch_target_time - collection_start)
            
            while len(batch) < target_collect_size and time.time() - collection_start < collection_timeout:
                try:
                    # Use short timeout for remaining items
                    item = self.eval_queue.get(timeout=0.001)
                    batch.append(item)
                    
                    # Check if we have enough for minimum batch size
                    if batch_too_small and len(batch) >= min_batch_size:
                        batch_too_small = False
                        # Since we have minimum, reduce remaining timeout
                        collection_timeout = min(collection_timeout, 0.005)
                except queue.Empty:
                    # If we have minimum batch size, wait less
                    if not batch_too_small:
                        break
                    time.sleep(0.0005)  # Very short sleep to avoid CPU spinning
            
            wait_time = time.time() - wait_start
            self.wait_times.append(wait_time)
            
            # Log batch collection stats
            if self.verbose and len(batch) > 1:
                logger.debug(f"Collected batch of {len(batch)} items in {wait_time*1000:.1f}ms")
            
            if not batch:
                # No items collected, sleep briefly
                time.sleep(0.001)
                continue
            
            # Process batch
            batch_size = len(batch)
            self.batch_sizes.append(batch_size)
            
            # Extract states for inference
            leaves, paths = zip(*batch)
            states = [leaf.state for leaf in leaves]
            
            # Perform inference
            inference_start = time.time()
            try:
                results = self._evaluate_batch(states)
                inference_time = time.time() - inference_start
                self.inference_times.append(inference_time)
                
                # Send results to result queue
                for i, (leaf, path) in enumerate(zip(leaves, paths)):
                    if i < len(results):
                        result = results[i]
                    else:
                        # Fallback for mismatch
                        result = (np.ones(9)/9, 0.0)
                    
                    self.result_queue.put((leaf, path, result))
                
                self.leaves_evaluated += batch_size
                self.batches_evaluated += 1
                
                # Log successful batch processing
                if self.verbose:
                    avg_batch = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
                    if batch_size > 1 or self.batches_evaluated % 10 == 0:
                        logger.debug(f"Processed batch {self.batches_evaluated}: size={batch_size}, " +
                                f"avg_size={avg_batch:.1f}, time={inference_time*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"Batch evaluation error: {e}")
                import traceback
                traceback.print_exc()
                
                # Return default values on error
                default_policy = np.ones(9) / 9  # Uniform policy
                default_value = 0.0  # Neutral value
                
                for leaf, path in batch:
                    self.result_queue.put((leaf, path, (default_policy, default_value)))
                    
        # Log final statistics on shutdown
        logger.debug(f"Evaluator shutdown - processed {self.batches_evaluated} batches, " +
                    f"avg size: {np.mean(self.batch_sizes) if self.batch_sizes else 0:.1f}")
    
    def _update_adaptive_batch_wait(self, current_batch_size):
        """Update batch wait time based on recent performance"""
        # Get batch fullness ratio
        batch_ratio = current_batch_size / self.max_batch_size
        
        # Measure queue growth rate
        queue_growth = 0
        if len(self.queue_sizes) >= 2:
            queue_growth = self.queue_sizes[-1] - self.queue_sizes[-2]
        
        # If batches are too small and queue isn't growing rapidly
        if batch_ratio < self.target_batch_ratio and queue_growth < 5:
            # Increase wait time, but not beyond max
            new_wait = min(
                self.max_batch_wait,
                self.current_batch_wait * (1 + self.adaptation_rate)
            )
        # If batches are consistently full or queue is growing rapidly, decrease wait time
        elif batch_ratio >= 0.95 or queue_growth > 10:
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
                        f"(batch ratio: {batch_ratio:.2f}, queue growth: {queue_growth})")
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
        
    def _profile_optimal_batch_size(self):
        """Profile different batch sizes to find optimal for this GPU"""
        if not self.setup_complete:
            self._setup()
            
        import torch
        
        # Create dummy batch for testing
        dummy_state = np.zeros((3, 3, 3), dtype=np.float32)  # Example state shape
        
        # Test different batch sizes
        batch_sizes = [64, 128, 256, 512]
        times = []
        
        for batch_size in batch_sizes:
            # Create test batch
            test_batch = [dummy_state] * batch_size
            tensors = torch.tensor(np.stack(test_batch), dtype=torch.float32).to(self.device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    if self.use_amp:
                        with torch.amp.autocast(device_type='cuda'):
                            self.model(tensors)
                    else:
                        self.model(tensors)
                        
            # Measure performance
            start = time.time()
            repeats = 20
            
            for _ in range(repeats):
                with torch.no_grad():
                    if self.use_amp:
                        with torch.amp.autocast(device_type='cuda'):
                            self.model(tensors)
                    else:
                        self.model(tensors)
            
            torch.cuda.synchronize()
            end = time.time()
            
            times.append((end - start) / repeats)
        
        # Find optimal batch size (lowest time per sample)
        throughputs = [size / time for size, time in zip(batch_sizes, times)]
        optimal_idx = np.argmax(throughputs)
        self.max_batch_size = batch_sizes[optimal_idx]
        
        logger.info(f"Optimal batch size for this GPU: {self.max_batch_size}")