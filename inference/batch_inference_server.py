# inference/batch_inference_server.py
"""
A completely new implementation of the inference server with explicit batch support
"""
import ray
import time
import queue
import threading
import numpy as np
from collections import OrderedDict

@ray.remote(num_gpus=0.2)
class BatchInferenceServer:
    """Inference server with explicit batch processing capabilities"""
    
    def __init__(self, batch_wait=0.005, cache_size=1000, max_batch_size=64):
        # Initialize basic attributes
        self.batch_wait = batch_wait
        self.max_batch_size = max_batch_size
        
        # Initialize request queues
        self.queue = queue.Queue()
        self.priority_queue = queue.PriorityQueue()
        
        # Performance monitoring
        self.total_requests = 0
        self.total_batch_requests = 0
        self.total_batches = 0
        self.total_cache_hits = 0
        self.batch_sizes = []
        self.inference_times = []
        self.last_stats_time = time.time()
        
        # Cache - reduced size to force more network evaluations
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()
        
        # Control flags
        self.setup_complete = False
        self.shutdown_flag = False
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
        
        print("BatchInferenceServer created - will initialize model on first use")
    
    def _setup(self):
        """Initialize PyTorch model on demand"""
        if self.setup_complete:
            return
            
        # Import torch only when needed
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
            
        self.setup_complete = True
        print(f"BatchInferenceServer initialized on {self.device}")
    
    def infer(self, state, priority=0):
        """Request inference for a single state"""
        # Ensure model is initialized
        if not self.setup_complete:
            self._setup()
            
        self.total_requests += 1
        
        # Try cache first
        board_key = str(state.board)
        with self.cache_lock:
            if board_key in self.cache:
                self.total_cache_hits += 1
                return self.cache[board_key]
        
        # Not in cache, queue for inference
        result_queue = queue.Queue()
        if priority > 0:
            self.priority_queue.put((priority, (state, result_queue)))
        else:
            self.queue.put((state, result_queue))
        
        return result_queue.get()
    
    def batch_infer(self, states):
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
                board_key = str(state.board)
                if board_key in self.cache:
                    results[i] = self.cache[board_key]
                    self.total_cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_states.append(state)
        
        # If all states were cached, return immediately
        if not uncached_states:
            return results
            
        # Process uncached states through neural network
        import torch
        
        # Record batch size for stats
        self.batch_sizes.append(len(uncached_states))
        self.total_batches += 1
        
        # Perform inference
        inference_start = time.time()
        try:
            # Convert states to numpy array then to tensor
            boards_array = np.array([s.board for s in uncached_states], dtype=np.float32)
            inputs = torch.tensor(boards_array, dtype=torch.float32).to(self.device)
            
            # Use mixed precision if on GPU
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    with torch.no_grad():
                        policy_batch, value_batch = self.model(inputs)
            else:
                with torch.no_grad():
                    policy_batch, value_batch = self.model(inputs)
                    
            # Move results back to CPU
            policy_batch = policy_batch.cpu().numpy()
            value_batch = value_batch.cpu().numpy()
            
            self.inference_times.append(time.time() - inference_start)
            
            # Update cache and fill results
            with self.cache_lock:
                for i, (idx, state) in enumerate(zip(uncached_indices, uncached_states)):
                    result = (policy_batch[i], value_batch[i][0])
                    
                    # Update cache
                    board_key = str(state.board)
                    self.cache[board_key] = result
                    
                    # Fill in result
                    results[idx] = result
                    
                # Trim cache if needed
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                    
        except Exception as e:
            print(f"Batch inference error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return default values for uncached states
            for idx in uncached_indices:
                results[idx] = (np.ones(9)/9, 0.0)
                
        return results
    
    def _batch_worker(self):
        """Worker thread that processes the queue"""
        while not self.shutdown_flag:
            # Only process once setup is complete
            if not self.setup_complete:
                time.sleep(0.1)
                continue
                
            # Import torch only when needed
            import torch
                
            states, futures = [], []
            
            # First handle priority requests
            while not self.priority_queue.empty() and len(states) < self.max_batch_size:
                _, (state, future) = self.priority_queue.get()
                states.append(state)
                futures.append(future)
            
            # Then handle regular requests
            timeout = self.batch_wait
            try_count = 0
            
            while len(states) < self.max_batch_size and try_count < 3:
                try:
                    state, future = self.queue.get(timeout=timeout)
                    states.append(state)
                    futures.append(future)
                    try_count = 0
                except:
                    try_count += 1
                    if len(states) > 0 or self.queue.empty():
                        break
                    time.sleep(0.001)
            
            # If we have states to process
            if states:
                # Process batch through neural network
                batch_results = self.batch_infer(states)
                
                # Return results through futures
                for i, future in enumerate(futures):
                    future.put(batch_results[i])
            
            # Log statistics periodically
            current_time = time.time()
            if current_time - self.last_stats_time > 10:  # Every 10 seconds
                avg_batch = sum(self.batch_sizes) / max(1, len(self.batch_sizes))
                avg_time = sum(self.inference_times) / max(1, len(self.inference_times))
                cache_hit_rate = self.total_cache_hits / max(1, self.total_requests) * 100
                
                print(f"BatchInferenceServer stats: Requests={self.total_requests}, "
                      f"Batch requests={self.total_batch_requests}, "
                      f"Batches={self.total_batches}, Avg batch={avg_batch:.1f}, "
                      f"Avg inference time={avg_time*1000:.2f}ms, "
                      f"Cache hit rate={cache_hit_rate:.1f}%, "
                      f"Queue size={self.queue.qsize()}, "
                      f"Cache size={len(self.cache)}")
                
                # Reset metrics
                self.batch_sizes = []
                self.inference_times = []
                self.last_stats_time = current_time
    
    def update_model(self, state_dict):
        """Update model with new weights"""
        # Import torch only when needed
        import torch
        
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
            
        return True
    
    def shutdown(self):
        """Gracefully shutdown the server"""
        self.shutdown_flag = True
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        print("BatchInferenceServer shutdown complete")