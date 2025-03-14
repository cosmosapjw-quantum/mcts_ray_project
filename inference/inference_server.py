# inference/inference_server.py
import ray
import time
import queue
import threading
import numpy as np
from collections import OrderedDict

# IMPORTANT: DO NOT import torch or model at the module level
# These will be imported inside methods as needed

@ray.remote(num_gpus=0.2)
class InferenceServer:
    """Inference server that avoids serialization issues with CUDA"""
    
    def __init__(self, batch_wait=0.01, cache_size=10000, max_batch_size=512):
        # Initialize basic attributes that don't require torch
        self.batch_wait = batch_wait
        self.max_batch_size = max_batch_size
        
        # Initialize request queues
        self.queue = queue.Queue()
        self.priority_queue = queue.PriorityQueue()
        
        # Performance monitoring
        self.total_requests = 0
        self.total_batches = 0
        self.total_cache_hits = 0
        self.batch_sizes = []
        self.inference_times = []
        self.last_stats_time = time.time()
        
        # Cache
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()
        
        # Control flags
        self.setup_complete = False
        self.shutdown_flag = False
        
        # Start worker thread - actual processing will wait for setup
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
        
        print("Inference server created - will initialize model on first use")
    
    def _setup(self):
        """Initialize PyTorch model on demand - imported only when called"""
        if self.setup_complete:
            return
            
        # Import torch only when needed (not during serialization)
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
        print(f"Inference server initialized on {self.device}")
    
    def infer(self, state, priority=0):
        """Request inference for a state with optional priority"""
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
    
    def _batch_worker(self):
        """Worker thread that batches inference requests"""
        # Loop until shutdown flag is set
        while not self.shutdown_flag:
            # Only process batches once setup is complete
            if not self.setup_complete:
                time.sleep(0.1)
                continue
                
            # Import torch only when needed (not during serialization)
            import torch
                
            batch_start_time = time.time()
            states, futures = [], []
            
            # First handle any priority requests
            while not self.priority_queue.empty() and len(states) < self.max_batch_size:
                _, (state, future) = self.priority_queue.get()
                states.append(state)
                futures.append(future)
            
            # Then handle regular requests
            timeout = max(0.001, self.batch_wait - (time.time() - batch_start_time))
            while len(states) < self.max_batch_size:
                try:
                    state, future = self.queue.get(timeout=timeout)
                    states.append(state)
                    futures.append(future)
                except:
                    break  # Timeout, move to inference
            
            # If we have states to process
            if states:
                self.total_batches += 1
                self.batch_sizes.append(len(states))
                
                # Perform batched inference
                inference_start = time.time()
                try:
                    # First convert all boards to a single numpy array, then to tensor
                    # This is much faster than creating a tensor from a list of arrays
                    boards_array = np.array([s.board for s in states], dtype=np.float32)
                    inputs = torch.tensor(boards_array, dtype=torch.float32).to(self.device)
                    
                    # Use mixed precision if on GPU (with updated API)
                    if self.device.type == 'cuda':
                        with torch.amp.autocast(device_type='cuda'):
                            with torch.no_grad():
                                policy_batch, value_batch = self.model(inputs)
                    else:
                        with torch.no_grad():
                            policy_batch, value_batch = self.model(inputs)
                    
                    # Move results back to CPU for Python processing
                    policy_batch = policy_batch.cpu().numpy()
                    value_batch = value_batch.cpu().numpy()
                    
                    self.inference_times.append(time.time() - inference_start)
                    
                    # Update cache and return results
                    for i, (state, future) in enumerate(zip(states, futures)):
                        result = (policy_batch[i], value_batch[i][0])
                        
                        # Cache the result
                        with self.cache_lock:
                            board_key = str(state.board)
                            self.cache[board_key] = result
                            # Trim cache if needed
                            if len(self.cache) > self.cache_size:
                                self.cache.popitem(last=False)
                        
                        future.put(result)
                
                except Exception as e:
                    # Handle errors gracefully
                    print(f"Inference error: {e}")
                    for future in futures:
                        # Return a reasonable default
                        future.put((np.ones(9)/9, 0.0))
            
            # Log statistics periodically
            current_time = time.time()
            if current_time - self.last_stats_time > 10:  # Every 10 seconds
                if self.total_batches > 0:
                    avg_batch = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
                    avg_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
                    cache_hit_rate = self.total_cache_hits / max(1, self.total_requests) * 100
                    
                    print(f"Inference stats: Requests={self.total_requests}, "
                          f"Batches={self.total_batches}, Avg batch={avg_batch:.1f}, "
                          f"Avg inference time={avg_time*1000:.2f}ms, "
                          f"Cache hit rate={cache_hit_rate:.1f}%")
                    
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
        # This is a critical step to avoid CUDA serialization issues
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
        print("Inference server shutdown complete")