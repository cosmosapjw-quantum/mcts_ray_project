# inference/state_batcher.py
import numpy as np
import torch
from collections import deque

class StateBatcher:
    """Efficient state batching for GPU inference"""
    
    def __init__(self, max_batch_size=256, wait_time=0.001, device="cuda"):
        self.max_batch_size = max_batch_size
        self.wait_time = wait_time
        self.device = device
        
        # Pending states and result future queues
        self.pending_states = []
        self.pending_futures = []
        
        # Performance tracking
        self.batch_sizes = deque(maxlen=100)
        self.wait_times = deque(maxlen=100)
    
    def preprocess_batch(self, states):
        """Preprocess a batch of states for the neural network"""
        # Handle different input types
        if isinstance(states[0], np.ndarray):
            # Convert numpy arrays to tensor
            batch = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        elif hasattr(states[0], 'encode'):
            # States with encode method
            batch = torch.tensor(np.stack([s.encode() for s in states]), 
                               dtype=torch.float32, device=self.device)
        else:
            # Assume already tensor-like
            batch = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
            
        return batch
    
    def add_to_batch(self, state, future):
        """Add a state to the current batch"""
        self.pending_states.append(state)
        self.pending_futures.append(future)
        
        return len(self.pending_states) >= self.max_batch_size
    
    def process_batch(self, model):
        """Process the current batch with the model"""
        if not self.pending_states:
            return
            
        # Record batch size
        batch_size = len(self.pending_states)
        self.batch_sizes.append(batch_size)
        
        # Create batch tensor
        batch = self.preprocess_batch(self.pending_states)
        
        # Run inference
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                policies, values = model(batch)
                
        # Convert to CPU and numpy
        policies = policies.cpu().numpy()
        values = values.cpu().numpy()
        
        # Return results to futures
        for i, future in enumerate(self.pending_futures):
            future.set_result((policies[i], values[i][0]))
            
        # Clear pending
        self.pending_states.clear()
        self.pending_futures.clear()