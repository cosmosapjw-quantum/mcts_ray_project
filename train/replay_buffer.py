# train/replay_buffer.py
"""
Experience replay buffer for AlphaZero-style training.
"""
import threading
import numpy as np
import torch
from collections import deque
from config import REPLAY_BUFFER_SIZE

class ReplayBuffer:
    """Thread-safe experience replay buffer with prioritized sampling"""
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, experience, priority=1.0):
        """
        Add an experience to the buffer with priority
        
        Args:
            experience: tuple of (state, policy, value)
            priority: Sample priority (higher values are sampled more frequently)
        """
        with self.lock:
            self.buffer.append(experience)
            self.priorities.append(priority)
        
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on priority
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            tuple or None: (states, policies, values) tensor batch or None if buffer is too small
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                batch_size = len(self.buffer)
                
            # Convert priorities to probabilities
            probs = np.array(self.priorities) / sum(self.priorities)
            
            # Sample indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            
            states, policies, values = [], [], []
            
            for idx in indices:
                state, policy, value = self.buffer[idx]
                
                # Make sure everything has the right shape
                if state.shape != torch.Size([9]):
                    print(f"Warning: unexpected state shape: {state.shape}, expected [9]")
                    continue
                    
                if policy.shape != torch.Size([9]):
                    print(f"Warning: unexpected policy shape: {policy.shape}, expected [9]")
                    continue
                
                states.append(state)
                policies.append(policy)
                values.append(value)
            
            # Safety check
            if not states:
                # Empty batch after filtering, return None
                return None
                
            return (
                torch.stack(states),
                torch.stack(policies),
                torch.tensor(values, dtype=torch.float)
            )
        
    def __len__(self):
        """Get current buffer size"""
        return len(self.buffer)