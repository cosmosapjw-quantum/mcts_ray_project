# train/efficient_data.py
import torch
from torch.utils.data import Dataset, DataLoader

class ReplayBufferDataset(Dataset):
    """Dataset for efficient loading from replay buffer"""
    
    def __init__(self, replay_buffer, transform=None):
        self.replay_buffer = replay_buffer
        self.transform = transform
        
    def __len__(self):
        return len(self.replay_buffer)
    
    def __getitem__(self, idx):
        # Sample from replay buffer with priority
        sample = self.replay_buffer.sample(1)
        if sample is None:
            # Return dummy sample
            return torch.zeros(9), torch.zeros(9), torch.zeros(1)
        
        # Unpack sample
        states, policies, values = sample
        
        # Return first (and only) item
        return states[0], policies[0], values[0]

def create_efficient_dataloader(replay_buffer, batch_size=512, num_workers=2, pin_memory=True):
    """Create efficient DataLoader for training"""
    dataset = ReplayBufferDataset(replay_buffer)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )