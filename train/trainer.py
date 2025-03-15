# train/trainer.py
"""
Neural network trainer for AlphaZero-style reinforcement learning.
"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from config import (
    BATCH_SIZE, TRAINING_EPOCHS, MIN_BUFFER_SIZE,
    CHECKPOINT_DIR
)

class Trainer:
    """Neural network trainer that handles optimization and checkpointing"""
    
    def __init__(self, model, optimizer, replay_buffer, device, scaler=None, log_dir='runs/trainer'):
        """
        Initialize the trainer
        
        Args:
            model: Neural network model
            optimizer: Optimizer for training
            replay_buffer: Experience replay buffer
            device: Device to train on (CPU/GPU)
            scaler: Optional gradient scaler for mixed precision training
            log_dir: Directory for TensorBoard logs
        """
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.device = device
        self.scaler = scaler
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-5
        )
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Training metrics
        self.train_steps = 0
        
    def train_batch(self):
        """
        Train on a batch of experiences for multiple epochs
        
        Returns:
            float or None: Average loss for the batch, or None if no training occurred
        """
        if len(self.replay_buffer) < MIN_BUFFER_SIZE:
            return None
            
        total_loss = 0
        
        # Train for multiple epochs on the same batch
        for epoch in range(TRAINING_EPOCHS):
            # Sample batch
            batch = self.replay_buffer.sample(BATCH_SIZE)
            if batch is None:
                continue
                
            states, policies, values = batch
            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Use mixed precision if available
            if self.scaler:
                with torch.amp.autocast(device_type='cuda'):
                    # Forward pass
                    pred_policies, pred_values = self.model(states)
                    
                    # Calculate loss
                    policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / policies.size(0)
                    value_loss = nn.MSELoss()(pred_values.squeeze(), values)
                    loss = policy_loss + value_loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                pred_policies, pred_values = self.model(states)
                
                # Calculate loss
                policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / policies.size(0)
                value_loss = nn.MSELoss()(pred_values.squeeze(), values)
                loss = policy_loss + value_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / TRAINING_EPOCHS if TRAINING_EPOCHS > 0 else 0
        
        # Update learning rate scheduler
        if avg_loss > 0:
            self.scheduler.step(avg_loss)
            
            # Log metrics
            self.train_steps += 1
            self.writer.add_scalar('Training/Loss', avg_loss, self.train_steps)
            self.writer.add_scalar('Training/LearningRate', self.scheduler.get_last_lr()[0], self.train_steps)
        
        return avg_loss
    
    def get_learning_rate(self):
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0]
        
    def save_checkpoint(self, name="model", additional_data=None):
        """
        Save model checkpoint
        
        Args:
            name: Checkpoint name
            additional_data: Additional data to include in the checkpoint
        """
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }
        
        # Add any additional data
        if additional_data:
            checkpoint.update(additional_data)
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
            
    def load_checkpoint(self, name="model"):
        """
        Load model checkpoint
        
        Args:
            name: Checkpoint name
            
        Returns:
            dict or None: Additional data from checkpoint or None if checkpoint not found
        """
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            # Load training steps
            if 'train_steps' in checkpoint:
                self.train_steps = checkpoint['train_steps']
                
            # Load scaler state if available
            if self.scaler and 'scaler' in checkpoint and checkpoint['scaler']:
                self.scaler.load_state_dict(checkpoint['scaler'])
                
            print(f"Loaded checkpoint: {path}")
            
            # Return any additional data
            return {k: v for k, v in checkpoint.items() 
                   if k not in ['model_state_dict', 'optimizer_state_dict', 
                                'scheduler_state_dict', 'train_steps', 'scaler']}
        
        else:
            print(f"Checkpoint not found: {path}")
            return None
    
    def close(self):
        """Close the trainer and release resources"""
        self.writer.close()