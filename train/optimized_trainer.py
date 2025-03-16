# train/optimized_trainer.py
import torch
import torch.nn as nn

class OptimizedTrainer:
    """Trainer with enhanced mixed precision support"""
    
    def __init__(self, model, optimizer, device, use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        
        # Initialize scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        
    def train_batch(self, states, policies, values):
        # Move data to device
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            pred_policies, pred_values = self.model(states)
            
            # Policy loss (cross-entropy)
            policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / policies.size(0)
            
            # Value loss (MSE)
            value_loss = self.value_loss_fn(pred_values.squeeze(), values)
            
            # Combined loss
            loss = policy_loss + value_loss
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }