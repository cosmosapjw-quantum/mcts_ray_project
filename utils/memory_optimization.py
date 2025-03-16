# utils/memory_optimization.py
import torch
import gc

def optimize_memory():
    """Optimize CUDA memory usage"""
    # Clear any cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Run garbage collection
    gc.collect()

def get_memory_stats():
    """Get current GPU memory usage statistics"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    stats = {}
    
    # Get current memory allocated
    stats["allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # Get maximum memory allocated
    stats["max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # Get current memory reserved
    stats["reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    
    # Get maximum memory reserved
    stats["max_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    
    return stats

def tensor_to_device(tensor_or_list, device, non_blocking=True):
    """Efficiently move tensors to device with support for nested structures"""
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.to(device, non_blocking=non_blocking)
    elif isinstance(tensor_or_list, list):
        return [tensor_to_device(t, device, non_blocking) for t in tensor_or_list]
    elif isinstance(tensor_or_list, tuple):
        return tuple(tensor_to_device(t, device, non_blocking) for t in tensor_or_list)
    elif isinstance(tensor_or_list, dict):
        return {k: tensor_to_device(v, device, non_blocking) for k, v in tensor_or_list.items()}
    else:
        return tensor_or_list