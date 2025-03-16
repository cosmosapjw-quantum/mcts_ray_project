# utils/profiling.py
import torch
import time
from contextlib import contextmanager

@contextmanager
def cuda_profiler(enabled=True):
    """Context manager for CUDA profiling"""
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        yield
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"CUDA operation took: {elapsed_time:.2f} ms")
    else:
        yield

def profile_model_inference(model, input_tensor, warmup=10, iterations=100):
    """Profile model inference performance"""
    model.eval()
    device = next(model.parameters()).device
    
    # Move input to the same device as model
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)
    
    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure performance
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(input_tensor)
    
    # Synchronize after timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    latency_ms = (elapsed_time / iterations) * 1000
    throughput = iterations / elapsed_time
    
    return {
        "latency_ms": latency_ms,
        "throughput": throughput,
        "total_time": elapsed_time,
        "iterations": iterations
    }