# ray_reset.py
"""
This script shuts down any existing Ray instances and restarts Ray.
Run this before running your training script to ensure a clean environment.
"""
import ray
import os
import time
import psutil

def kill_ray_processes():
    """Find and kill all Ray processes"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'ray:' in proc.name() or 'raylet' in proc.name():
                print(f"Killing Ray process: {proc.name()} (PID: {proc.pid})")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def reset_ray():
    """Shutdown and restart Ray"""
    # Try graceful shutdown first
    if ray.is_initialized():
        print("Shutting down Ray...")
        ray.shutdown()
        time.sleep(2)
    
    # Force kill any remaining Ray processes
    kill_ray_processes()
    time.sleep(2)
    
    # Restart Ray
    print("Starting Ray...")
    ray.init()
    print("Ray restarted successfully!")

if __name__ == "__main__":
    reset_ray()