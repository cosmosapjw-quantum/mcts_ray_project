# fix_buffer_size.py
"""
Adjust the minimum buffer size for training to start sooner
"""

def adjust_buffer_size():
    """Reduce the minimum buffer size to start training sooner"""
    with open('train_optimized.py', 'r') as file:
        content = file.read()
    
    # Look for MIN_BUFFER_SIZE definition
    if "MIN_BUFFER_SIZE = BATCH_SIZE * 4" in content:
        # Replace with a smaller multiplier
        fixed_content = content.replace(
            "MIN_BUFFER_SIZE = BATCH_SIZE * 4",
            "MIN_BUFFER_SIZE = max(BATCH_SIZE // 2, 32)  # Start training with fewer samples"
        )
    else:
        # If not found, add it manually just before the check
        fixed_content = content.replace(
            "if len(self.replay_buffer) < BATCH_SIZE:",
            "MIN_BUFFER_SIZE = max(BATCH_SIZE // 2, 32)  # Start training with fewer samples\n        if len(self.replay_buffer) < MIN_BUFFER_SIZE:"
        )
    
    # Print current BATCH_SIZE for reference
    import re
    batch_size_match = re.search(r'BATCH_SIZE\s*=\s*(\d+)', content)
    if batch_size_match:
        batch_size = int(batch_size_match.group(1))
        print(f"Current BATCH_SIZE is {batch_size}")
        print(f"Setting MIN_BUFFER_SIZE to {max(batch_size // 2, 32)}")
    
    # Also reduce BATCH_SIZE itself for flexibility
    fixed_content = fixed_content.replace(
        "BATCH_SIZE = 256",
        "BATCH_SIZE = 128  # Reduced for earlier training"
    )
    
    with open('train_optimized.py', 'w') as file:
        file.write(fixed_content)
    
    print("Adjusted buffer size requirements to allow training to start sooner")

if __name__ == "__main__":
    adjust_buffer_size()