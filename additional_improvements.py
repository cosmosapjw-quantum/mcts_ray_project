# additional_improvements.py
"""
Apply additional improvements to train_optimized.py
"""

def make_improvements():
    """Apply additional improvements to the training script"""
    with open('train_optimized.py', 'r') as file:
        content = file.read()
    
    # 1. Add minimum buffer size for training to avoid training too early
    improvements = content.replace(
        "if len(self.replay_buffer) < BATCH_SIZE:",
        "MIN_BUFFER_SIZE = BATCH_SIZE * 4  # Wait for a good amount of samples\n        if len(self.replay_buffer) < MIN_BUFFER_SIZE:"
    )
    
    # 2. Add batch size info in game statistics
    improvements = improvements.replace(
        "print(f\"Game {game_id}: {move_count} moves, outcome={outcome}, time={game_time:.1f}s\")",
        "print(f\"Game {game_id}: {move_count} moves, outcome={outcome}, time={game_time:.1f}s, buffer={len(self.replay_buffer)}\")"
    )
    
    # 3. Add learning rate monitoring
    improvements = improvements.replace(
        "    def train(self, num_games=100):",
        "    def train(self, num_games=100):\n        # Learning rate scheduler\n        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)"
    )
    
    # 4. Add learning rate scheduling
    improvements = improvements.replace(
        "            if loss is not None:",
        "            if loss is not None:\n                # Update learning rate scheduler\n                scheduler.step(loss)"
    )
    
    # 5. Add current learning rate to status output
    improvements = improvements.replace(
        "print(f\"  Loss: {loss:.4f if loss is not None else 'N/A'}\")",
        "print(f\"  Loss: {loss:.4f if loss is not None else 'N/A'}, LR: {self.optimizer.param_groups[0]['lr']:.2e}\")"
    )
    
    # Save the improved file
    with open('train_optimized.py', 'w') as file:
        file.write(improvements)
    
    print("Applied additional improvements to train_optimized.py")

if __name__ == "__main__":
    make_improvements()