# add_batch_monitoring.py
"""
Add monitoring for batch statistics during training
"""

def add_monitoring():
    """Add monitoring for batch statistics"""
    with open('train_optimized.py', 'r') as file:
        content = file.read()
    
    # Add batch info to mcts_search function
    monitor_content = content.replace(
        "def mcts_search(root_state, inference_server, num_simulations, batch_size=MCTS_BATCH_SIZE):",
        "def mcts_search(root_state, inference_server, num_simulations, batch_size=MCTS_BATCH_SIZE, verbose=False):"
    )
    
    # Add batch size tracking
    monitor_content = monitor_content.replace(
        "            # Get leaf states\n            states = [leaf.state for leaf in leaves]",
        "            # Get leaf states\n            states = [leaf.state for leaf in leaves]\n            if verbose and len(states) > 1:\n                print(f\"  Batch size: {len(states)}\")"
    )
    
    # Add batch size tracking to game generation
    monitor_content = monitor_content.replace(
        "            # Perform MCTS with batching\n            root = mcts_search(state, self.inference_actor, NUM_SIMULATIONS)",
        "            # Perform MCTS with batching\n            verbose = (self.game_count < 3 or self.game_count % 10 == 0)  # Show batch info for first few games and occasionally\n            root = mcts_search(state, self.inference_actor, NUM_SIMULATIONS, verbose=verbose)"
    )
    
    # Add detailed stats to training summary
    monitor_content = monitor_content.replace(
        "            if self.game_count % 5 == 0 or games_completed == num_games:",
        "            # Get batch stats from inference server if available\n            if self.game_count % 5 == 0 or games_completed == num_games:"
    )
    
    with open('train_optimized.py', 'w') as file:
        file.write(monitor_content)
    
    print("Added batch statistics monitoring")

if __name__ == "__main__":
    add_monitoring()