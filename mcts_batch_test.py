# mcts_batch_test.py
"""
A simplified test script for MCTS with batch inference
"""
import ray
import time
import numpy as np
from utils.state_utils import TicTacToeState
from inference.batch_inference_server import BatchInferenceServer
from mcts.node import Node

# Simple implementation of MCTS node selection
def select_node(node, exploration_weight=1.0):
    """Select a leaf node from the tree"""
    path = [node]
    while node.children:
        # Find best child according to UCB formula
        best_score = float('-inf')
        best_child = None
        for child in node.children:
            # PUCT formula
            if child.visits == 0:
                score = float('inf')  # Prioritize unexplored
            else:
                q_value = child.value / child.visits
                u_value = exploration_weight * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        node = best_child
        path.append(node)
    return node, path

# Simple implementation of node expansion
def expand_node(node, priors):
    """Expand a node with actions and their priors"""
    actions = node.state.get_legal_actions()
    if not actions:
        return  # Terminal node
        
    for action in actions:
        child_state = node.state.apply_action(action)
        child = Node(child_state, node)
        child.prior = priors[action]  # Use the prior for this action
        node.children.append(child)

# Simple backpropagation
def backpropagate(path, value):
    """Update statistics in the path"""
    for node in reversed(path):
        node.visits += 1
        node.value += value
        value = -value  # Flip for opponent's perspective

def test_mcts_batching():
    """Test MCTS with batch inference"""
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    
    # Create inference server
    print("Creating BatchInferenceServer...")
    inference_server = BatchInferenceServer.remote(batch_wait=0.002, cache_size=100)
    
    # Create initial state
    root_state = TicTacToeState()
    
    # Create root node
    root = Node(root_state)
    
    # Get initial policy and value
    print("Evaluating root node...")
    policy, value = ray.get(inference_server.infer.remote(root_state))
    
    # Expand root node
    expand_node(root, policy)
    root.value = value
    root.visits = 1
    
    # Perform MCTS simulations - traditional approach
    print("\nRunning MCTS with traditional (one-by-one) leaf evaluation...")
    start_time = time.time()
    num_simulations = 64
    for i in range(num_simulations):
        # Select a leaf node
        leaf, path = select_node(root)
        
        # Evaluate leaf
        if leaf.state.is_terminal():
            value = leaf.state.winner if leaf.state.winner is not None else 0
        else:
            # Use neural network
            policy, value = ray.get(inference_server.infer.remote(leaf.state))
            # Expand the leaf node
            expand_node(leaf, policy)
        
        # Backpropagate
        backpropagate(path, value)
        
        if i % 10 == 0:
            print(f"Completed {i+1}/{num_simulations} simulations")
    
    traditional_time = time.time() - start_time
    print(f"Traditional MCTS took {traditional_time:.3f}s for {num_simulations} simulations")
    
    # Now test with batch evaluation
    print("\nRunning MCTS with batch leaf evaluation...")
    start_time = time.time()
    
    # Reset the root
    root = Node(root_state)
    policy, value = ray.get(inference_server.infer.remote(root_state))
    expand_node(root, policy)
    root.value = value
    root.visits = 1
    
    # Run in batches of 16
    batch_size = 16
    remaining_sims = num_simulations
    
    while remaining_sims > 0:
        # Collect leaves for evaluation
        leaves = []
        paths = []
        terminal_leaves = []
        terminal_paths = []
        
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, remaining_sims)
        
        # Select leaves until batch is full
        while len(leaves) + len(terminal_leaves) < current_batch_size:
            leaf, path = select_node(root)
            
            if leaf.state.is_terminal():
                terminal_leaves.append(leaf)
                terminal_paths.append(path)
            else:
                leaves.append(leaf)
                paths.append(path)
        
        # Process terminal states immediately
        for leaf, path in zip(terminal_leaves, terminal_paths):
            value = leaf.state.winner if leaf.state.winner is not None else 0
            backpropagate(path, value)
            remaining_sims -= 1
        
        # Process non-terminal leaves with batch inference
        if leaves:
            # Get leaf states
            states = [leaf.state for leaf in leaves]
            
            # Batch inference
            results = ray.get(inference_server.batch_infer.remote(states))
            
            # Process results
            for leaf, path, result in zip(leaves, paths, results):
                policy, value = result
                expand_node(leaf, policy)
                backpropagate(path, value)
                remaining_sims -= 1
        
        print(f"Remaining simulations: {remaining_sims}")
    
    batch_time = time.time() - start_time
    print(f"Batch MCTS took {batch_time:.3f}s for {num_simulations} simulations")
    print(f"Speedup: {traditional_time/batch_time:.1f}x")
    
    # Check server stats
    time.sleep(12)  # Wait for stats to update
    
    # Clean up
    ray.shutdown()
    print("Test completed")

if __name__ == "__main__":
    test_mcts_batching()