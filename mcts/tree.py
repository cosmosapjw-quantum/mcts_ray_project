# mcts/tree.py
"""
Optimized MCTS implementations using Numba.
This module provides high-performance implementations of MCTS algorithms
for single-threaded execution with CPU optimization.
"""
import numpy as np
from numba import jit, njit, prange
from mcts.node import Node
from mcts.core import expand_node, backpropagate
from config import EXPLORATION_WEIGHT

# Pre-allocate memory for PUCT calculations to avoid repeated allocations
# in the hot path of the selection algorithm
@njit(cache=True, fastmath=True)
def compute_puct(values, visits, priors, total_visits, exploration_weight=1.0):
    """
    Compute PUCT values for node selection with Numba optimization.
    
    Args:
        values: Array of node values
        visits: Array of visit counts
        priors: Array of prior probabilities
        total_visits: Total visits at parent node
        exploration_weight: Exploration weight factor
        
    Returns:
        np.array: Array of PUCT scores
    """
    # Compute Q-values (mean action values)
    q_values = np.zeros_like(values, dtype=np.float32)
    mask = visits > 0
    q_values[mask] = values[mask] / visits[mask]
    
    # Compute U-values (exploration bonus)
    u_values = exploration_weight * priors * np.sqrt(total_visits) / (1 + visits)
    
    # Combine Q and U values
    return q_values + u_values

@njit(cache=True)
def argmax(x):
    """
    Fast argmax implementation for Numba.
    
    Args:
        x: Array to find the argmax for
        
    Returns:
        int: Index of the maximum value
    """
    max_idx = 0
    max_val = x[0]
    for i in range(1, len(x)):
        if x[i] > max_val:
            max_idx = i
            max_val = x[i]
    return max_idx

def select_optimized(node, exploration_weight=EXPLORATION_WEIGHT):
    """
    Select a leaf node using optimized PUCT algorithm with virtual loss.
    
    This is a high-performance implementation that vectorizes operations
    and applies virtual loss to discourage thread collisions.
    
    Args:
        node: Root node to start selection from
        exploration_weight: Controls exploration vs exploitation
        
    Returns:
        tuple: (leaf_node, path) - selected leaf node and path from root
    """
    current = node
    path = [current]
    
    # Follow tree policy down to a leaf node
    while current.children:
        n_children = len(current.children)
        
        # Extract node statistics for vectorized PUCT computation
        values = np.zeros(n_children, dtype=np.float32)
        visits = np.zeros(n_children, dtype=np.int32)
        priors = np.zeros(n_children, dtype=np.float32)
        
        for i, child in enumerate(current.children):
            values[i] = child.value
            visits[i] = child.visits
            priors[i] = child.prior
        
        total_visits = np.sum(visits) or 1  # Avoid division by zero
        
        # Calculate PUCT scores
        puct_scores = compute_puct(values, visits, priors, total_visits, exploration_weight)
        
        # Select the child with the highest PUCT score
        best_idx = argmax(puct_scores)
        current = current.children[best_idx]
        
        # Apply virtual loss to discourage other threads from exploring this path
        current.visits += 1  # Virtual loss: temporarily increment visit count
        current.value -= 0.1  # Virtual loss: temporarily decrease value
        
        path.append(current)
    
    return current, path

@njit(cache=True)
def dirichlet_noise(alpha, size):
    """
    Fast Dirichlet noise generation with Numba.
    
    Args:
        alpha: Dirichlet concentration parameter
        size: Size of the distribution
        
    Returns:
        np.array: Dirichlet noise samples
    """
    # Approximate Dirichlet using Gamma distribution
    samples = np.zeros(size, dtype=np.float32)
    for i in range(size):
        # Generate gamma sample for Dirichlet
        # Shape parameter = alpha, scale parameter = 1.0
        shape, scale = alpha, 1.0
        # Simple gamma sample approximation
        sample = 0.0
        for _ in range(10):  # Approximate with sum of exponentials
            sample -= np.log(np.random.random())
        sample *= scale / 10.0 * shape
        samples[i] = sample
    
    # Normalize to get Dirichlet
    total = np.sum(samples)
    if total > 0:
        samples /= total
    else:
        samples[:] = 1.0 / size
    
    return samples

def expand_optimized(node, priors, alpha=0.3, epsilon=0.25, add_noise=True):
    """
    Expand a node with improved vectorized operations and optional noise.
    
    Args:
        node: Node to expand
        priors: Policy vector of action probabilities
        alpha: Dirichlet concentration parameter
        epsilon: Noise weight factor
        add_noise: Whether to add Dirichlet noise at root
    """
    actions = node.state.get_legal_actions()
    n_actions = len(actions)
    
    if n_actions == 0:
        return  # No legal actions, cannot expand
    
    # Process priors
    action_priors = priors[actions]
    
    # Normalize priors for legal actions
    if np.sum(action_priors) > 0:
        action_priors = action_priors / np.sum(action_priors)
    else:
        # If all priors are zero, use uniform distribution
        action_priors = np.ones(n_actions, dtype=np.float32) / n_actions
    
    # Add Dirichlet noise at root for exploration
    if add_noise and node.parent is None:
        noise = dirichlet_noise(alpha, n_actions)
        action_priors = (1 - epsilon) * action_priors + epsilon * noise
    
    # Create child nodes
    for i, action in enumerate(actions):
        child_state = node.state.apply_action(action)
        child_node = Node(child_state, node)
        child_node.prior = action_priors[i]
        child_node.action = action
        node.children.append(child_node)

def backpropagate_optimized(path, value):
    """
    Backpropagate the evaluation through the path with virtual loss correction.
    
    Args:
        path: List of nodes from root to leaf
        value: Value to backpropagate
    """
    # Reverse the path to go from leaf to root
    for node in reversed(path):
        # Remove virtual loss effect and apply real update
        node.value += value + 0.1  # Add back the virtual loss deduction
        # No need to decrement visits as we're adding the real visit below
        
        # Update statistics
        node.visits += 1  # Real visit update
        
        # Flip value for opponent's perspective
        value = -value

def mcts_search_optimized(root_state, inference_fn, num_simulations, exploration_weight=EXPLORATION_WEIGHT):
    """
    Optimized single-threaded MCTS search.
    
    This uses Numba-accelerated functions for improved performance.
    
    Args:
        root_state: Initial game state
        inference_fn: Function that takes a state and returns (policy, value)
        num_simulations: Number of MCTS simulations to run
        exploration_weight: Controls exploration vs exploitation
        
    Returns:
        Node: Root node of the search tree
    """
    # Create root node
    root = Node(root_state)
    
    # Get initial policy and value
    policy, value = inference_fn(root_state)
    
    # Expand root node
    expand_optimized(root, policy, add_noise=True)
    root.value = value
    root.visits = 1
    
    # Run simulations
    for _ in range(num_simulations - 1):  # -1 because we already did root
        # Selection phase
        leaf, path = select_optimized(root, exploration_weight)
        
        # Check if leaf is terminal
        if leaf.state.is_terminal():
            value = leaf.state.get_winner()
        else:
            # Expansion phase
            policy, value = inference_fn(leaf.state)
            expand_optimized(leaf, policy, add_noise=False)
        
        # Backpropagation phase
        backpropagate_optimized(path, value)
    
    return root