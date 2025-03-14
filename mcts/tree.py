# mcts/tree.py
import numpy as np
from numba import jit, njit, prange
from mcts.node import Node

# Pre-allocate memory for PUCT calculations to avoid repeated allocations
# in the hot path of the selection algorithm
@njit(cache=True, fastmath=True)
def compute_puct(values, visits, priors, total_visits, exploration_weight=1.0):
    """Compute PUCT values for node selection with Numba optimization"""
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
    """Fast argmax implementation for Numba"""
    max_idx = 0
    max_val = x[0]
    for i in range(1, len(x)):
        if x[i] > max_val:
            max_idx = i
            max_val = x[i]
    return max_idx

def select(node, exploration_weight=1.0):
    """Select a leaf node to expand using PUCT algorithm with virtual loss"""
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
    """Fast Dirichlet noise generation with Numba"""
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

def expand(node, priors, alpha=0.3, epsilon=0.25, add_noise=True):
    """Expand a node with improved vectorized operations and optional noise"""
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
        node.children.append(child_node)

def backpropagate(path, value):
    """Backpropagate the evaluation through the path with virtual loss correction"""
    # Reverse the path to go from leaf to root
    for node in reversed(path):
        # Remove virtual loss effect and apply real update
        node.value += value + 0.1  # Add back the virtual loss deduction
        # No need to decrement visits as we're adding the real visit below
        
        # Update statistics
        node.visits += 1  # Real visit update
        
        # Flip value for opponent's perspective
        value = -value