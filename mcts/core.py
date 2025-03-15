# mcts/core.py
"""
Core Monte Carlo Tree Search algorithms with common interfaces.
This module provides the foundational MCTS algorithms without optimizations.
For performance-optimized implementations, see tree.py.
For distributed and parallel search, see search.py.
"""
import numpy as np
from mcts.node import Node
from config import EXPLORATION_WEIGHT

def select_node(node, exploration_weight=EXPLORATION_WEIGHT):
    """
    Select a leaf node from the tree using PUCT algorithm.
    
    This is the basic implementation without optimizations.
    For a performance-optimized version, see tree.py.
    
    Args:
        node: Root node to start selection from
        exploration_weight: Controls exploration vs exploitation tradeoff
        
    Returns:
        tuple: (leaf_node, path) - selected leaf node and path from root
    """
    path = [node]
    
    while node.children:
        # Find best child according to UCB formula
        best_score = float('-inf')
        best_child = None
        
        for child in node.children:
            # Skip if no visits (shouldn't happen normally)
            if child.visits == 0:
                # Prioritize unexplored nodes
                score = float('inf')
            else:
                # PUCT formula
                q_value = child.value / child.visits
                u_value = exploration_weight * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        node = best_child
        path.append(node)
    
    return node, path

def expand_node(node, priors, add_noise=False, alpha=0.3, epsilon=0.25):
    """
    Expand a node with possible actions and their prior probabilities.
    
    Args:
        node: Node to expand
        priors: Policy vector of action probabilities
        add_noise: Whether to add Dirichlet noise at root
        alpha: Dirichlet noise parameter
        epsilon: Noise weight factor
    """
    # Get legal actions
    actions = node.state.get_legal_actions()
    
    if not actions:
        return  # Terminal state, can't expand
    
    # Extract priors for legal actions
    legal_priors = [priors[a] for a in actions]
    
    # Normalize priors
    prior_sum = sum(legal_priors)
    if prior_sum > 0:
        legal_priors = [p / prior_sum for p in legal_priors]
    else:
        # If all priors are 0, use uniform distribution
        legal_priors = [1.0 / len(actions) for _ in actions]
    
    # Add Dirichlet noise at root for exploration
    if add_noise and node.parent is None:
        noise = np.random.dirichlet([alpha] * len(actions))
        legal_priors = [(1 - epsilon) * p + epsilon * n for p, n in zip(legal_priors, noise)]
    
    # Create children nodes
    for i, action in enumerate(actions):
        # Apply action to get new state
        child_state = node.state.apply_action(action)
        # Create child node
        child = Node(child_state, parent=node, prior=legal_priors[i], action=action)
        # Add to parent's children
        node.children.append(child)

def backpropagate(path, value):
    """
    Update statistics for nodes in the path.
    
    Args:
        path: List of nodes from root to leaf
        value: Value to backpropagate
    """
    # For each node in the path (from leaf to root)
    for node in reversed(path):
        # Update node statistics
        node.visits += 1
        node.value += value
        
        # Flip value perspective for opponent
        value = -value

def mcts_search_basic(root_state, inference_fn, num_simulations, exploration_weight=EXPLORATION_WEIGHT):
    """
    Basic single-threaded MCTS search.
    For optimized or parallel versions, see tree.py or search.py.
    
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
    
    # Expand root with initial policy
    expand_node(root, policy, add_noise=True)
    root.value = value
    root.visits = 1
    
    # Run simulations
    for _ in range(num_simulations - 1):  # -1 because we already did root
        # Selection phase - traverse tree to find leaf node
        leaf, path = select_node(root, exploration_weight)
        
        # If leaf is terminal, use game result
        if leaf.state.is_terminal():
            # Use actual game outcome
            value = leaf.state.get_winner()
        else:
            # Expansion phase - expand leaf and evaluate
            policy, value = inference_fn(leaf.state)
            expand_node(leaf, policy, add_noise=False)
        
        # Backpropagation phase - update statistics up the tree
        backpropagate(path, value)
    
    return root