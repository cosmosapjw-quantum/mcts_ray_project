# mcts/core.py
"""
Core Monte Carlo Tree Search algorithms with common interfaces.
This module provides the foundational MCTS algorithms without optimizations.
For performance-optimized implementations, see tree.py.
For distributed and parallel search, see search.py.
"""
import numpy as np
import logging
from mcts.node import Node
from config import EXPLORATION_WEIGHT

logger = logging.getLogger(__name__)

def expand_node(node, priors, add_noise=False, alpha=0.3, epsilon=0.25):
    """
    Expand a node with possible actions and their prior probabilities.
    This version is game-agnostic and works with any GameState implementation.
    
    Args:
        node: Node to expand
        priors: Policy vector of action probabilities
        add_noise: Whether to add Dirichlet noise at root
        alpha: Dirichlet noise parameter
        epsilon: Noise weight factor
    """
    # Get legal actions and policy size
    actions = node.state.get_legal_actions()
    policy_size = node.state.policy_size if hasattr(node.state, 'policy_size') else len(priors)
    
    if not actions:
        return  # Terminal state, can't expand
    
    # Extract priors for legal actions
    legal_priors = []
    for action in actions:
        # Handle case where action index is out of bounds
        if action < len(priors):
            legal_priors.append(priors[action])
        else:
            logger.warning(f"Action {action} out of bounds for priors length {len(priors)}")
            legal_priors.append(0.0)
    
    # Normalize priors
    prior_sum = sum(legal_priors)
    if prior_sum > 0:
        legal_priors = [p / prior_sum for p in legal_priors]
    else:
        # If all priors are 0, use uniform distribution
        legal_priors = [1.0 / len(actions) for _ in actions]
    
    # Add Dirichlet noise at root for exploration
    if add_noise and node.parent is None:
        # Generate Dirichlet noise
        try:
            noise = np.random.dirichlet([alpha] * len(actions))
        except Exception as e:
            logger.error(f"Error generating Dirichlet noise: {e}")
            noise = np.ones(len(actions)) / len(actions)  # Fallback to uniform
            
        legal_priors = [(1 - epsilon) * p + epsilon * n for p, n in zip(legal_priors, noise)]
    
    # Create children nodes
    for i, action in enumerate(actions):
        # Apply action to get new state
        child_state = node.state.apply_action(action)
        # Create child node
        child = Node(child_state, parent=node, prior=legal_priors[i], action=action)
        # Add to parent's children
        node.children.append(child)

def select_node(node, exploration_weight=1.4):
    """
    Select a leaf node from the tree using PUCT algorithm.
    Game-agnostic version without any hardcoded parameters.
    
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

def backpropagate(path, value):
    """
    Update statistics for nodes in the path.
    Game-agnostic version without any hardcoded parameters.
    
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

def mcts_search(root_state, inference_fn, num_simulations, exploration_weight=1.4):
    """
    Game-agnostic MCTS search that works with any GameState implementation.
    
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
    try:
        policy, value = inference_fn(root_state)
    except Exception as e:
        logger.error(f"Error getting root policy and value: {e}")
        # Generate fallback policy
        policy_size = root_state.policy_size if hasattr(root_state, 'policy_size') else 9
        policy = np.ones(policy_size) / policy_size
        value = 0.0
    
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
            try:
                policy, value = inference_fn(leaf.state)
                expand_node(leaf, policy, add_noise=False)
            except Exception as e:
                logger.error(f"Error expanding leaf: {e}")
                # Use fallback values
                policy_size = leaf.state.policy_size if hasattr(leaf.state, 'policy_size') else 9
                policy = np.ones(policy_size) / policy_size
                value = 0.0
                expand_node(leaf, policy, add_noise=False)
        
        # Backpropagation phase - update statistics up the tree
        backpropagate(path, value)
    
    return root