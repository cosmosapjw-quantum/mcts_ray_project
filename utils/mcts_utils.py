# utils/mcts_utils.py
"""
Utility functions for MCTS operations.
"""
import numpy as np

def apply_temperature(visits, actions, temperature=1.0):
    """
    Apply temperature to visit counts and select an action
    
    Args:
        visits: Array of visit counts for each action
        actions: List of available actions
        temperature: Temperature parameter controlling exploration
        
    Returns:
        tuple: (selected_action, action_probabilities)
    """
    # Safety check for empty actions
    if len(actions) == 0:
        raise ValueError("No actions provided to apply_temperature")
    
    # For single action, return it immediately
    if len(actions) == 1:
        return actions[0], np.array([1.0])
    
    if temperature == 0:
        # Deterministic selection (argmax)
        best_idx = np.argmax(visits)
        probs = np.zeros_like(visits, dtype=float)
        probs[best_idx] = 1.0
        return actions[best_idx], probs
    
    # Apply temperature scaling
    if temperature == 1.0:
        probs = visits / np.sum(visits)
    else:
        # Temperature scaling
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / np.sum(visits_temp)
    
    # Handle numerical issues
    if np.isnan(probs).any() or np.sum(probs) == 0:
        # Fall back to uniform distribution
        probs = np.ones_like(visits, dtype=float) / len(visits)
    
    # Sample action based on probabilities
    try:
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx], probs
    except ValueError as e:
        # Debug information for sampling errors
        print(f"Error sampling from probs: {e}")
        print(f"Probs: {probs}, sum: {np.sum(probs)}")
        # Fall back to argmax
        best_idx = np.argmax(visits)
        return actions[best_idx], probs

def visits_to_policy(visits, actions, board_size):
    """
    Convert visit counts to a policy vector
    
    Args:
        visits: Array of visit counts for each action
        actions: List of available actions
        board_size: Total size of the policy vector
        
    Returns:
        np.array: Full policy vector
    """
    policy = np.zeros(board_size)
    total_visits = np.sum(visits)
    if total_visits > 0:
        for i, action in enumerate(actions):
            policy[action] = visits[i] / total_visits
    else:
        # If no visits (shouldn't happen normally), use uniform policy
        for action in actions:
            policy[action] = 1.0 / len(actions)
    return policy

def get_temperature(move_count, temperature_schedule):
    """
    Get temperature based on move count and schedule
    
    Args:
        move_count: Current move count in the game
        temperature_schedule: Dictionary mapping move thresholds to temperatures
        
    Returns:
        float: Temperature to use for the current move
    """
    temperature = 1.0  # Default temperature
    
    # Apply temperature schedule (if provided)
    if temperature_schedule:
        for move_threshold, temp in sorted(temperature_schedule.items()):
            if move_count >= move_threshold:
                temperature = temp
    
    return temperature

def get_action_distribution(root):
    """
    Extract the action distribution from a root node
    
    Args:
        root: Root node of the MCTS search tree
        
    Returns:
        tuple: (actions, visits, probabilities)
    """
    # Extract visit counts and actions
    visits = np.array([child.visits for child in root.children])
    actions = [child.action for child in root.children]
    
    # Convert to probabilities
    total_visits = np.sum(visits)
    if total_visits > 0:
        probs = visits / total_visits
    else:
        probs = np.ones_like(visits) / len(visits)
    
    return actions, visits, probs

def ucb_score(node, parent_visits, exploration_weight=1.4):
    """
    Calculate the UCB score for a node
    
    Args:
        node: Node to calculate score for
        parent_visits: Number of visits at the parent node
        exploration_weight: Controls exploration vs exploitation
        
    Returns:
        float: UCB score
    """
    if node.visits == 0:
        return float('inf')
    
    # Q-value (exploitation)
    q_value = node.value / node.visits
    
    # U-value (exploration)
    u_value = exploration_weight * node.prior * np.sqrt(parent_visits) / (1 + node.visits)
    
    return q_value + u_value

def track_mcts_statistics(root):
    """
    Gather statistics about an MCTS tree for debugging
    
    Args:
        root: Root node of the MCTS search tree
        
    Returns:
        dict: Dictionary of statistics
    """
    # Calculate tree depth
    def get_depth(node):
        if not node.children:
            return 0
        return 1 + max(get_depth(child) for child in node.children)
    
    # Calculate number of nodes
    def count_nodes(node):
        if not node.children:
            return 1
        return 1 + sum(count_nodes(child) for child in node.children)
    
    # Get visit distribution
    visits = [child.visits for child in root.children]
    actions = [child.action for child in root.children]
    
    # Calculate statistics
    stats = {
        'tree_depth': get_depth(root),
        'node_count': count_nodes(root),
        'root_visits': root.visits,
        'max_child_visits': max(visits) if visits else 0,
        'min_child_visits': min(visits) if visits else 0,
        'avg_child_visits': sum(visits) / len(visits) if visits else 0,
        'visit_distribution': dict(zip(actions, visits)) if actions else {},
        'value_estimate': root.value / root.visits if root.visits > 0 else 0
    }
    
    return stats