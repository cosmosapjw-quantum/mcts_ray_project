# utils/mcts_utils.py
"""
Utility functions for MCTS operations.
"""
import numpy as np
import logging
from utils.game_interface import GameState
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

def create_uniform_policy(game_state: GameState) -> np.ndarray:
    """
    Create a uniform policy over legal actions for any game state.
    
    Args:
        game_state: Game state to create policy for
        
    Returns:
        np.ndarray: Uniform policy vector
    """
    # Get policy size and legal actions
    policy_size = game_state.policy_size if hasattr(game_state, 'policy_size') else 9
    legal_actions = game_state.get_legal_actions()
    
    # Create zero policy
    policy = np.zeros(policy_size, dtype=np.float32)
    
    # If no legal actions, return uniform over all actions
    if not legal_actions:
        return np.ones(policy_size, dtype=np.float32) / policy_size
    
    # Set uniform probability for legal actions
    uniform_prob = 1.0 / len(legal_actions)
    for action in legal_actions:
        policy[action] = uniform_prob
    
    return policy

def create_prioritized_policy(game_state: GameState, heuristic_func=None) -> np.ndarray:
    """
    Create a policy with prioritized weights based on a heuristic function.
    
    Args:
        game_state: Game state to create policy for
        heuristic_func: Function that takes (state, action) and returns a score
                       (higher is better)
        
    Returns:
        np.ndarray: Prioritized policy vector
    """
    # Get policy size and legal actions
    policy_size = game_state.policy_size if hasattr(game_state, 'policy_size') else 9
    legal_actions = game_state.get_legal_actions()
    
    # Create zero policy
    policy = np.zeros(policy_size, dtype=np.float32)
    
    # If no heuristic or no legal actions, return uniform
    if heuristic_func is None or not legal_actions:
        return create_uniform_policy(game_state)
    
    # Calculate scores for each action
    scores = []
    for action in legal_actions:
        try:
            score = heuristic_func(game_state, action)
            scores.append((action, score))
        except Exception as e:
            logger.error(f"Error calculating heuristic for action {action}: {e}")
            scores.append((action, 0.0))
    
    # Normalize scores with softmax
    if scores:
        actions, values = zip(*scores)
        values = np.array(values, dtype=np.float32)
        
        # Apply softmax scaling
        if np.max(values) - np.min(values) > 1e-10:  # Avoid division by zero
            values = np.exp(values - np.max(values))  # Subtract max for numerical stability
            probs = values / np.sum(values)
        else:
            # If all scores are equal, use uniform
            probs = np.ones_like(values) / len(values)
            
        # Fill policy
        for action, prob in zip(actions, probs):
            policy[action] = prob
    
    return policy

def default_fallback_policy(state: GameState, inference_fn, error_context: str = "") -> tuple:
    """
    Create a fallback policy and value for when inference fails.
    
    Args:
        state: Game state to create policy for
        inference_fn: The inference function that failed
        error_context: Additional context about the error
        
    Returns:
        tuple: (policy, value) fallback
    """
    logger.warning(f"Using fallback policy due to inference error: {error_context}")
    
    # Create uniform policy
    policy = create_uniform_policy(state)
    
    # Use neutral value 
    value = 0.0
    
    return policy, value

def apply_temperature(visits, actions, temperature=1.0):
    """
    Apply temperature to visit counts and select an action.
    Game-agnostic version that doesn't hardcode policy size.
    
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
        logger.error(f"Error sampling from probs: {e}")
        logger.error(f"Probs: {probs}, sum: {np.sum(probs)}")
        # Fall back to argmax
        best_idx = np.argmax(visits)
        return actions[best_idx], probs

def apply_gradual_temperature(state: GameState, visits: np.ndarray, actions: List[int], 
                            temperature: float = 1.0, decay_strength: float = 2.0) -> Tuple[int, np.ndarray]:
    """
    Apply temperature to visits using a gradual spatial decay function.
    
    This creates a distribution that:
    - At temperature = 0, selects the maximum visit count deterministically
    - At temperature > 0, creates a spatially-aware distribution where:
      - The probability peaks at the maximum visit count
      - Probability decreases with distance from the peak
      - The rate of decrease is controlled by temperature (higher = slower decay)
    
    Args:
        state: Game state (used to calculate move distances)
        visits: Array of visit counts
        actions: List of available actions
        temperature: Temperature parameter (0 = deterministic, higher = more exploration)
        decay_strength: Controls the rate of spatial decay (higher = faster decay)
        
    Returns:
        tuple: (selected_action, action_probabilities)
    """
    # Safety checks
    if len(actions) == 0:
        raise ValueError("No actions provided")
    
    if len(actions) == 1:
        return actions[0], np.array([1.0])
    
    # Deterministic case
    if temperature == 0:
        best_idx = np.argmax(visits)
        probs = np.zeros_like(visits, dtype=float)
        probs[best_idx] = 1.0
        return actions[best_idx], probs
    
    # Get the best move by visit count for peak probability
    best_idx = np.argmax(visits)
    best_action = actions[best_idx]
    
    # Create a distance matrix between moves
    distances = np.zeros(len(actions), dtype=float)
    for i, action in enumerate(actions):
        if hasattr(state, 'calculate_move_distance'):
            # Use game-specific distance calculation
            distances[i] = state.calculate_move_distance(best_action, action)
        else:
            # Default distance (0 for same action, 1 for different)
            distances[i] = 0.0 if action == best_action else 1.0
    
    # Calculate probabilities using a spatial decay function
    # Invert temperature for computation (higher temp = slower decay)
    # We use the original visits as a base for weighting
    visit_weights = visits / np.sum(visits) if np.sum(visits) > 0 else np.ones_like(visits) / len(visits)
    
    # Scale temperature for appropriate control (0.1 to 5.0 is a good range)
    scaled_temp = max(0.1, min(5.0, temperature))
    
    # Calculate decay factor based on visits and distance
    # visits_factor: Weight by visits (higher visits = higher probability)
    # distance_factor: Decrease probability with distance, controlled by temperature
    probs = visit_weights * np.exp(-decay_strength * distances / scaled_temp)
    
    # Normalize to get a proper probability distribution
    probs_sum = np.sum(probs)
    if probs_sum <= 0 or np.isnan(probs).any():
        # Fallback to standard visit-based distribution
        logger.warning("Gradual temperature calculation failed, falling back to standard method")
        probs = visits / np.sum(visits) if np.sum(visits) > 0 else np.ones_like(visits) / len(visits)
    else:
        probs = probs / probs_sum
    
    # Sample from the resulting distribution
    try:
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx], probs
    except ValueError as e:
        # Fallback if sampling fails
        logger.error(f"Error sampling from gradual temperature distribution: {e}")
        return actions[best_idx], probs

def visualize_temperature_distribution(state: GameState, visits: np.ndarray, actions: List[int], 
                                    temperature: float, decay_strength: float = 2.0) -> Dict[str, Any]:
    """
    Visualize the effect of temperature on action probabilities.
    For 2D games, returns a matrix representation of the distribution.
    
    Args:
        state: Game state
        visits: Visit counts
        actions: List of available actions
        temperature: Temperature value
        decay_strength: Decay strength parameter
        
    Returns:
        dict: Visualization data
    """
    # Apply gradual temperature to get probabilities
    _, probs = apply_gradual_temperature(state, visits, actions, temperature, decay_strength)
    
    # Create result data
    result = {
        "probabilities": dict(zip(actions, probs)),
        "temperature": temperature,
        "decay_strength": decay_strength,
        "max_prob": float(np.max(probs)),
        "entropy": float(-np.sum(probs * np.log(probs + 1e-10)))
    }
    
    # For TicTacToe, create a visual grid
    if hasattr(state, 'BOARD_SIZE') and state.BOARD_SIZE == 3:
        grid = np.zeros((3, 3), dtype=float)
        for i, action in enumerate(actions):
            grid[action // 3, action % 3] = probs[i]
        result["grid"] = grid.tolist()
    
    # For Connect Four, create a column distribution
    elif hasattr(state, 'COLS') and state.COLS == 7:
        column_probs = np.zeros(7, dtype=float)
        for i, action in enumerate(actions):
            column_probs[action] = probs[i]
        result["column_distribution"] = column_probs.tolist()
    
    return result

def visits_to_policy(visits, actions, policy_size=None, state=None):
    """
    Convert visit counts to a policy vector.
    Game-agnostic version that handles different policy sizes.
    
    Args:
        visits: Array of visit counts for each action
        actions: List of available actions
        policy_size: Size of the policy vector (or None to detect from state)
        state: Game state to determine policy size (optional)
        
    Returns:
        np.array: Full policy vector
    """
    # Determine policy size
    if policy_size is None:
        if state is not None:
            policy_size = state.policy_size if hasattr(state, 'policy_size') else 9
        else:
            # Estimate from maximum action index
            policy_size = max(actions) + 1 if actions else 9
    
    # Create policy vector
    policy = np.zeros(policy_size, dtype=np.float32)
    
    # Fill with normalized visit counts
    total_visits = np.sum(visits)
    if total_visits > 0:
        for i, action in enumerate(actions):
            if action < policy_size:
                policy[action] = visits[i] / total_visits
            else:
                logger.warning(f"Action {action} out of bounds for policy size {policy_size}")
    else:
        # If no visits (shouldn't happen normally), use uniform policy
        for action in actions:
            if action < policy_size:
                policy[action] = 1.0 / len(actions)
            else:
                logger.warning(f"Action {action} out of bounds for policy size {policy_size}")
    
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