# mcts/search.py
import ray
import numpy as np
from mcts.node import Node

# Simple Node class if you need to update it
# class Node:
#    __slots__ = ('state', 'parent', 'children', 'visits', 'value', 'prior', 'action')
#    
#    def __init__(self, state, parent=None, prior=0.0, action=None):
#        self.state = state
#        self.parent = parent
#        self.children = []
#        self.visits = 0
#        self.value = 0.0
#        self.prior = prior
#        self.action = action

def select_node(node, exploration_weight=1.0):
    """Select a leaf node from the tree using PUCT algorithm"""
    path = [node]
    
    while node.children:
        # Calculate UCB score for all children
        best_score = float('-inf')
        best_child = None
        
        for child in node.children:
            # Skip if no visits (shouldn't happen normally)
            parent_visits = max(1, node.visits)
            
            if child.visits == 0:
                # Prioritize unexplored nodes
                score = float('inf')
            else:
                # PUCT formula
                q_value = child.value / child.visits
                u_value = exploration_weight * child.prior * np.sqrt(parent_visits) / (1 + child.visits)
                score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        node = best_child
        path.append(node)
    
    return node, path

def expand_node(node, priors, add_noise=False, alpha=0.3, epsilon=0.25):
    """Expand a node with possible actions and their prior probabilities"""
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
    """Update statistics for nodes in the path"""
    # For each node in the path (from leaf to root)
    for node in reversed(path):
        # Update node statistics
        node.visits += 1
        node.value += value
        
        # Flip value perspective for opponent
        value = -value

@ray.remote
def mcts_search(root_state, inference_actor, num_simulations, exploration_weight=1.0):
    """Perform MCTS search from root state"""
    # Create root node
    root = Node(root_state)
    
    # Get initial policy and value for root
    policy, value = ray.get(inference_actor.infer.remote(root_state))
    
    # Expand root with initial policy
    expand_node(root, policy, add_noise=True)
    root.value = value
    root.visits = 1
    
    # Run simulations
    for _ in range(num_simulations):
        # Select a leaf node
        leaf, path = select_node(root, exploration_weight)
        
        # Evaluate leaf
        if leaf.state.is_terminal():
            # Use game result if terminal
            value = leaf.state.winner if leaf.state.winner is not None else 0
        else:
            # Otherwise use neural network
            policy, value = ray.get(inference_actor.infer.remote(leaf.state))
            # Expand the leaf node
            expand_node(leaf, policy)
        
        # Backpropagate value
        backpropagate(path, value)
    
    return root

def parallel_mcts(root_state, inference_actor, simulations_per_worker, num_workers, 
                 exploration_weight=1.0, temperature=1.0, return_action_probs=True):
    """Run multiple MCTS searches in parallel and combine results"""
    # Launch parallel searches
    root_futures = [
        mcts_search.remote(root_state, inference_actor, simulations_per_worker, exploration_weight)
        for _ in range(num_workers)
    ]
    
    # Get results
    roots = ray.get(root_futures)
    
    # Extract action visit counts from all searches
    action_visits = {}
    
    for root in roots:
        for child in root.children:
            action = child.action
            if action in action_visits:
                action_visits[action] += child.visits
            else:
                action_visits[action] = child.visits
    
    # Convert to numpy arrays for processing
    actions = list(action_visits.keys())
    visits = np.array([action_visits[a] for a in actions])
    
    # Apply temperature
    if temperature == 0:
        # Deterministic selection (argmax)
        best_idx = np.argmax(visits)
        action = actions[best_idx]
    else:
        # Apply temperature to visit counts
        if temperature == 1.0:
            probs = visits / np.sum(visits)
        else:
            # Apply temperature scaling
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)
        
        # Sample action based on distribution
        action = np.random.choice(actions, p=probs)
    
    # Create policy vector for training
    policy = np.zeros(len(root_state.get_legal_actions()))
    legal_actions = root_state.get_legal_actions()
    for i, a in enumerate(actions):
        try:
            policy[legal_actions.index(a)] = visits[i] / np.sum(visits)
        except ValueError:
            # In case there's some mismatch
            pass
    
    if return_action_probs:
        return action, policy
    else:
        return action

# Legacy API compatibility function
def mcts_worker(root_state, inference_actor, num_simulations):
    """Simulate a legacy MCTS worker function"""
    return mcts_search(root_state, inference_actor, num_simulations)