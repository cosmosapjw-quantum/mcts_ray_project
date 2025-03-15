# mcts/search.py
import ray
import numpy as np
from mcts.node import Node

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
class BatchMCTSWorker:
    """MCTS worker that processes leaves in batches using direct batch inference"""
    
    def __init__(self, inference_actor, batch_size=32, exploration_weight=1.0):
        self.inference_actor = inference_actor
        self.batch_size = batch_size
        self.exploration_weight = exploration_weight
    
    def search(self, root_state, num_simulations, add_noise=True):
        """Perform MCTS search from root state with truly batched leaf evaluation"""
        # Create root node
        root = Node(root_state)
        
        # First get root expansion without batching
        policy, value = ray.get(self.inference_actor.infer.remote(root_state))
        
        # Expand root with initial policy
        expand_node(root, policy, add_noise=add_noise)
        root.value = value
        root.visits = 1
        
        # Run simulations in batches
        sims_completed = 1  # Count root evaluation
        
        while sims_completed < num_simulations:
            # How many simulations to run in this batch
            batch_size = min(self.batch_size, num_simulations - sims_completed)
            
            # Collect leaves for batch evaluation
            leaves = []
            paths = []
            terminal_leaves = []
            terminal_paths = []
            
            # Select leaves until batch size or all leaves are terminal
            while len(leaves) + len(terminal_leaves) < batch_size:
                leaf, path = select_node(root, self.exploration_weight)
                
                # Process terminal states immediately
                if leaf.state.is_terminal():
                    terminal_leaves.append(leaf)
                    terminal_paths.append(path)
                else:
                    leaves.append(leaf)
                    paths.append(path)
            
            # Process terminal states
            for leaf, path in zip(terminal_leaves, terminal_paths):
                value = leaf.state.winner if leaf.state.winner is not None else 0
                backpropagate(path, value)
                sims_completed += 1
            
            # Process non-terminal leaves with batch inference
            if leaves:
                # Get all states for batch inference
                states = [leaf.state for leaf in leaves]
                
                # Use batch_infer to evaluate all at once
                results = ray.get(self.inference_actor.batch_infer.remote(states))
                
                # Process results and backpropagate
                for i, (leaf, path, result) in enumerate(zip(leaves, paths, results)):
                    policy, value = result
                    expand_node(leaf, policy)
                    backpropagate(path, value)
                    sims_completed += 1
        
        return root

def parallel_mcts(root_state, inference_actor, simulations_per_worker, num_workers, 
                 exploration_weight=1.0, temperature=1.0, return_action_probs=True):
    """Run parallel MCTS searches with proper batching"""
    # Create workers
    workers = [BatchMCTSWorker.remote(
        inference_actor, 
        batch_size=32  # Force larger batches
    ) for _ in range(num_workers)]
    
    # Distribute simulations across workers
    sims_per_worker = simulations_per_worker // num_workers
    root_futures = [worker.search.remote(
        root_state, sims_per_worker, add_noise=True
    ) for worker in workers]
    
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
    
    # Create policy vector for training - ALWAYS size 9 for TicTacToe
    policy = np.zeros(9)  # Fixed size for all board states
    for i, a in enumerate(actions):
        policy[a] = visits[i] / np.sum(visits)
    
    if return_action_probs:
        return action, policy
    else:
        return action

# Legacy API compatibility function
def mcts_worker(root_state, inference_actor, num_simulations):
    """Legacy compatibility function"""
    worker = BatchMCTSWorker.remote(inference_actor, batch_size=32)
    return ray.get(worker.search.remote(root_state, num_simulations))