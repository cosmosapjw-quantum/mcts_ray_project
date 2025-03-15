# mcts/search.py
"""
Parallel and distributed MCTS implementations using Ray.
This module provides scalable search algorithms for multi-core
and distributed execution with batch processing.
"""
import ray
import numpy as np
import time
from mcts.node import Node
from mcts.core import expand_node, backpropagate, select_node
from config import EXPLORATION_WEIGHT, MCTS_BATCH_SIZE

@ray.remote
class BatchMCTSWorker:
    """
    MCTS worker that processes leaves in batches using batch inference.
    
    This enables efficient use of GPUs for inference by batching 
    multiple states together.
    """
    
    def __init__(self, inference_actor, batch_size=MCTS_BATCH_SIZE, exploration_weight=EXPLORATION_WEIGHT):
        """
        Initialize the MCTS worker.
        
        Args:
            inference_actor: Ray actor handling neural network inference
            batch_size: Maximum batch size for inference
            exploration_weight: Controls exploration vs exploitation
        """
        self.inference_actor = inference_actor
        self.batch_size = batch_size
        self.exploration_weight = exploration_weight
    
    def search(self, root_state, num_simulations, add_noise=True):
        """
        Perform MCTS search from root state with batch leaf evaluation.
        
        Args:
            root_state: Initial game state
            num_simulations: Number of simulations to run
            add_noise: Whether to add Dirichlet noise at root
            
        Returns:
            dict: Search results containing action visit counts
        """
        # Create root node
        root = Node(root_state)
        
        try:
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
                    value = leaf.state.get_winner() if leaf.state.is_terminal() else 0
                    backpropagate(path, value)
                    sims_completed += 1
                
                # Process non-terminal leaves with batch inference
                if leaves:
                    # Get all states for batch inference
                    states = [leaf.state for leaf in leaves]
                    
                    try:
                        # Use batch_infer to evaluate all at once
                        results = ray.get(self.inference_actor.batch_infer.remote(states))
                        
                        # Process results and backpropagate
                        for leaf, path, result in zip(leaves, paths, results):
                            policy, value = result
                            expand_node(leaf, policy)
                            backpropagate(path, value)
                            sims_completed += 1
                            
                    except Exception as e:
                        # If batch inference fails, fall back to processing one at a time
                        print(f"Batch inference failed, processing leaves individually: {e}")
                        for leaf, path in zip(leaves, paths):
                            try:
                                policy, value = ray.get(self.inference_actor.infer.remote(leaf.state))
                                expand_node(leaf, policy)
                                backpropagate(path, value)
                            except Exception as inner_e:
                                # Use default values if individual inference fails
                                print(f"Individual inference failed: {inner_e}")
                                # Use uniform policy and neutral value as fallback
                                policy = np.ones(9) / 9  # Assuming TicTacToe with 9 positions
                                value = 0.0
                                expand_node(leaf, policy)
                                backpropagate(path, value)
                            sims_completed += 1
            
            # Extract action visit counts instead of returning the full tree
            result = {"actions": {}, "visits": {}}
            for child in root.children:
                result["actions"][child.action] = child.visits
            
            return result
            
        except Exception as e:
            # Return partial results if search fails
            print(f"MCTS search failed: {e}")
            # Create a minimal result that can be properly serialized
            result = {"actions": {}, "visits": {}, "error": str(e)}
            
            # Add any children if they exist
            if hasattr(root, 'children'):
                for child in root.children:
                    if hasattr(child, 'action') and hasattr(child, 'visits'):
                        result["actions"][child.action] = child.visits
            
            return result

# Simple MCTS worker function that doesn't require a Ray actor
@ray.remote
def mcts_worker_task(root_state, inference_actor, num_simulations, batch_size=MCTS_BATCH_SIZE):
    """
    Task-based MCTS worker function that doesn't use Ray actors.
    This should be more stable than actor-based workers.
    
    Args:
        root_state: Initial game state
        inference_actor: Ray actor for neural network inference
        num_simulations: Number of simulations to run
        batch_size: Maximum batch size for inference
        
    Returns:
        dict: Dictionary mapping actions to visit counts
    """
    try:
        # Use batched_mcts_search but return only action visit counts
        root = batched_mcts_search(
            root_state,
            inference_actor,
            num_simulations,
            batch_size=batch_size,
            verbose=False
        )
        
        # Extract action visit counts
        action_visits = {}
        for child in root.children:
            action_visits[child.action] = child.visits
            
        return action_visits
    except Exception as e:
        print(f"MCTS worker task failed: {e}")
        return {}

def parallel_mcts(root_state, inference_actor, simulations_per_worker, num_workers, 
                 exploration_weight=EXPLORATION_WEIGHT, temperature=1.0, return_action_probs=True):
    """
    Run parallel MCTS searches and aggregate results using Ray tasks instead of actors.
    
    This uses root parallelization where multiple independent MCTS 
    searches are run in parallel and their results are combined.
    
    Args:
        root_state: Initial game state
        inference_actor: Ray actor for neural network inference
        simulations_per_worker: Total simulations to distribute across workers
        num_workers: Number of parallel workers to use
        exploration_weight: Controls exploration vs exploitation
        temperature: Temperature for action selection
        return_action_probs: Whether to return action probabilities for training
        
    Returns:
        tuple or int: (action, policy) if return_action_probs else action
    """
    try:
        # Limit the number of workers to avoid overwhelming the system
        actual_workers = min(num_workers, 4)  # Maximum 4 workers for stability
        
        # Distribute simulations across workers
        sims_per_worker = max(1, simulations_per_worker // actual_workers)
        
        # Submit search tasks with simpler Ray tasks
        task_futures = []
        for _ in range(actual_workers):
            future = mcts_worker_task.remote(
                root_state, 
                inference_actor, 
                sims_per_worker, 
                batch_size=MCTS_BATCH_SIZE
            )
            task_futures.append(future)
        
        # Get results with timeout
        try:
            # First try getting all results at once
            results = ray.get(task_futures, timeout=30.0)
        except Exception as e:
            print(f"Parallel tasks timed out, trying individually: {e}")
            # If that fails, try getting them individually
            results = []
            for future in task_futures:
                try:
                    result = ray.get(future, timeout=10.0)
                    results.append(result)
                except Exception as inner_e:
                    print(f"Individual task failed or timed out: {inner_e}")
        
        # Aggregate action visits
        action_visits = {}
        for result in results:
            for action, visits in result.items():
                if action in action_visits:
                    action_visits[action] = action_visits[action] + visits
                else:
                    action_visits[action] = visits
        
        # If we got no results, fall back to batched search
        if not action_visits:
            print("No valid results from parallel tasks, falling back to batched search")
            root = batched_mcts_search(
                root_state,
                inference_actor,
                simulations_per_worker,
                batch_size=MCTS_BATCH_SIZE
            )
            for child in root.children:
                action_visits[child.action] = child.visits
                
        if not action_visits:
            raise ValueError("Failed to get any valid action visits")
        
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
        policy_size = getattr(root_state, 'policy_size', 9)  # Default to 9 for TicTacToe
        policy = np.zeros(policy_size)
        
        # Fill policy vector
        total_visits = np.sum(visits)
        if total_visits > 0:
            for i, a in enumerate(actions):
                policy[a] = visits[i] / total_visits
        
        if return_action_probs:
            return action, policy
        else:
            return action
            
    except Exception as e:
        print(f"Task-based parallel MCTS failed: {e}")
        print("Falling back to single batched search")
        
        # Fall back to a simple batched search
        try:
            root = batched_mcts_search(
                root_state,
                inference_actor,
                simulations_per_worker,  # Use all simulations in one worker
                batch_size=MCTS_BATCH_SIZE
            )
            
            # Extract visit counts
            visits = np.array([child.visits for child in root.children])
            actions = [child.action for child in root.children]
            
            # Apply temperature and select action
            if temperature == 0:
                best_idx = np.argmax(visits)
                action = actions[best_idx]
            else:
                if temperature == 1.0:
                    probs = visits / np.sum(visits)
                else:
                    visits_temp = visits ** (1.0 / temperature)
                    probs = visits_temp / np.sum(visits_temp)
                action = np.random.choice(actions, p=probs)
            
            # Create policy vector
            policy_size = getattr(root_state, 'policy_size', 9)
            policy = np.zeros(policy_size)
            total_visits = np.sum(visits)
            if total_visits > 0:
                for i, a in enumerate(actions):
                    policy[a] = visits[i] / total_visits
                    
            if return_action_probs:
                return action, policy
            else:
                return action
                
        except Exception as final_e:
            print(f"Single batched search failed too: {final_e}")
            # Last resort: random legal move
            legal_actions = root_state.get_legal_actions()
            if not legal_actions:
                raise ValueError("No legal actions available")
                
            # Use uniform random policy as fallback
            action = np.random.choice(legal_actions)
            policy = np.zeros(getattr(root_state, 'policy_size', 9))
            for a in legal_actions:
                policy[a] = 1.0 / len(legal_actions)
                
            if return_action_probs:
                return action, policy
            else:
                return action

def batched_mcts_search(root_state, inference_actor, num_simulations, 
                        batch_size=MCTS_BATCH_SIZE, exploration_weight=EXPLORATION_WEIGHT, 
                        verbose=False):
    """
    Perform batched MCTS search using a single worker.
    
    This version collects leaves for batch evaluation without 
    parallelizing at the root level.
    
    Args:
        root_state: Initial game state
        inference_actor: Ray actor for neural network inference
        num_simulations: Number of simulations to run
        batch_size: Maximum batch size for inference
        exploration_weight: Controls exploration vs exploitation
        verbose: Whether to print batch statistics
        
    Returns:
        Node: Root node of the search tree
    """
    try:
        # Create root node
        root = Node(root_state)
        
        # Get initial policy and value for root
        policy, value = ray.get(inference_actor.infer.remote(root_state), timeout=10.0)
        
        # Expand root with initial policy
        expand_node(root, policy, add_noise=True)
        root.value = value
        root.visits = 1
        
        # Track timings for debugging
        start_time = time.time()
        inference_times = []
        batch_sizes = []
        
        # Run simulations
        remaining_sims = num_simulations - 1  # -1 for root expansion
        
        while remaining_sims > 0:
            # Collect leaves for evaluation
            leaves = []
            paths = []
            terminal_leaves = []
            terminal_paths = []
            
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, remaining_sims)
            
            # Select leaves until batch is full
            batch_collection_start = time.time()
            while len(leaves) + len(terminal_leaves) < current_batch_size:
                leaf, path = select_node(root, exploration_weight)
                
                if leaf.state.is_terminal():
                    terminal_leaves.append(leaf)
                    terminal_paths.append(path)
                else:
                    leaves.append(leaf)
                    paths.append(path)
                    
                # If we've collected enough leaves, or if there are no more unexpanded nodes
                if len(leaves) + len(terminal_leaves) >= current_batch_size:
                    break
            
            if verbose:
                batch_collection_time = time.time() - batch_collection_start
                if leaves:
                    print(f"  Collected {len(leaves)} non-terminal leaves in {batch_collection_time:.3f}s")
            
            # Process terminal states immediately
            for leaf, path in zip(terminal_leaves, terminal_paths):
                value = leaf.state.get_winner() if leaf.state.is_terminal() else 0
                backpropagate(path, value)
                remaining_sims -= 1
            
            # Process non-terminal leaves with batch inference
            if leaves:
                # Get leaf states
                states = [leaf.state for leaf in leaves]
                batch_sizes.append(len(states))
                
                # Track inference time
                inference_start = time.time()
                
                try:
                    # Batch inference with timeout
                    results = ray.get(inference_actor.batch_infer.remote(states), timeout=10.0)
                    
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)
                    
                    if verbose:
                        print(f"  Batch inference for {len(states)} states took {inference_time:.3f}s")
                    
                    # Process results
                    for leaf, path, result in zip(leaves, paths, results):
                        policy, value = result
                        expand_node(leaf, policy)
                        backpropagate(path, value)
                        remaining_sims -= 1
                        
                except Exception as e:
                    # Handle batch inference failure by falling back to individual inference
                    print(f"Batch inference failed: {e}")
                    for leaf, path in zip(leaves, paths):
                        try:
                            # Try individual inference with timeout
                            policy, value = ray.get(inference_actor.infer.remote(leaf.state), timeout=5.0)
                        except Exception as inner_e:
                            # Use default values if individual inference fails
                            print(f"Individual inference failed: {inner_e}")
                            policy = np.ones(9) / 9  # Uniform policy for TicTacToe
                            value = 0.0
                        
                        expand_node(leaf, policy)
                        backpropagate(path, value)
                        remaining_sims -= 1
        
        # Print statistics if verbose
        if verbose:
            total_time = time.time() - start_time
            avg_batch_size = sum(batch_sizes) / max(1, len(batch_sizes))
            avg_inference_time = sum(inference_times) / max(1, len(inference_times))
            
            print(f"MCTS completed {num_simulations} simulations in {total_time:.2f}s")
            print(f"Average batch size: {avg_batch_size:.1f}")
            print(f"Average inference time: {avg_inference_time:.3f}s")
        
        return root
        
    except Exception as e:
        print(f"Batched MCTS search failed: {e}")
        # Return a minimal valid root node
        root = Node(root_state)
        
        # Add at least one valid child if possible
        legal_actions = root_state.get_legal_actions()
        if legal_actions:
            for action in legal_actions:
                child_state = root_state.apply_action(action)
                child = Node(child_state, parent=root, action=action)
                child.prior = 1.0 / len(legal_actions)
                child.visits = 1
                root.children.append(child)
            
            # Ensure root has valid statistics
            root.visits = sum(child.visits for child in root.children)
        
        return root

def mcts_with_timeout(root_state, inference_actor, max_time_seconds, 
                     batch_size=MCTS_BATCH_SIZE, exploration_weight=EXPLORATION_WEIGHT):
    """
    Run MCTS search with a time limit instead of simulation count.
    
    Args:
        root_state: Initial game state
        inference_actor: Ray actor for neural network inference
        max_time_seconds: Maximum time to run search in seconds
        batch_size: Maximum batch size for inference
        exploration_weight: Controls exploration vs exploitation
        
    Returns:
        Node: Root node of the search tree
    """
    try:
        # Create root node
        root = Node(root_state)
        
        # Get initial policy and value
        policy, value = ray.get(inference_actor.infer.remote(root_state), timeout=10.0)
        
        # Expand root with initial policy
        expand_node(root, policy, add_noise=True)
        root.value = value
        root.visits = 1
        
        # Set end time
        end_time = time.time() + max_time_seconds
        sim_count = 1  # Count root evaluation
        
        # Run until time limit is reached
        while time.time() < end_time:
            # Determine batch size - smaller as we get closer to the end time
            time_left = end_time - time.time()
            if time_left <= 0:
                break
                
            current_batch_size = min(batch_size, max(1, int(batch_size * time_left / max_time_seconds)))
            
            # Collect leaves for evaluation
            leaves = []
            paths = []
            terminal_leaves = []
            terminal_paths = []
            
            # Select leaves until batch is full or time is up
            leaves_start_time = time.time()
            leaves_timeout = min(time_left, 1.0)  # Max 1 second for leaf collection
            
            while (len(leaves) + len(terminal_leaves) < current_batch_size and 
                   time.time() - leaves_start_time < leaves_timeout):
                if time.time() >= end_time:
                    break
                    
                leaf, path = select_node(root, exploration_weight)
                
                if leaf.state.is_terminal():
                    terminal_leaves.append(leaf)
                    terminal_paths.append(path)
                else:
                    leaves.append(leaf)
                    paths.append(path)
            
            # Process terminal states
            for leaf, path in zip(terminal_leaves, terminal_paths):
                value = leaf.state.get_winner() if leaf.state.is_terminal() else 0
                backpropagate(path, value)
                sim_count += 1
            
            # Check if we're out of time
            if time.time() >= end_time:
                break
            
            # Process non-terminal leaves with batch inference
            if leaves:
                # Get leaf states
                states = [leaf.state for leaf in leaves]
                
                try:
                    # Calculate remaining time for inference
                    inference_timeout = end_time - time.time()
                    if inference_timeout <= 0:
                        break
                        
                    # Batch inference with timeout
                    results = ray.get(inference_actor.batch_infer.remote(states), timeout=inference_timeout)
                    
                    # Process results
                    for leaf, path, result in zip(leaves, paths, results):
                        policy, value = result
                        expand_node(leaf, policy)
                        backpropagate(path, value)
                        sim_count += 1
                        
                except Exception as e:
                    print(f"Time-based batch inference failed: {e}")
                    # Skip processing rather than risk timing out on individual inferences
        
        print(f"Completed {sim_count} simulations in {max_time_seconds:.2f}s")
        return root
        
    except Exception as e:
        print(f"Time-based MCTS search failed: {e}")
        # Return a minimal valid root node
        root = Node(root_state)
        
        # Add at least one valid child if possible
        legal_actions = root_state.get_legal_actions()
        if legal_actions:
            for action in legal_actions:
                child_state = root_state.apply_action(action)
                child = Node(child_state, parent=root, action=action)
                child.prior = 1.0 / len(legal_actions)
                child.visits = 1
                root.children.append(child)
            
            # Ensure root has valid statistics
            root.visits = sum(child.visits for child in root.children)
        
        return root