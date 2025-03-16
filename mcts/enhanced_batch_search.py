# mcts/enhanced_batch_search.py
"""
Optimized batched MCTS search implementation with improved batch processing
and detailed performance profiling.
"""

import time
import numpy as np
import torch
import logging
from functools import wraps
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Performance timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Save timing info in stats if available
        if 'stats' in kwargs and kwargs['stats'] is not None:
            method_name = func.__name__
            if method_name not in kwargs['stats']:
                kwargs['stats'][method_name] = []
            kwargs['stats'][method_name].append(execution_time)
        
        return result
    return wrapper

class BatchCollector:
    """
    Advanced batch collection for MCTS with dynamic batching strategies.
    
    This class manages batch collection strategies to optimize MCTS
    performance based on runtime statistics.
    """
    def __init__(self, batch_size, adaptive=True, strategy='balanced'):
        """
        Initialize batch collector.
        
        Args:
            batch_size: Maximum batch size
            adaptive: Whether to use adaptive batch sizing
            strategy: Batch collection strategy ('greedy', 'balanced', 'quality')
        """
        self.max_batch_size = batch_size
        self.adaptive = adaptive
        self.strategy = strategy
        
        # Performance tracking
        self.batch_sizes = []
        self.collection_times = []
        self.inference_utilization = []  # Ratio of batch size to max batch size
        
        # Adaptive sizing parameters
        self.min_batch_size = 4
        self.target_wait_time = 0.001  # Target wait time in seconds
        self.current_batch_size = batch_size
        self.consecutive_timeouts = 0
        self.batch_growth_rate = 1.1   # Multiplicative increase
        self.batch_shrink_rate = 0.8   # Multiplicative decrease
        
        logger.info(f"BatchCollector initialized with strategy: {strategy}, "
                    f"adaptive: {adaptive}, max_batch_size: {batch_size}")
    
    @timing_decorator
    def collect_batch(self, root, select_func, stats=None):
        """
        Collect a batch of leaves for evaluation.
        
        Args:
            root: Root node of the MCTS tree
            select_func: Function to select a leaf node
            stats: Dictionary to track timing statistics
            
        Returns:
            tuple: (leaves, paths, terminal_leaves, terminal_paths)
        """
        # Determine current target batch size
        target_batch_size = self.current_batch_size if self.adaptive else self.max_batch_size
        
        # Initialize collection variables
        leaves = []
        paths = []
        terminal_leaves = []
        terminal_paths = []
        
        collection_start = time.time()
        timeout = False
        
        # Select leaves until we reach target batch size or timeout
        while len(leaves) + len(terminal_leaves) < target_batch_size:
            # Apply collection strategy
            if self.strategy == 'greedy':
                # Greedy strategy: Always select highest UCB
                leaf, path = select_func(root)
            elif self.strategy == 'quality':
                # Quality-focused: Select highest UCB but with randomness
                # This helps avoid getting stuck in the same branches
                leaf, path = self._quality_select(root, select_func)
            else:
                # Balanced strategy (default): Standard selection
                leaf, path = select_func(root)
            
            # Process leaf based on whether it's terminal
            if leaf.state.is_terminal():
                terminal_leaves.append(leaf)
                terminal_paths.append(path)
            else:
                leaves.append(leaf)
                paths.append(path)
            
            # Check timeout if adaptive batch sizing is enabled
            if self.adaptive:
                if time.time() - collection_start > self.target_wait_time:
                    timeout = True
                    break
        
        # Update batch collection statistics
        batch_size = len(leaves) + len(terminal_leaves)
        self.batch_sizes.append(batch_size)
        collection_time = time.time() - collection_start
        self.collection_times.append(collection_time)
        
        # Calculate inference utilization
        inference_utilization = batch_size / self.max_batch_size
        self.inference_utilization.append(inference_utilization)
        
        # Adjust batch size if adaptive
        if self.adaptive:
            if timeout:
                # Shrink batch size if we timed out
                self.current_batch_size = max(
                    self.min_batch_size,
                    int(self.current_batch_size * self.batch_shrink_rate)
                )
                self.consecutive_timeouts += 1
            else:
                # Grow batch size if we finished quickly
                if collection_time < self.target_wait_time * 0.5:
                    self.current_batch_size = min(
                        self.max_batch_size,
                        int(self.current_batch_size * self.batch_growth_rate)
                    )
                    self.consecutive_timeouts = 0
        
        # Log batch statistics periodically
        if len(self.batch_sizes) % 50 == 0:
            avg_size = sum(self.batch_sizes[-50:]) / 50
            avg_time = sum(self.collection_times[-50:]) / 50
            avg_util = sum(self.inference_utilization[-50:]) / 50
            logger.debug(f"Batch collection: avg_size={avg_size:.1f}, "
                        f"avg_time={avg_time*1000:.2f}ms, utilization={avg_util:.2f}")
            
            if self.adaptive:
                logger.debug(f"Current target batch size: {self.current_batch_size}")
        
        return leaves, paths, terminal_leaves, terminal_paths
    
    def _quality_select(self, root, select_func):
        """Select with a focus on exploration quality"""
        # 90% of the time, use standard selection
        if np.random.random() < 0.9:
            return select_func(root)
            
        # 10% of the time, randomly select from top nodes
        # This helps ensure we're not stuck in the same branches
        if root.children:
            from mcts.node import Node
            
            # Get UCB scores
            ucb_scores = []
            for child in root.children:
                if child.visits == 0:
                    score = float('inf')
                else:
                    q_value = child.value / child.visits
                    u_value = 1.4 * child.prior * np.sqrt(root.visits) / (1 + child.visits)
                    score = q_value + u_value
                ucb_scores.append(score)
            
            # Sort children by UCB score
            sorted_indices = np.argsort(ucb_scores)[::-1]  # Descending
            
            # Select from top 3 with weighted probability
            top_n = min(3, len(sorted_indices))
            weights = np.array([1.5, 1.0, 0.5][:top_n])
            weights = weights / np.sum(weights)
            
            idx = np.random.choice(range(top_n), p=weights)
            child_idx = sorted_indices[idx]
            
            # Now do regular selection from this child
            child = root.children[child_idx]
            result = select_func(child)
            
            # Add the child to the path
            return result[0], [root] + result[1]
        
        # Fall back to standard selection
        return select_func(root)
    
    def get_stats(self):
        """Get batch collection statistics"""
        if not self.batch_sizes:
            return {}
            
        return {
            "avg_batch_size": sum(self.batch_sizes) / len(self.batch_sizes),
            "avg_collection_time": sum(self.collection_times) / len(self.collection_times),
            "avg_inference_utilization": sum(self.inference_utilization) / len(self.inference_utilization),
            "current_batch_size": self.current_batch_size if self.adaptive else self.max_batch_size,
            "consecutive_timeouts": self.consecutive_timeouts,
            "batch_count": len(self.batch_sizes)
        }

@timing_decorator
def optimized_batched_mcts_search(state, inference_actor, num_simulations, 
                                 batch_size=64, exploration_weight=1.4, 
                                 batch_strategy='balanced',
                                 adaptive_batching=True,
                                 batch_wait_time=0.001,
                                 verbose=False,
                                 collect_stats=False,
                                 stats=None):
    """
    Optimized batched MCTS search with enhanced batch processing and profiling.
    
    This implementation includes:
    - Advanced batch collection strategies
    - Adaptive batch sizing
    - Detailed performance profiling
    - Memory usage optimization
    
    Args:
        state: Initial game state
        inference_actor: Actor for neural network inference
        num_simulations: Number of simulations to run
        batch_size: Maximum batch size for inference
        exploration_weight: Controls exploration vs exploitation
        batch_strategy: Strategy for batch collection ('greedy', 'balanced', 'quality')
        adaptive_batching: Whether to use adaptive batch sizing
        batch_wait_time: Target wait time for batch collection (if adaptive)
        verbose: Whether to print detailed information
        collect_stats: Whether to collect and return detailed statistics
        stats: Dictionary to store timing statistics
        
    Returns:
        tuple: (root, stats_dict) if collect_stats else root
    """
    from mcts.node import Node
    from mcts.core import expand_node, backpropagate
    
    # Initialize timing stats dictionary if requested
    if stats is None and collect_stats:
        stats = {}
    
    # Create batch collector with specified strategy
    batch_collector = BatchCollector(
        batch_size=batch_size,
        adaptive=adaptive_batching,
        strategy=batch_strategy
    )
    
    # Set target wait time if specified
    if batch_wait_time:
        batch_collector.target_wait_time = batch_wait_time
    
    # Create timing stats for detailed profiling
    detailed_timing = {
        "selection": 0.0,
        "inference": 0.0,
        "expansion": 0.0,
        "backpropagation": 0.0,
        "total": 0.0
    }
    
    # Create buffer for stats collection to reduce memory allocations
    if collect_stats:
        stats.update({
            "batch_sizes": [],
            "inference_times": [],
            "selection_times": [],
            "expansion_times": [],
            "backprop_times": []
        })
    
    # Start timing
    search_start = time.time()
    
    # Create root node
    root = Node(state)
    
    # Perform root evaluation
    inference_start = time.time()
    policy, value = inference_actor.infer(state)
    inference_time = time.time() - inference_start
    detailed_timing["inference"] += inference_time
    
    if collect_stats and "inference_times" in stats:
        stats["inference_times"].append(inference_time)
    
    # Expand root with initial policy
    expansion_start = time.time()
    expand_node(root, policy, add_noise=True)
    expansion_time = time.time() - expansion_start
    detailed_timing["expansion"] += expansion_time
    
    if collect_stats and "expansion_times" in stats:
        stats["expansion_times"].append(expansion_time)
    
    # Initialize root
    root.value = value
    root.visits = 1
    
    # Run simulations
    remaining_sims = num_simulations - 1  # -1 for root evaluation
    
    # Define select function for batch collector
    def select_func(node):
        from mcts.core import select_node
        return select_node(node, exploration_weight)
    
    # Main simulation loop
    while remaining_sims > 0:
        # Collect batch of leaves
        batch_start = time.time()
        leaves, paths, terminal_leaves, terminal_paths = batch_collector.collect_batch(
            root, select_func, stats=stats if collect_stats else None
        )
        selection_time = time.time() - batch_start
        detailed_timing["selection"] += selection_time
        
        if collect_stats:
            stats["selection_times"].append(selection_time)
            stats["batch_sizes"].append(len(leaves) + len(terminal_leaves))
        
        # Process terminal states
        backprop_start = time.time()
        for leaf, path in zip(terminal_leaves, terminal_paths):
            value = leaf.state.get_winner()
            backpropagate(path, value)
            remaining_sims -= 1
        backprop_time = time.time() - backprop_start
        detailed_timing["backpropagation"] += backprop_time
        
        if collect_stats and "backprop_times" in stats:
            stats["backprop_times"].append(backprop_time)
        
        # Process non-terminal leaves with batch inference
        if leaves:
            # Get leaf states
            states = [leaf.state for leaf in leaves]
            
            # Perform batch inference
            inference_start = time.time()
            try:
                results = inference_actor.batch_infer(states)
                inference_time = time.time() - inference_start
                detailed_timing["inference"] += inference_time
                
                if collect_stats and "inference_times" in stats:
                    stats["inference_times"].append(inference_time)
                
                # Process results
                for leaf, path, result in zip(leaves, paths, results):
                    policy, value = result
                    
                    # Expansion phase
                    expansion_start = time.time()
                    expand_node(leaf, policy)
                    expansion_time = time.time() - expansion_start
                    detailed_timing["expansion"] += expansion_time
                    
                    if collect_stats and "expansion_times" in stats:
                        stats["expansion_times"].append(expansion_time)
                    
                    # Backpropagation phase
                    backprop_start = time.time()
                    backpropagate(path, value)
                    backprop_time = time.time() - backprop_start
                    detailed_timing["backpropagation"] += backprop_time
                    
                    if collect_stats and "backprop_times" in stats:
                        stats["backprop_times"].append(backprop_time)
                    
                    remaining_sims -= 1
            
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                # Process leaves individually as fallback
                for leaf, path in zip(leaves, paths):
                    try:
                        policy, value = inference_actor.infer(leaf.state)
                        
                        # Expansion
                        expansion_start = time.time()
                        expand_node(leaf, policy)
                        expansion_time = time.time() - expansion_start
                        detailed_timing["expansion"] += expansion_time
                        
                        # Backpropagation
                        backprop_start = time.time()
                        backpropagate(path, value)
                        backprop_time = time.time() - backprop_start
                        detailed_timing["backpropagation"] += backprop_time
                        
                        remaining_sims -= 1
                    except Exception as inner_e:
                        logger.error(f"Individual inference failed: {inner_e}")
    
    # Calculate total search time
    detailed_timing["total"] = time.time() - search_start
    
    # Add batch collection stats if collecting stats
    if collect_stats and stats is not None:
        stats.update({
            "batch_collector": batch_collector.get_stats(),
            "timing": detailed_timing,
            "total_nodes": get_node_count(root),
            "depth": get_tree_depth(root),
            "total_time": detailed_timing["total"]
        })
    
    # Print summary if verbose
    if verbose:
        print("\nMCTS Search Summary:")
        print(f"  Total time: {detailed_timing['total']:.3f}s")
        print(f"  Simulations: {num_simulations}")
        print(f"  Speed: {num_simulations / detailed_timing['total']:.1f} sims/second")
        
        # Time breakdown
        print("\nTime Breakdown:")
        for phase, time_spent in detailed_timing.items():
            if phase != "total":
                percentage = time_spent / detailed_timing["total"] * 100
                print(f"  {phase}: {time_spent:.3f}s ({percentage:.1f}%)")
        
        # Batch statistics
        if collect_stats and "batch_collector" in stats:
            batch_stats = stats["batch_collector"]
            print("\nBatch Statistics:")
            print(f"  Average batch size: {batch_stats.get('avg_batch_size', 0):.1f}")
            print(f"  Average collection time: {batch_stats.get('avg_collection_time', 0)*1000:.2f}ms")
            print(f"  Average utilization: {batch_stats.get('avg_inference_utilization', 0)*100:.1f}%")
            
            if adaptive_batching:
                print(f"  Final adaptive batch size: {batch_stats.get('current_batch_size', batch_size)}")
    
    # Return root node and stats if requested
    if collect_stats:
        return root, stats
    else:
        return root

def get_node_count(root):
    """Count total nodes in the search tree"""
    if root is None:
        return 0
        
    count = 1  # Count the root
    for child in root.children:
        count += get_node_count(child)
    
    return count

def get_tree_depth(root):
    """Get the maximum depth of the search tree"""
    if root is None or not root.children:
        return 0
        
    return 1 + max(get_tree_depth(child) for child in root.children)

def analyze_search_performance(stats):
    """
    Analyze search performance from collected statistics.
    
    Args:
        stats: Dictionary of search statistics
        
    Returns:
        dict: Performance analysis results
    """
    if not stats:
        return {}
    
    # Basic metrics
    analysis = {
        "total_time": stats.get("total_time", 0),
        "total_nodes": stats.get("total_nodes", 0),
        "tree_depth": stats.get("depth", 0)
    }
    
    # Time breakdown
    if "timing" in stats:
        timing = stats["timing"]
        total_time = timing.get("total", 1)
        
        analysis["time_breakdown"] = {
            phase: {
                "time": time_spent,
                "percentage": time_spent / total_time * 100
            }
            for phase, time_spent in timing.items()
            if phase != "total"
        }
        
        # Calculate bottleneck
        bottleneck_phase = max(
            [p for p in timing.keys() if p != "total"],
            key=lambda p: timing[p]
        )
        
        analysis["bottleneck"] = {
            "phase": bottleneck_phase,
            "time": timing[bottleneck_phase],
            "percentage": timing[bottleneck_phase] / total_time * 100
        }
    
    # Batch efficiency
    if "batch_collector" in stats:
        batch_stats = stats["batch_collector"]
        analysis["batch_efficiency"] = {
            "avg_batch_size": batch_stats.get("avg_batch_size", 0),
            "avg_utilization": batch_stats.get("avg_inference_utilization", 0),
            "recommended_batch_size": batch_stats.get("current_batch_size", 64)
        }
    
    # Calculate performance metrics
    if "timing" in stats and "total_nodes" in stats:
        analysis["performance"] = {
            "nodes_per_second": stats["total_nodes"] / max(0.001, stats["total_time"]),
            "inference_ratio": stats["timing"].get("inference", 0) / max(0.001, stats["total_time"])
        }
    
    # Generate recommendations
    analysis["recommendations"] = []
    
    if "bottleneck" in analysis:
        if analysis["bottleneck"]["phase"] == "inference":
            if analysis.get("batch_efficiency", {}).get("avg_utilization", 1) < 0.7:
                analysis["recommendations"].append(
                    "Consider decreasing batch size to improve inference efficiency"
                )
            else:
                analysis["recommendations"].append(
                    "Inference is the bottleneck - consider optimization of neural network or hardware"
                )
        
        elif analysis["bottleneck"]["phase"] == "selection":
            analysis["recommendations"].append(
                "Selection is the bottleneck - consider optimizing tree traversal algorithm"
            )
    
    if "batch_efficiency" in analysis:
        if analysis["batch_efficiency"]["avg_utilization"] < 0.5:
            analysis["recommendations"].append(
                f"Low batch utilization ({analysis['batch_efficiency']['avg_utilization']*100:.1f}%) - "
                f"recommended batch size: {analysis['batch_efficiency']['recommended_batch_size']}"
            )
    
    return analysis

def plot_search_performance(stats, filename=None):
    """
    Plot search performance from collected statistics.
    
    Args:
        stats: Dictionary of search statistics
        filename: Optional file to save the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        import matplotlib.pyplot as plt
        
        if not stats or not all(k in stats for k in ["timing", "batch_sizes", "inference_times"]):
            logger.warning("Insufficient statistics for plotting")
            return None
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("MCTS Search Performance Analysis", fontsize=16)
        
        # Time breakdown pie chart
        if "timing" in stats:
            timing = stats["timing"]
            phases = [p for p in timing.keys() if p != "total"]
            times = [timing[p] for p in phases]
            
            axs[0, 0].pie(times, labels=phases, autopct='%1.1f%%')
            axs[0, 0].set_title("Time Breakdown")
        
        # Batch size distribution
        if "batch_sizes" in stats:
            axs[0, 1].hist(stats["batch_sizes"], bins=min(20, len(set(stats["batch_sizes"]))),
                          alpha=0.7, color='blue')
            axs[0, 1].set_title("Batch Size Distribution")
            axs[0, 1].set_xlabel("Batch Size")
            axs[0, 1].set_ylabel("Frequency")
        
        # Inference time distribution
        if "inference_times" in stats:
            axs[1, 0].hist(np.array(stats["inference_times"]) * 1000, bins=20,
                          alpha=0.7, color='green')
            axs[1, 0].set_title("Inference Time Distribution")
            axs[1, 0].set_xlabel("Inference Time (ms)")
            axs[1, 0].set_ylabel("Frequency")
        
        # Batch utilization over time
        if "batch_collector" in stats and "batch_sizes" in stats:
            batch_sizes = stats["batch_sizes"]
            target_size = stats["batch_collector"].get("current_batch_size", 64)
            
            # Calculate utilization over time
            utilization = [min(1.0, b / target_size) for b in batch_sizes]
            
            # Plot utilization
            axs[1, 1].plot(utilization, color='red')
            axs[1, 1].set_title("Batch Utilization Over Time")
            axs[1, 1].set_xlabel("Batch Number")
            axs[1, 1].set_ylabel("Utilization Ratio")
            axs[1, 1].set_ylim(0, 1.1)
            axs[1, 1].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axs[1, 1].text(len(utilization) * 0.1, 0.72, "Target Minimum (70%)", fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {filename}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error generating performance plot: {e}")
        return None

# Example usage:
def test_optimized_search():
    """Test the optimized search implementation"""
    from utils.state_utils import TicTacToeState
    from train.patch_self_play import MockBatchInferenceServer
    
    # Create mock inference server
    inference_server = MockBatchInferenceServer(
        batch_wait=0.001,
        cache_size=1000,
        max_batch_size=64
    )
    
    # Create initial state
    state = TicTacToeState()
    
    # Run optimized search with stats collection
    root, stats = optimized_batched_mcts_search(
        state=state,
        inference_actor=inference_server,
        num_simulations=200,
        batch_size=32,
        batch_strategy='balanced',
        adaptive_batching=True,
        verbose=True,
        collect_stats=True
    )
    
    # Analyze performance
    analysis = analyze_search_performance(stats)
    
    print("\nPerformance Analysis:")
    for key, value in analysis.items():
        if key != "recommendations":
            print(f"  {key}: {value}")
    
    print("\nRecommendations:")
    for recommendation in analysis.get("recommendations", []):
        print(f"  - {recommendation}")
    
    # Plot performance
    fig = plot_search_performance(stats, "search_performance.png")
    
    return root, stats, analysis

if __name__ == "__main__":
    test_optimized_search()