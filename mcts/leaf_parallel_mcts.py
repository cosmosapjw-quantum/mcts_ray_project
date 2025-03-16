# mcts/leaf_parallel_mcts.py - Fixed implementation
import time
import threading
import queue
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Set, Callable, Union
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LeafParallelMCTS")

class LeafParallelMCTS:
    """
    Leaf parallelization MCTS implementation with operation pipelining.
    
    This implementation uses multiple worker threads to maximize throughput:
    - Multiple leaf collectors select nodes from the tree
    - Evaluator batches leaves for neural network inference
    - Result processor updates the tree with evaluation results
    
    The key advantage is overlapping operations:
    - While GPU is evaluating one batch, CPU is selecting the next batch
    - While CPU is backpropagating, GPU is evaluating the next batch
    """
    
    def __init__(self, 
                inference_fn: Callable,
                num_collectors: int = 2,
                batch_size: int = 32,
                exploration_weight: float = 1.4,
                collect_stats: bool = True,
                collector_timeout: float = 0.01,  # Increased from 0.002 to 10ms
                min_batch_size: int = 8,          # New parameter for minimum batch size
                evaluator_wait_time: float = 0.02): # New parameter for wait time
        """
        Initialize the MCTS search with leaf parallelization.
        
        Args:
            inference_fn: Function for neural network inference
            num_collectors: Number of leaf collector threads
            batch_size: Maximum batch size for inference
            exploration_weight: Exploration constant for PUCT
            collect_stats: Whether to collect performance statistics
            collector_timeout: Maximum time for a collector to collect nodes (in seconds)
            min_batch_size: Minimum batch size before evaluation (new parameter)
            evaluator_wait_time: Maximum time to wait for batch formation (new parameter)
        """
        self.inference_fn = inference_fn
        self.num_collectors = num_collectors
        self.batch_size = batch_size
        self.exploration_weight = exploration_weight
        self.collect_stats = collect_stats
        self.collector_timeout = collector_timeout
        self.min_batch_size = min_batch_size  # Store new parameter
        self.evaluator_wait_time = evaluator_wait_time  # Store new parameter
        
        # Create thread-safe resources
        self.tree_lock = threading.RLock()
        self.eval_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Worker threads with proper coordination
        self.collectors = []
        self.evaluator = None
        self.processor = None
        
        # Add timeout and health monitoring
        self.start_time = None
        self.shutdown_flag = threading.Event()
        
        # Add stats for diagnostics
        self.pending_nodes = set()  # Track nodes currently being processed
        self.expanded_nodes = set()  # Track expanded nodes
        
        logger.info(f"LeafParallelMCTS initialized with {num_collectors} collectors, " +
                   f"batch_size={batch_size}, exploration_weight={exploration_weight}")
    
    def search(self, root_state, num_simulations: int = 800, add_dirichlet_noise: bool = True) -> Tuple[Any, Dict]:
        """
        Perform MCTS search from root state.
        
        Args:
            root_state: Initial game state
            num_simulations: Number of simulations to run
            add_dirichlet_noise: Whether to add Dirichlet noise at root
            
        Returns:
            Tuple: (root node, statistics dictionary)
        """
        from mcts.node import Node
        from mcts.core import expand_node, backpropagate

        # Start timing
        self.start_time = time.time()
        self.total_simulations = num_simulations
        
        # Reset diagnostic counters
        self.pending_nodes.clear()
        self.expanded_nodes.clear()
        
        # Create root node
        root = Node(root_state)
        
        # Initialize with first evaluation - WITH ERROR HANDLING
        try:
            # Handle Ray actor references or regular function returns
            inference_result = self.inference_fn(root_state)
            
            # If this is a Ray ObjectRef, it should have been resolved already by the inference_fn
            if hasattr(inference_result, "_ray_object_ref"):
                # This shouldn't happen if inference_fn is implemented correctly, but just in case
                try:
                    import ray
                    logger.warning("Got Ray reference instead of actual result - attempting to resolve")
                    inference_result = ray.get(inference_result, timeout=5.0)
                except Exception as e:
                    logger.error(f"Failed to resolve Ray reference: {e}")
                    # Use default policy and value as fallback
                    policy = np.ones(9) / 9  # Uniform policy for TicTacToe
                    value = 0.0
            else:
                # Normal case - unpack the result
                policy, value = inference_result
        except Exception as e:
            logger.error(f"Error during root evaluation: {e}", exc_info=True)
            # Use default policy and value
            policy = np.ones(9) / 9  # Uniform policy for TicTacToe
            value = 0.0
        
        # Expand root node with initial policy
        with self.tree_lock:
            expand_node(root, policy, add_noise=add_dirichlet_noise)
            root.value = value
            root.visits = 1
            root.is_expanded = True
            
            # Add to expanded nodes set for tracking
            self.expanded_nodes.add(id(root))
        
        # Initialize workers
        self._start_workers(root)
        
        # Wait for simulations to complete
        sims_completed = 1  # Count root evaluation
        last_update_time = time.time()
        update_interval = 1.0  # 1 second
        last_sims_count = 1
        stall_start_time = None
        
        try:
            while sims_completed < num_simulations:
                # Check simulation progress
                # Each processed result corresponds to one completed simulation
                sims_completed = 1 + self.processor.results_processed if self.processor else 1
                
                # Detect stalls (no progress for 5 seconds)
                if sims_completed == last_sims_count:
                    if stall_start_time is None:
                        stall_start_time = time.time()
                    elif time.time() - stall_start_time > 5.0:
                        # Log detailed diagnostic information
                        self._log_diagnostic_info(root)
                        
                        # Reset stall detection
                        stall_start_time = time.time()
                else:
                    # Progress was made, reset stall detection
                    stall_start_time = None
                    last_sims_count = sims_completed
                
                # Periodically log progress
                current_time = time.time()
                if self.collect_stats and current_time - last_update_time > update_interval:
                    elapsed = current_time - self.start_time
                    sims_per_second = sims_completed / elapsed
                    logger.debug(f"MCTS progress: {sims_completed}/{num_simulations} simulations " +
                               f"({sims_per_second:.1f} sims/s)")
                    last_update_time = current_time
                
                # Check for timeout (60 seconds)
                if current_time - self.start_time > 60.0:
                    logger.warning(f"Search timeout reached with {sims_completed}/{num_simulations} simulations")
                    break
                    
                # Short sleep to avoid tight loop
                time.sleep(0.01)  # 10ms
                
        finally:
            # Shutdown workers
            self._shutdown_workers()
        
        # Record statistics
        self.end_time = time.time()
        self.total_nodes = self._count_nodes(root)
        
        # Log search summary
        if self.collect_stats:
            self._log_search_summary(root)
        
        # Get stats
        stats = self.get_search_stats()
        
        return root, stats
    
    def _start_workers(self, root):
        """Start worker threads for search"""
        
        # Create and start leaf collectors
        self.collectors = []
        for i in range(self.num_collectors):
            collector = LeafCollector(
                root=root,
                eval_queue=self.eval_queue,
                result_queue=self.result_queue,
                lock=self.tree_lock,
                batch_size=max(1, self.batch_size // self.num_collectors),
                max_queue_size=self.batch_size * 2,
                exploration_weight=self.exploration_weight,
                max_collection_time=self.collector_timeout,
                expanded_nodes=self.expanded_nodes,
                pending_nodes=self.pending_nodes
            )
            thread = threading.Thread(
                target=collector.run,
                daemon=True,
                name=f"leaf_collector_{i}"
            )
            thread.start()
            self.collectors.append((collector, thread))
        
        # Create and start evaluator with improved batching parameters
        self.evaluator = Evaluator(
            root=root,
            inference_fn=self.inference_fn,
            eval_queue=self.eval_queue,
            result_queue=self.result_queue,
            batch_size=self.batch_size,
            max_wait_time=0.020,  # Increased from 0.005 to 20ms
            min_batch_size=8,     # Process batches of at least 8 nodes
            expanded_nodes=self.expanded_nodes,
            pending_nodes=self.pending_nodes
        )
        self.evaluator_thread = threading.Thread(
            target=self.evaluator.run,
            daemon=True,
            name="evaluator"
        )
        self.evaluator_thread.start()
        
        # Create and start result processor
        self.processor = ResultProcessor(
            result_queue=self.result_queue,
            lock=self.tree_lock,
            expanded_nodes=self.expanded_nodes,  # Pass expanded nodes set
            pending_nodes=self.pending_nodes  # Pass pending nodes set
        )
        self.processor_thread = threading.Thread(
            target=self.processor.run,
            daemon=True,
            name="result_processor"
        )
        self.processor_thread.start()
    
    def _shutdown_workers(self):
        """Shutdown worker threads"""
        # Signal shutdown
        for collector, _ in self.collectors:
            collector.shutdown()
        if self.evaluator:
            self.evaluator.shutdown()
        if self.processor:
            self.processor.shutdown()
        
        # Wait for threads to exit (with timeout)
        for _, thread in self.collectors:
            thread.join(timeout=0.5)  # Increased timeout
        if self.evaluator_thread:
            self.evaluator_thread.join(timeout=0.5)  # Increased timeout
        if self.processor_thread:
            self.processor_thread.join(timeout=0.5)  # Increased timeout
    
    def _count_nodes(self, node):
        """Count total nodes in the search tree"""
        if not node:
            return 0
            
        count = 1  # Count this node
        for child in node.children:
            count += self._count_nodes(child)
            
        return count
    
    def _log_diagnostic_info(self, root):
        """Log detailed diagnostic information when stalls are detected"""
        logger.warning("=== MCTS STALL DETECTED ===")
        logger.warning(f"Root node has {len(root.children)} children and {root.visits} visits")
        logger.warning(f"Pending nodes: {len(self.pending_nodes)}")
        logger.warning(f"Expanded nodes: {len(self.expanded_nodes)}")
        logger.warning(f"Eval queue size: {self.eval_queue.qsize()}")
        logger.warning(f"Result queue size: {self.result_queue.qsize()}")
        
        # Log collector stats
        for i, (collector, _) in enumerate(self.collectors):
            stats = collector.get_stats()
            logger.warning(f"Collector {i}: collected={stats['leaves_collected']}, "
                          f"time={stats['avg_collection_time']*1000:.2f}ms")
        
        # Log evaluator stats
        if self.evaluator:
            stats = self.evaluator.get_stats()
            logger.warning(f"Evaluator: batches={stats['batches_evaluated']}, "
                          f"size={stats['avg_batch_size']:.1f}, "
                          f"time={stats['avg_evaluation_time']*1000:.2f}ms")
        
        # Log processor stats
        if self.processor:
            stats = self.processor.get_stats()
            logger.warning(f"Processor: processed={stats['results_processed']}, "
                          f"time={stats['avg_processing_time']*1000:.2f}ms")
    
    def _log_search_summary(self, root):
        """Log search performance summary"""
        # Calculate timings
        elapsed = self.end_time - self.start_time
        sims_per_second = self.total_simulations / elapsed if elapsed > 0 else 0
        
        # Get worker statistics
        collector_stats = [collector.get_stats() for collector, _ in self.collectors]
        evaluator_stats = self.evaluator.get_stats() if self.evaluator else {}
        processor_stats = self.processor.get_stats() if self.processor else {}
        
        # Calculate aggregate collector statistics
        total_leaves_collected = sum(stats.get("leaves_collected", 0) for stats in collector_stats)
        avg_collection_times = [stats.get("avg_collection_time", 0) for stats in collector_stats 
                              if stats.get("avg_collection_time", 0) > 0]
        avg_collection_time = np.mean(avg_collection_times) if avg_collection_times else 0
        
        # Log summary
        logger.info("\nMCTS Search Summary:")
        logger.info(f"  Total time: {elapsed:.3f}s")
        logger.info(f"  Simulations: {self.total_simulations}")
        logger.info(f"  Speed: {sims_per_second:.1f} sims/second")
        logger.info(f"  Total nodes: {self.total_nodes}")
        
        # Log evaluator statistics
        if evaluator_stats:
            logger.info("\nEvaluator Statistics:")
            logger.info(f"  Batches: {evaluator_stats.get('batches_evaluated', 0)}")
            logger.info(f"  Avg batch size: {evaluator_stats.get('avg_batch_size', 0):.1f}")
            logger.info(f"  Avg evaluation time: {evaluator_stats.get('avg_evaluation_time', 0)*1000:.2f}ms")
        
        # Log collector statistics
        logger.info("\nCollector Statistics:")
        logger.info(f"  Leaves collected: {total_leaves_collected}")
        logger.info(f"  Avg collection time: {avg_collection_time*1000:.2f}ms")
        
        # Log processor statistics
        if processor_stats:
            logger.info("\nProcessor Statistics:")
            logger.info(f"  Results processed: {processor_stats.get('results_processed', 0)}")
            logger.info(f"  Avg processing time: {processor_stats.get('avg_processing_time', 0)*1000:.2f}ms")
    
    def get_search_stats(self):
        """Get comprehensive search statistics"""
        if not self.collect_stats:
            return {}
            
        stats = {
            "search_time": (self.end_time - self.start_time) if hasattr(self, 'end_time') and self.end_time else 0,
            "total_simulations": getattr(self, 'total_simulations', 0),
            "total_nodes": getattr(self, 'total_nodes', 0),
            "sims_per_second": (getattr(self, 'total_simulations', 0) / 
                               (self.end_time - self.start_time)) 
                              if hasattr(self, 'end_time') and self.end_time and self.start_time else 0
        }
        
        # Add worker statistics
        if hasattr(self, 'collectors') and self.collectors:
            collector_stats = [collector.get_stats() for collector, _ in self.collectors]
            leaves_collected = [s.get("leaves_collected", 0) for s in collector_stats]
            collection_times = [s.get("avg_collection_time", 0) for s in collector_stats 
                              if s.get("avg_collection_time", 0) > 0]
            
            stats["collector"] = {
                "total_leaves_collected": sum(leaves_collected),
                "avg_collection_time": np.mean(collection_times) if collection_times else 0,
                "num_collectors": len(self.collectors)
            }
        
        if hasattr(self, 'evaluator') and self.evaluator:
            stats["evaluator"] = self.evaluator.get_stats()
            
        if hasattr(self, 'processor') and self.processor:
            stats["processor"] = self.processor.get_stats()
            
        return stats

# Leaf Collector Implementation
class LeafCollector:
    """
    Worker thread that selects leaf nodes for evaluation.
    
    This collector runs in a separate thread, continuously selecting
    leaves from the shared search tree and adding them to the evaluation queue.
    """
    
    def __init__(self, 
                 root, 
                 eval_queue,
                 result_queue,
                 lock,
                 batch_size=8,
                 max_queue_size=32,
                 exploration_weight=1.4,
                 max_collection_time=0.01,  # Increased from 0.002 to 10ms
                 select_func=None,
                 expanded_nodes=None,
                 pending_nodes=None):
        """Initialize the leaf collector"""
        self.root = root
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.lock = lock
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.exploration_weight = exploration_weight
        self.max_collection_time = max_collection_time  # New parameter
        
        # Sets for tracking node status
        self.expanded_nodes = expanded_nodes if expanded_nodes is not None else set()
        self.pending_nodes = pending_nodes if pending_nodes is not None else set()
        
        # Use provided selection function or default
        self.select_func = select_func or _default_select
        
        # Statistics
        self.leaves_collected = 0
        self.collection_times = deque(maxlen=100)
        self.wait_times = deque(maxlen=100)
        self.consecutive_empty_batches = 0  # Track how many empty batches in a row
        
        # Control flags
        self.shutdown_flag = False
    
    def run(self):
        """Main worker loop for leaf collection"""
        while not self.shutdown_flag:
            try:
                # Wait if evaluation queue is too full
                if self.eval_queue.qsize() >= self.max_queue_size:
                    wait_start = time.time()
                    time.sleep(0.001)  # 1ms
                    self.wait_times.append(time.time() - wait_start)
                    continue
                
                # Collect batch of leaves
                collection_start = time.time()
                batch = self._collect_batch()
                collection_time = time.time() - collection_start
                
                if batch:
                    self.collection_times.append(collection_time)
                    self.leaves_collected += len(batch)
                    self.consecutive_empty_batches = 0  # Reset counter
                    
                    # Add to evaluation queue
                    for leaf, path in batch:
                        self.eval_queue.put((leaf, path))
                else:
                    # No leaves collected, log if this keeps happening
                    self.consecutive_empty_batches += 1
                    if self.consecutive_empty_batches >= 10:
                        # Log diagnostic info every 10 consecutive empty batches
                        logger.debug(f"Collector has produced {self.consecutive_empty_batches} empty batches in a row")
                        self.consecutive_empty_batches = 0  # Reset counter
                    
                    # Short sleep to avoid CPU spinning
                    time.sleep(0.001)  # 1ms
            
            except Exception as e:
                logger.error(f"Error in leaf collector: {e}")
                import traceback
                traceback.print_exc()
                
                # Short sleep to avoid error flooding
                time.sleep(0.01)  # 10ms
    
    def _collect_batch(self):
        """Collect a batch of unexpanded leaf nodes"""
        batch = []
        
        # Calculate target batch size
        target_size = min(self.batch_size, self.max_queue_size - self.eval_queue.qsize())
        if target_size <= 0:
            return []
        
        # Collect leaf nodes with tree lock
        start_time = time.time()
        
        # Fixed max_retries to avoid infinite loops
        max_retries = 30  # Set a limit for node selection attempts
        retries = 0
            
        while len(batch) < target_size and (time.time() - start_time) < self.max_collection_time and retries < max_retries:
            try:
                with self.lock:
                    # Select a leaf node
                    leaf, path = self.select_func(self.root, self.exploration_weight)
                    
                    leaf_id = id(leaf)
                    
                    # Skip terminal nodes (process them immediately)
                    if leaf.state.is_terminal():
                        try:
                            # Get winner directly
                            value = leaf.state.get_winner()
                            # Put the result directly in the result queue
                            self.result_queue.put((leaf, path, (value, None)))
                        except Exception as e:
                            logger.error(f"Error processing terminal node: {e}")
                        retries += 1
                        continue
                    
                    # Skip already-expanded nodes
                    if leaf.is_expanded or leaf_id in self.expanded_nodes:
                        retries += 1
                        continue
                    
                    # Skip nodes that are already being processed
                    if leaf_id in self.pending_nodes:
                        retries += 1
                        continue
                    
                    # Mark node as pending to avoid duplicates
                    self.pending_nodes.add(leaf_id)
                    batch.append((leaf, path))
            except Exception as e:
                logger.error(f"Error selecting node: {e}")
                retries += 1
        
        if retries >= max_retries and len(batch) == 0:
            logger.debug(f"Reached retry limit ({max_retries}) without finding valid nodes")
        
        return batch
    
    def shutdown(self):
        """Signal the worker to shut down"""
        self.shutdown_flag = True
    
    def get_stats(self):
        """Get statistics about leaf collection"""
        return {
            "leaves_collected": self.leaves_collected,
            "avg_collection_time": np.mean(self.collection_times) if self.collection_times else 0,
            "avg_wait_time": np.mean(self.wait_times) if self.wait_times else 0,
            "current_eval_queue_size": self.eval_queue.qsize(),
            "consecutive_empty_batches": self.consecutive_empty_batches
        }

# Evaluator Implementation
class Evaluator:
    """
    Worker thread that evaluates batches of leaves using the neural network.
    """
    def __init__(self, 
                root,
                inference_fn,
                eval_queue,
                result_queue,
                batch_size=8,
                max_wait_time=0.005,  # Increased from 0.001 to 5ms
                min_batch_size=1,     # Process batches of at least 1 node
                exploration_weight=1.4,
                expanded_nodes=None,
                pending_nodes=None):
        """Initialize the evaluator"""
        self.root = root
        self.inference_fn = inference_fn
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size  # New parameter
        self.max_queue_size = batch_size * 2
        
        # Sets for tracking node status
        self.expanded_nodes = expanded_nodes if expanded_nodes is not None else set()
        self.pending_nodes = pending_nodes if pending_nodes is not None else set()
        
        # Statistics
        self.batches_evaluated = 0
        self.leaves_evaluated = 0
        self.batch_sizes = deque(maxlen=100)
        self.evaluation_times = deque(maxlen=100)
        self.wait_times = deque(maxlen=100)
        
        # Control flags
        self.shutdown_flag = False
        self.exploration_weight = exploration_weight
    
    def run(self):
        """Main worker loop for batch evaluation"""
        while not self.shutdown_flag:
            try:
                # Collect batch from eval queue
                wait_start = time.time()
                batch = self._collect_batch_from_queue()
                wait_time = time.time() - wait_start
                
                if not batch:
                    # No items to process, short sleep
                    time.sleep(0.001)  # 1ms
                    continue
                
                self.wait_times.append(wait_time)
                
                # Extract states for inference
                leaves, paths = zip(*batch)
                states = [leaf.state for leaf in leaves]
                
                # Evaluate batch
                evaluation_start = time.time()
                results = self._evaluate_batch(states)
                evaluation_time = time.time() - evaluation_start
                
                self.batches_evaluated += 1
                self.leaves_evaluated += len(batch)
                self.batch_sizes.append(len(batch))
                self.evaluation_times.append(evaluation_time)
                
                # Send results to result queue
                for i, (leaf, path) in enumerate(zip(leaves, paths)):
                    if i < len(results):
                        result = results[i]
                    else:
                        # Fallback in case of mismatch
                        result = (np.ones(9)/9, 0.0)
                    
                    self.result_queue.put((leaf, path, result))
                
            except Exception as e:
                logger.error(f"Error in evaluator: {e}")
                import traceback
                traceback.print_exc()
                
                # Short sleep to avoid error flooding
                time.sleep(0.01)  # 10ms
    
    def _collect_batch_from_queue(self):
        """Collect a batch from the eval queue with improved batching strategy"""
        batch = []
        
        # Get current queue size for adaptive behavior
        queue_size = self.eval_queue.qsize()
        
        # Determine adaptive wait time based on queue conditions
        target_wait_time = self.max_wait_time
        if queue_size > self.batch_size / 2:
            # Queue is filling up, wait longer for better batching
            target_wait_time = min(0.05, self.max_wait_time * 2)  # Up to 50ms
        
        # Try to get first item with longer timeout
        try:
            # Wait for first item with sufficient timeout
            first_item_timeout = min(0.02, target_wait_time)  # At least 20ms
            item = self.eval_queue.get(timeout=first_item_timeout)
            batch.append(item)
        except queue.Empty:
            return []
        
        # Start timing batch collection
        collection_start = time.time()
        
        # Determine target batch size based on queue dynamics
        target_batch_size = max(
            self.min_batch_size,  
            min(self.batch_size, int(queue_size * 1.2) + 1)
        )
        
        # Try to collect up to target_batch_size
        while len(batch) < target_batch_size:
            # Check if we've waited long enough
            elapsed = time.time() - collection_start
            
            if elapsed > target_wait_time:
                # If we have enough items, proceed
                if len(batch) >= self.min_batch_size:
                    break
                # If we've waited extremely long but don't have minimum batch,
                # proceed anyway if we have something
                elif elapsed > target_wait_time * 2 and len(batch) > 0:
                    break
                
            try:
                # Use small timeout instead of nowait for smoother collection
                item = self.eval_queue.get(timeout=0.002)  # 2ms timeout
                batch.append(item)
            except queue.Empty:
                # If we already have minimum batch size, don't wait too long
                if len(batch) >= self.min_batch_size:
                    break
                
                # Short sleep to avoid CPU spinning
                time.sleep(0.002)  # 2ms
        
        # Log batch size periodically for debugging
        if self.batch_sizes and (len(self.batch_sizes) % 20 == 0):
            avg_size = sum(self.batch_sizes) / len(self.batch_sizes)
            logger.debug(f"Average batch size: {avg_size:.2f}, current: {len(batch)}")
        
        return batch
    
    def _evaluate_batch(self, states):
        """Evaluate a batch of states with the neural network"""
        try:
            # Handle empty batch (shouldn't happen)
            if not states:
                return []
                
            # Special case for batch size 1
            if len(states) == 1:
                # Single state evaluation
                try:
                    return [self.inference_fn(states[0])]
                except Exception as e:
                    logger.error(f"Error evaluating single state: {e}")
                    return [(np.ones(9)/9, 0.0)]  # Return default
            
            # Batch inference
            try:
                batch_results = self.inference_fn(states)
                
                # Validate result format (should be a list of tuples)
                if not isinstance(batch_results, list):
                    logger.error(f"Expected list result but got {type(batch_results)}")
                    return [(np.ones(9)/9, 0.0) for _ in states]
                
                if len(batch_results) != len(states):
                    logger.error(f"Result length mismatch: got {len(batch_results)}, expected {len(states)}")
                    # Extend or truncate results to match states
                    if len(batch_results) < len(states):
                        # Extend with defaults
                        return batch_results + [(np.ones(9)/9, 0.0) for _ in range(len(states) - len(batch_results))]
                    else:
                        # Truncate
                        return batch_results[:len(states)]
                
                return batch_results
            except Exception as e:
                logger.error(f"Error during batch evaluation: {e}")
                return [(np.ones(9)/9, 0.0) for _ in states]
            
        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}")
            
            # Return default values on error
            default_policy = np.ones(9) / 9  # Uniform policy
            default_value = 0.0  # Neutral value
            
            return [(default_policy, default_value) for _ in states]
    
    def shutdown(self):
        """Signal the worker to shut down"""
        self.shutdown_flag = True
    
    def get_stats(self):
        """Get statistics about evaluation"""
        return {
            "batches_evaluated": self.batches_evaluated,
            "leaves_evaluated": self.leaves_evaluated,
            "avg_batch_size": np.mean(self.batch_sizes) if self.batch_sizes else 0,
            "avg_evaluation_time": np.mean(self.evaluation_times) if self.evaluation_times else 0,
            "avg_wait_time": np.mean(self.wait_times) if self.wait_times else 0,
            "current_queue_size": self.eval_queue.qsize()
        }

# Result Processor Implementation
class ResultProcessor:
    """
    Worker thread that processes evaluation results and updates the tree.
    """
    
    def __init__(self, 
                 result_queue,
                 lock,
                 expanded_nodes=None,
                 pending_nodes=None):
        """Initialize the result processor"""
        self.result_queue = result_queue
        self.lock = lock
        
        # Sets for tracking node status
        self.expanded_nodes = expanded_nodes if expanded_nodes is not None else set()
        self.pending_nodes = pending_nodes if pending_nodes is not None else set()
        
        # Statistics
        self.results_processed = 0
        self.processing_times = deque(maxlen=100)
        self.wait_times = deque(maxlen=100)
        self.errors = 0
        
        # Control flags
        self.shutdown_flag = False
    
    def run(self):
        """Main worker loop for result processing"""
        while not self.shutdown_flag:
            try:
                # Get result from queue with timeout
                try:
                    wait_start = time.time()
                    leaf, path, result = self.result_queue.get(timeout=0.01)  # 10ms timeout
                    self.wait_times.append(time.time() - wait_start)
                except queue.Empty:
                    continue
                
                # Process result
                processing_start = time.time()
                success = self._process_result(leaf, path, result)
                self.processing_times.append(time.time() - processing_start)
                if success:
                    self.results_processed += 1
                
            except Exception as e:
                logger.error(f"Error in result processor: {e}")
                import traceback
                traceback.print_exc()
                self.errors += 1
                
                # Short sleep to avoid error flooding
                time.sleep(0.01)  # 10ms
    
    def _process_result(self, leaf, path, result):
        """Process an evaluation result and update the tree"""
        from mcts.core import expand_node, backpropagate
        
        leaf_id = id(leaf)
        
        with self.lock:
            try:
                # Remove leaf from pending nodes set
                self.pending_nodes.discard(leaf_id)
                
                # Skip if node is already expanded (to avoid race conditions)
                if leaf.is_expanded or leaf_id in self.expanded_nodes:
                    return True
                
                # Check if this is a terminal node result
                if isinstance(result, tuple) and len(result) == 2 and result[1] is None:
                    # Just backpropagate the game result
                    value = result[0]
                    backpropagate(path, value)
                    return True
                
                # Handle normal evaluation result
                try:
                    # Standard format: (policy, value)
                    policy, value = result
                    
                    # Expand leaf node with the policy
                    expand_node(leaf, policy)
                    
                    # Mark as expanded
                    leaf.is_expanded = True
                    self.expanded_nodes.add(leaf_id)
                    
                    # Backpropagate the value
                    backpropagate(path, value)
                    return True
                    
                except ValueError:
                    # Try reversing the order (some implementations use value, policy)
                    try:
                        value, policy = result
                        expand_node(leaf, policy)
                        leaf.is_expanded = True
                        self.expanded_nodes.add(leaf_id)
                        backpropagate(path, value)
                        return True
                    except Exception as e2:
                        logger.error(f"Failed alternative result processing: {e2}")
                        raise
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                self.errors += 1
                return False
    
    def shutdown(self):
        """Signal the worker to shut down"""
        self.shutdown_flag = True
    
    def get_stats(self):
        """Get statistics about result processing"""
        return {
            "results_processed": self.results_processed,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "avg_wait_time": np.mean(self.wait_times) if self.wait_times else 0,
            "current_result_queue_size": self.result_queue.qsize(),
            "errors": self.errors
        }

def _default_select(node, exploration_weight):
    """Default PUCT-based leaf selection function"""
    from mcts.core import select_node
    return select_node(node, exploration_weight)

# Convenience function with improved error handling
def leaf_parallel_search(root_state, inference_fn, num_simulations=800, 
                        num_collectors=2, batch_size=32, exploration_weight=1.4,
                        add_dirichlet_noise=True, collect_stats=True,
                        collector_timeout=0.01, min_batch_size=8,
                        evaluator_wait_time=0.02):
    """
    Run leaf-parallel MCTS search from the given root state with robust error handling.
    
    Args:
        root_state: Initial game state
        inference_fn: Function for neural network inference
        num_simulations: Number of simulations to run
        num_collectors: Number of leaf collector threads
        batch_size: Maximum batch size for inference
        exploration_weight: Exploration constant for PUCT
        add_dirichlet_noise: Whether to add Dirichlet noise at root
        collect_stats: Whether to collect performance statistics
        collector_timeout: Maximum time for a collector to collect nodes (in seconds)
        min_batch_size: Minimum batch size before evaluation (new parameter)
        evaluator_wait_time: Maximum time to wait for batch formation (new parameter)
        
    Returns:
        tuple: (root_node, stats)
    """
    try:
        # Create MCTS instance with improved parameters
        mcts = LeafParallelMCTS(
            inference_fn=inference_fn,
            num_collectors=num_collectors,
            batch_size=batch_size,
            exploration_weight=exploration_weight,
            collect_stats=collect_stats,
            collector_timeout=collector_timeout,
            min_batch_size=min_batch_size,  # Pass new parameter
            evaluator_wait_time=evaluator_wait_time  # Pass new parameter
        )
        
        # Run search
        root, stats = mcts.search(
            root_state=root_state,
            num_simulations=num_simulations,
            add_dirichlet_noise=add_dirichlet_noise
        )
        
        # Return root and stats
        return root, stats
    except Exception as e:
        logger.error(f"Error during leaf-parallel search: {e}", exc_info=True)
        
        # Create a minimal root node as fallback
        from mcts.node import Node
        root = Node(root_state)
        
        # If state has legal actions, add children with uniform policy
        legal_actions = root_state.get_legal_actions()
        if legal_actions:
            from mcts.core import expand_node
            # Create uniform policy
            policy = np.zeros(9)
            for action in legal_actions:
                policy[action] = 1.0 / len(legal_actions)
                
            # Expand root with uniform policy
            expand_node(root, policy)
            root.visits = 1
        
        # Return root and minimal stats
        stats = {
            "error": str(e),
            "search_time": 0,
            "total_simulations": 0,
            "total_nodes": 1,
            "sims_per_second": 0
        }
        return root, stats