# mcts/leaf_parallel_mcts.py - Fixed implementation
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import queue
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Set, Callable, Union
from collections import deque

from mcts.core import select_node_with_node_locks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LeafParallelMCTS")

class LeafParallelMCTS:
    """Leaf parallelization MCTS with adaptive parameters and enhanced thread management"""
    
    def __init__(self, 
                inference_fn: Callable,
                num_collectors: int = 8,      # Increased default from 2 to 8
                batch_size: int = 64,
                exploration_weight: float = 1.4,
                collect_stats: bool = True,
                collector_timeout: float = 0.01,
                min_batch_size: int = 16,      # Increased from 8 to 16
                evaluator_wait_time: float = 0.02,
                verbose: bool = False,
                adaptive_parameters: bool = True):
        """Initialize with improved thread management and synchronization"""
        # Store all parameters
        self.inference_fn = inference_fn
        self.num_collectors = num_collectors
        self.batch_size = batch_size
        self.exploration_weight = exploration_weight
        self.collect_stats = collect_stats
        self.collector_timeout = collector_timeout
        self.min_batch_size = min_batch_size
        self.evaluator_wait_time = evaluator_wait_time
        self.verbose = verbose
        self.adaptive_parameters = adaptive_parameters
        
        # Create thread-safe resources
        self.tree_lock = threading.RLock()  # Global lock for shared data structures
        self.eval_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Worker threads
        self.collectors = []
        self.evaluator = None
        self.processor = None
        
        # Timeout and health monitoring
        self.start_time = None
        self.shutdown_flag = threading.Event()
        
        # Tracking sets
        self.pending_nodes = set()
        self.expanded_nodes = set()
        
        # Adaptive parameters
        self.adaptation_interval = 10.0  # Seconds between parameter adjustments
        self.last_adaptation_time = time.time()
        self.adaptation_stats = {
            'progress_rate': [],
            'batch_sizes': [],
            'collection_success_rate': []
        }
        
        # Tree health tracking
        self.health_check_history = []
        self.applied_fixes = []
        self.parameter_adjustments = []
        
        # Thread pool for collectors (alternative to individual threads)
        self.use_thread_pool = True  # Flag to control whether to use thread pool
        if self.use_thread_pool:
            from concurrent.futures import ThreadPoolExecutor
            self.collector_pool = ThreadPoolExecutor(
                max_workers=self.num_collectors,
                thread_name_prefix="leaf_collector_"
            )
        
        logger.info(f"LeafParallelMCTS initialized with {num_collectors} collectors, " +
                   f"batch_size={batch_size}, exploration_weight={exploration_weight}" +
                   (", adaptive parameters enabled" if adaptive_parameters else ""))
            
    def check_tree_health(self, root):
        """
        Perform a comprehensive tree health check to detect inconsistencies.
        
        Args:
            root: Root node of the search tree
            
        Returns:
            dict: Health check results with potential issues
        """
        health = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Check child-parent visit consistency
        child_visits = sum(child.visits for child in root.children) if root.children else 0
        if root.visits < child_visits:
            health["status"] = "inconsistent"
            health["issues"].append({
                "type": "visit_inconsistency",
                "message": f"Root has {root.visits} visits but children have {child_visits} visits combined",
                "severity": "critical"
            })
            
        # Check for extreme visit distribution skew
        if root.children:
            visits = [child.visits for child in root.children]
            max_visits = max(visits)
            min_visits = min(visits)
            avg_visits = sum(visits) / len(visits)
            health["metrics"]["visit_stats"] = {
                "max": max_visits,
                "min": min_visits,
                "avg": avg_visits,
                "std_dev": np.std(visits)
            }
            
            # If one child dominates, we might have poor exploration
            if max_visits > 10 * avg_visits:
                health["issues"].append({
                    "type": "skewed_distribution",
                    "message": f"Highly skewed visit distribution (max: {max_visits}, avg: {avg_visits:.1f})",
                    "severity": "warning"
                })
        
        # Better detection of pending nodes buildup
        if len(self.pending_nodes) > 20 and self.processor and self.processor.results_processed < 10:
            health["issues"].append({
                "type": "pending_nodes_buildup",
                "message": f"{len(self.pending_nodes)} pending nodes but few results processed",
                "severity": "warning"
            })
        
        # Check expansion rate
        expanded_count = self._count_expanded_nodes(root)
        total_nodes = self._count_nodes(root) 
        expansion_ratio = expanded_count / max(1, total_nodes)
        health["metrics"]["expansion_ratio"] = expansion_ratio
        
        if expansion_ratio > 0.9 and total_nodes > 20:
            health["issues"].append({
                "type": "high_expansion_ratio",
                "message": f"Most nodes ({expanded_count}/{total_nodes}) are marked expanded",
                "severity": "warning"
            })
        
        # Calculate health score (0-100)
        if health["status"] == "healthy" and not health["issues"]:
            health["score"] = 100
        else:
            # Reduce score based on issue severity
            score = 100
            for issue in health["issues"]:
                if issue["severity"] == "critical":
                    score -= 50
                elif issue["severity"] == "warning":
                    score -= 15
            health["score"] = max(0, score)
        
        return health

    def apply_health_fixes(self, root, health):
        """
        Apply fixes based on tree health check results.
        
        Args:
            root: Root node of the search tree
            health: Health check results from check_tree_health()
            
        Returns:
            list: Applied fixes
        """
        applied_fixes = []
        
        for issue in health["issues"]:
            if issue["type"] == "visit_inconsistency":
                # Correct visit counts
                with root.node_lock:
                    old_visits = root.visits
                    new_visits = sum(child.visits for child in root.children)
                    root.visits = max(new_visits, old_visits)
                    fix = {"type": "visit_correction", "message": f"Corrected root visits from {old_visits} to {root.visits}"}
                    applied_fixes.append(fix)
                    logger.warning(fix["message"])
            
            elif issue["type"] == "pending_nodes_buildup":
                # More aggressive clearing of pending nodes
                pending_size = len(self.pending_nodes)
                self.pending_nodes.clear()
                fix = {"type": "pending_clear", "message": f"Cleared {pending_size} pending nodes to unblock search"}
                applied_fixes.append(fix)
                logger.warning(fix["message"])
        
        return applied_fixes

    def adapt_search_parameters(self, root, health):
        """
        Adapt search parameters based on tree health and performance.
        
        Args:
            root: Root node of the search tree
            health: Health check results
            
        Returns:
            dict: Parameter adjustments
        """
        adjustments = {}
        
        # Get current parameters
        current_exploration = self.exploration_weight
        
        # If tree health is poor, make parameter adjustments
        if health["score"] < 70:
            # Extreme skew might indicate exploitation issues
            if any(issue["type"] == "skewed_distribution" for issue in health["issues"]):
                # Increase exploration to encourage broader search
                new_exploration = min(2.5, current_exploration * 1.2)
                if abs(new_exploration - current_exploration) > 0.05:
                    self.exploration_weight = new_exploration
                    adjustments["exploration_weight"] = {
                        "old": current_exploration,
                        "new": new_exploration,
                        "reason": "Increasing exploration to address visit skew"
                    }
                    logger.info(f"Increasing exploration weight: {current_exploration:.2f} → {new_exploration:.2f}")
        
        # If we're making good progress but batch utilization is low
        elif health["score"] >= 85 and hasattr(self.evaluator, 'batch_sizes'):
            avg_batch_size = np.mean(self.evaluator.batch_sizes) if self.evaluator.batch_sizes else 0
            if avg_batch_size < self.min_batch_size / 2:
                # Tree is too deeply exploited - increase exploration
                new_exploration = min(2.5, current_exploration * 1.15)
                if abs(new_exploration - current_exploration) > 0.05:
                    self.exploration_weight = new_exploration
                    adjustments["exploration_weight"] = {
                        "old": current_exploration,
                        "new": new_exploration,
                        "reason": "Increasing exploration to improve batch utilization"
                    }
                    logger.info(f"Increasing exploration weight: {current_exploration:.2f} → {new_exploration:.2f}")
        
        # Update collectors with new exploration weight if changed
        if "exploration_weight" in adjustments:
            for collector, _ in self.collectors:
                collector.exploration_weight = self.exploration_weight
        
        return adjustments
    
    def search(self, root_state, num_simulations: int = 800, add_dirichlet_noise: bool = True) -> Tuple[Any, Dict]:
        """
        Perform MCTS search from root state with performance monitoring and tree health checks.
        
        Args:
            root_state: Initial game state
            num_simulations: Number of simulations to run
            add_dirichlet_noise: Whether to add Dirichlet noise at root
                    
        Returns:
            Tuple: (root node, statistics dictionary)
        """
        from mcts.node import Node
        from mcts.core import expand_node, backpropagate

        # Start timing and monitoring
        self.start_time = time.time()
        self.total_simulations = num_simulations
        
        # Create performance monitor
        if self.collect_stats:
            self.performance_monitor = MCTSPerformanceMonitor(self)
            self.performance_monitor.start_monitoring()
        
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
            
            # Mark all immediate children as unexpanded
            for child in root.children:
                child.is_expanded = False
                
            # Add to expanded nodes set for tracking
            self.expanded_nodes.add(id(root))
        
        # CRITICAL FIX: Handle case where root has no children (terminal state or bug)
        if not root.children:
            logger.warning("Root node has no children - this may be a terminal state or a policy issue")
            return root, {"total_time": 0.0, "total_simulations": 1, "sims_per_second": 0.0}
        
        # Initialize workers
        self._start_workers(root)
        
        # Wait for simulations to complete with monitoring
        sims_completed = 1  # Count root evaluation
        last_update_time = time.time()
        update_interval = 1.0  # 1 second
        last_sims_count = 1
        stall_start_time = None
        
        # Track health check metrics
        last_health_check_time = time.time()
        health_check_interval = 3.0  # Seconds between health checks
        health_checks_performed = 0
        parameter_adaptations = []
        health_fixes_applied = []
        
        try:
            while sims_completed < num_simulations:
                # Check simulation progress
                sims_completed = 1 + self.processor.results_processed if self.processor else 1
                
                # Record progress in performance monitor
                if self.collect_stats:
                    self.performance_monitor.record_progress(sims_completed, num_simulations)
                    if self.performance_monitor.check_performance():
                        # If performance issues detected, log warnings
                        bottlenecks = self.performance_monitor.bottlenecks
                        if bottlenecks and len(bottlenecks) > 0:
                            latest = bottlenecks[-1]
                            logger.warning(f"Performance issue: {latest['message']}")
                
                # Detect stalls
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
                
                # Periodic tree health check with adaptive parameters
                current_time = time.time()
                if current_time - last_health_check_time > health_check_interval:
                    # Run tree health check
                    health = self.check_tree_health(root)
                    health_checks_performed += 1
                    
                    # Log health summary if issues found
                    if health["status"] != "healthy" or health["issues"]:
                        issues_count = len(health["issues"])
                        logger.info(f"Tree health check: score={health['status']}, score={health['score']}/100, {issues_count} issues")
                    
                    # Apply fixes if health score is low
                    if health["score"] < 85:
                        fixes = self.apply_health_fixes(root, health)
                        health_fixes_applied.extend(fixes)
                    
                    # Adapt search parameters based on tree health
                    adaptations = self.adapt_search_parameters(root, health)
                    if adaptations:
                        parameter_adaptations.append({
                            "time": current_time - self.start_time,
                            "adaptations": adaptations
                        })
                    
                    # Update health check time
                    last_health_check_time = current_time
                    
                    # Adjust health check interval based on health score
                    # Check more frequently if health is poor
                    if health["score"] < 70:
                        health_check_interval = 1.0  # Check every second if health is poor
                    elif health["score"] < 85:
                        health_check_interval = 2.0  # Check every 2 seconds if health is moderate
                    else:
                        health_check_interval = 3.0  # Check every 3 seconds if health is good
                
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
        
        # Generate performance report
        if self.collect_stats and hasattr(self, 'performance_monitor'):
            self.performance_report = self.performance_monitor.generate_report()
        else:
            self.performance_report = {}
        
        # Log search summary
        if self.collect_stats:
            self._log_search_summary(root)
        
        # Get stats
        stats = self.get_search_stats()
        
        # Add performance report to stats if available
        if hasattr(self, 'performance_report'):
            stats['performance'] = self.performance_report
        
        # Add health checks information
        stats['health_checks'] = {
            'count': health_checks_performed,
            'fixes_applied': health_fixes_applied,
            'parameter_adaptations': parameter_adaptations
        }
        
        return root, stats
    
    def _start_workers(self, root):
        """Start worker threads for search with improved thread management"""
        
        if self.use_thread_pool:
            # Use thread pool for collectors
            self.collector_futures = []
            for i in range(self.num_collectors):
                collector = LeafCollector(
                    root=root,
                    eval_queue=self.eval_queue,
                    result_queue=self.result_queue,
                    lock=self.tree_lock,  # Pass tree_lock as 'lock' for backward compatibility
                    batch_size=max(1, self.batch_size // self.num_collectors),
                    min_batch_size=self.min_batch_size,
                    max_queue_size=self.batch_size * 2,
                    exploration_weight=self.exploration_weight,
                    max_collection_time=self.collector_timeout,
                    expanded_nodes=self.expanded_nodes,
                    pending_nodes=self.pending_nodes,
                    verbose=self.verbose
                )
                future = self.collector_pool.submit(collector.run)
                self.collector_futures.append((collector, future))
        else:
            # Traditional approach with individual threads
            self.collectors = []
            for i in range(self.num_collectors):
                collector = LeafCollector(
                    root=root,
                    eval_queue=self.eval_queue,
                    result_queue=self.result_queue,
                    lock=self.tree_lock,  # Pass tree_lock as 'lock' for backward compatibility
                    batch_size=max(1, self.batch_size // self.num_collectors),
                    max_queue_size=self.batch_size * 2,
                    exploration_weight=self.exploration_weight,
                    max_collection_time=self.collector_timeout,
                    expanded_nodes=self.expanded_nodes,
                    pending_nodes=self.pending_nodes,
                    verbose=self.verbose
                )
                thread = threading.Thread(
                    target=collector.run,
                    daemon=True,
                    name=f"leaf_collector_{i}"
                )
                thread.start()
                self.collectors.append((collector, thread))
        
        # Create and start evaluator with improved parameters and verbose flag
        self.evaluator = Evaluator(
            root=root,
            inference_fn=self.inference_fn,
            eval_queue=self.eval_queue,
            result_queue=self.result_queue,
            batch_size=self.batch_size,
            max_wait_time=self.evaluator_wait_time,
            min_batch_size=self.min_batch_size,
            expanded_nodes=self.expanded_nodes,
            pending_nodes=self.pending_nodes,
            verbose=self.verbose  # Pass verbose flag
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
        """Shutdown worker threads with improved cleanup"""
        # Signal shutdown
        self.shutdown_flag.set()
        
        # Shutdown collectors
        if self.use_thread_pool:
            # Shutdown with thread pool approach
            for collector, future in self.collector_futures:
                collector.shutdown()
            
            # Shutdown the thread pool with a timeout
            self.collector_pool.shutdown(wait=False)
            
            # Give a brief timeout for collector futures
            for _, future in self.collector_futures:
                try:
                    future.result(timeout=0.5)  # Wait up to 0.5s for each collector
                except:
                    pass  # Ignore errors during shutdown
        else:
            # Traditional approach
            for collector, thread in self.collectors:
                collector.shutdown()
                thread.join(timeout=0.5)  # Increased timeout
        
        # Shutdown evaluator and result processor
        if self.evaluator:
            self.evaluator.shutdown()
        if self.processor:
            self.processor.shutdown()
        
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
        """Enhanced diagnostic logging for MCTS stalls with tree health integration"""
        logger.warning("=== MCTS STALL DETECTED ===")
        
        # Run tree health check first
        health = self.check_tree_health(root)
        
        # Log health status
        if health["issues"]:
            logger.warning(f"Tree health issues: {len(health['issues'])}")
            for issue in health["issues"]:
                logger.warning(f"- {issue['severity'].upper()}: {issue['message']}")
        
        # Root node analysis
        expandable = sum(1 for c in root.children if not c.is_expanded)
        logger.warning(f"Root node: {len(root.children)} children, {root.visits} visits, {expandable} unexpanded")
        
        # Child visit distribution to detect exploration issues
        if root.children:
            visits = [child.visits for child in root.children]
            max_visits = max(visits)
            min_visits = min(visits)
            avg_visits = sum(visits) / len(visits)
            logger.warning(f"Child visits: min={min_visits}, max={max_visits}, avg={avg_visits:.1f}")
            
            # Check for highly skewed visit distribution
            if max_visits > 10 * avg_visits:
                logger.warning("Visit distribution is highly skewed - may indicate policy issues")
        
        # Queue and node tracking analysis
        logger.warning(f"Pending nodes: {len(self.pending_nodes)}")
        logger.warning(f"Expanded nodes: {len(self.expanded_nodes)}")
        logger.warning(f"Eval queue size: {self.eval_queue.qsize()}")
        logger.warning(f"Result queue size: {self.result_queue.qsize()}")
        
        # Worker analysis
        for i, (collector, _) in enumerate(self.collectors):
            stats = collector.get_stats()
            logger.warning(f"Collector {i}: collected={stats.get('leaves_collected', 0)}, "
                        f"empty batches={getattr(collector, 'consecutive_empty_batches', 0)}")
        
        # Evaluator stats
        if self.evaluator:
            stats = self.evaluator.get_stats()
            logger.warning(f"Evaluator: batches={stats.get('batches_evaluated', 0)}, "
                        f"size={stats.get('avg_batch_size', 0):.1f}")
        
        # Processor stats
        if self.processor:
            stats = self.processor.get_stats()
            logger.warning(f"Processor: processed={stats.get('results_processed', 0)}, "
                        f"errors={stats.get('errors', 0)}")
        
        # CRITICAL: Tree health check
        expanded_nodes_in_tree = self._count_expanded_nodes(root)
        if expanded_nodes_in_tree != len(self.expanded_nodes):
            logger.warning(f"Tree state mismatch: {expanded_nodes_in_tree} expanded nodes in tree, "
                        f"but {len(self.expanded_nodes)} in tracking set")
        
        # CRITICAL: Apply tree health fixes and parameter adjustments when stalled
        if health["score"] < 85:  # Health is not optimal
            # Apply aggressive fixes during stalls
            fixes = self.apply_health_fixes(root, health)
            if fixes:
                logger.warning(f"Applied {len(fixes)} health fixes to recover from stall:")
                for fix in fixes:
                    logger.warning(f"- {fix['message']}")
            
            # Apply more aggressive parameter adaptations during stalls
            adaptations = self.adapt_search_parameters(root, health)
            if adaptations:
                for param, info in adaptations.items():
                    logger.warning(f"Adjusted {param} from {info['old']:.2f} to {info['new']:.2f} - {info['reason']}")
        else:
            # If tree health is good but we're still stalled, use more targeted recovery
            self._recover_from_stall(root)

    def _recover_from_stall(self, root):
        """Attempt to recover from stalled search"""
        # If all root children are expanded but search is stalled
        all_expanded = all(child.is_expanded for child in root.children) if root.children else False
        
        if all_expanded:
            logger.warning("All root children are marked expanded but search is stalled")
            
            # Reset expansion state for children to enable further exploration
            reset_count = 0
            for child in root.children:
                # Reset nodes with fewer visits than average to enable exploration
                avg_visits = root.visits / max(1, len(root.children))
                if child.visits < avg_visits * 0.7:
                    child.is_expanded = False
                    self.expanded_nodes.discard(id(child))
                    reset_count += 1
            
            if reset_count > 0:
                logger.warning(f"Reset expansion state of {reset_count} nodes to enable further exploration")
        
        # Increase exploration to try different paths
        old_exploration = self.exploration_weight
        new_exploration = min(2.5, old_exploration * 1.15)
        if abs(new_exploration - old_exploration) > 0.1:
            self.exploration_weight = new_exploration
            logger.info(f"Increasing exploration weight: {old_exploration:.2f} → {new_exploration:.2f}")
            
            # Update collectors with new exploration weight
            for collector, _ in self.collectors:
                collector.exploration_weight = self.exploration_weight

    def _count_expanded_nodes(self, node):
        """Count expanded nodes in tree for diagnostic purposes"""
        if node is None:
            return 0
            
        count = 1 if node.is_expanded else 0
        for child in node.children:
            count += self._count_expanded_nodes(child)
                
        return count
    
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
                batch_size=32,
                min_batch_size=8,
                max_queue_size=32,
                exploration_weight=1.4,
                max_collection_time=0.01,
                select_func=select_node_with_node_locks,
                expanded_nodes=None,
                pending_nodes=None,
                verbose=False):
        """Initialize the leaf collector with all required attributes"""
        self.root = root
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.tree_lock = lock  # Explicitly store as tree_lock
        self.lock = lock       # Keep for backward compatibility
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.max_queue_size = max_queue_size
        self.exploration_weight = exploration_weight
        self.max_collection_time = max_collection_time
        
        # Sets for tracking node status
        self.expanded_nodes = expanded_nodes if expanded_nodes is not None else set()
        self.pending_nodes = pending_nodes if pending_nodes is not None else set()
        
        # Use provided selection function or default
        self.select_func = select_func
        
        # Statistics
        self.leaves_collected = 0
        self.collection_times = deque(maxlen=100)
        self.wait_times = deque(maxlen=100)
        self.consecutive_empty_batches = 0  # Add this counter
        
        # Control flags
        self.shutdown_flag = False
        self.verbose = verbose
    
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
                    for leaf, path, virtual_visits in batch:
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
        """Collect batch of nodes with improved selection logic"""
        batch = []
        
        # Calculate target batch size based on queue capacity
        queue_capacity = self.max_queue_size - self.eval_queue.qsize()
        target_size = min(self.batch_size, max(1, queue_capacity))
        if target_size <= 0:
            # Queue is full, no point collecting more nodes
            return []
            
        # Track collection time
        start_time = time.time()
        
        # Use a more dynamic retry strategy with exponential backoff
        max_retries = 20  # Reduced from 100 to avoid wasting time
        retries = 0
        
        # Track already visited nodes during this collection
        visited_nodes = set()
        
        # Important: Get a snapshot of expanded and pending nodes
        with self.tree_lock:
            local_expanded = set(self.expanded_nodes)
            local_pending = set(self.pending_nodes)
        
        # Main collection loop with better time management
        while (len(batch) < target_size and 
            (time.time() - start_time) < self.max_collection_time and 
            retries < max_retries):
            
            try:
                # Use proper virtual loss tracking
                leaf, path, virtual_losses = self.select_func(self.root, self.exploration_weight)
                leaf_id = id(leaf)
                
                # Skip if we've already visited this node in this batch collection
                if leaf_id in visited_nodes:
                    retries += 1
                    continue
                
                # Mark as visited to avoid checking the same node repeatedly
                visited_nodes.add(leaf_id)
                
                # Handle terminal nodes immediately
                if leaf.state.is_terminal():
                    with self.tree_lock:
                        value = leaf.state.get_winner()
                    # Send directly to result queue with virtual loss info
                    self.result_queue.put((leaf, path, (value, None), virtual_losses))
                    retries += 1
                    continue
                
                # Check node status with proper synchronization
                is_eligible = False
                with self.tree_lock:
                    # Double-check status under lock
                    is_eligible = (not leaf.is_expanded and 
                                leaf_id not in self.expanded_nodes and 
                                leaf_id not in self.pending_nodes)
                    
                    if is_eligible:
                        # Add to pending set
                        self.pending_nodes.add(leaf_id)
                        # Add to batch
                        batch.append((leaf, path, virtual_losses))
                
                if not is_eligible:
                    retries += 1
                    # Apply short sleep to reduce contention
                    if retries % 10 == 0:
                        time.sleep(0.001)  # 1ms sleep every 10 retries
                    
            except Exception as e:
                logger.error(f"Error collecting node: {e}")
                retries += 1
                time.sleep(0.001)  # Short sleep to avoid tight loop
        
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
                batch_size=32,
                max_wait_time=0.020,  # Increased from 0.005 to 20ms
                min_batch_size=8,     # Process batches of at least 8 nodes
                exploration_weight=1.4,
                expanded_nodes=None,
                pending_nodes=None,
                verbose=False):
        """Initialize the evaluator with all required attributes"""
        self.root = root
        self.inference_fn = inference_fn
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        self.exploration_weight = exploration_weight
        self.max_queue_size = batch_size * 2
        self.verbose = verbose
        
        # Sets for tracking node status
        self.expanded_nodes = expanded_nodes if expanded_nodes is not None else set()
        self.pending_nodes = pending_nodes if pending_nodes is not None else set()
        
        # Statistics - MAKE SURE ALL COLLECTIONS ARE INITIALIZED
        self.batches_evaluated = 0
        self.leaves_evaluated = 0
        self.batch_sizes = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)  # THIS WAS MISSING
        self.wait_times = deque(maxlen=100)
        
        # Control flags
        self.shutdown_flag = False
    
    def run(self):
        """Main worker loop for batch evaluation with improved batching strategy"""
        while not self.shutdown_flag:
            try:
                # Add debug logging to track queue size
                if self.verbose:
                    queue_size = self.eval_queue.qsize()
                    if queue_size > 0:
                        logger.debug(f"Eval queue size: {queue_size}")
                
                # Wait for substantial batch or timeout
                wait_start = time.time()
                batch_target_time = wait_start + self.max_wait_time
                min_batch_size = max(1, min(self.min_batch_size, self.batch_size // 2))
                
                # Determine batch collection strategy based on queue size
                queue_size = self.eval_queue.qsize()
                if queue_size >= self.batch_size:
                    # Queue has enough items for a full batch, collect it immediately
                    target_collect_size = min(self.batch_size, queue_size)
                    max_wait_time = 0.002  # Very short wait (2ms)
                elif queue_size >= min_batch_size:
                    # Queue has enough for a minimum batch, wait a bit for more
                    target_collect_size = min(self.batch_size, queue_size)
                    max_wait_time = self.max_wait_time / 2
                else:
                    # Queue has few items, wait longer for batch formation
                    target_collect_size = self.batch_size
                    max_wait_time = self.max_wait_time
                
                # Collect batch with timeout strategy
                batch = []
                batch_too_small = True
                
                # Try to collect first item
                try:
                    item = self.eval_queue.get(timeout=max_wait_time)
                    batch.append(item)
                    batch_too_small = len(batch) < min_batch_size
                except queue.Empty:
                    # No items available, sleep briefly and continue
                    time.sleep(0.001)
                    self.wait_times.append(time.time() - wait_start)
                    continue
                
                # Now try to collect remaining items up to target size
                collection_start = time.time()
                collection_timeout = min(max_wait_time, batch_target_time - collection_start)
                
                while len(batch) < target_collect_size and time.time() - collection_start < collection_timeout:
                    try:
                        # Use short timeout for remaining items
                        item = self.eval_queue.get(timeout=0.001)
                        batch.append(item)
                        
                        # Check if we have enough for minimum batch size
                        if batch_too_small and len(batch) >= min_batch_size:
                            batch_too_small = False
                            # Since we have minimum, reduce remaining timeout
                            collection_timeout = min(collection_timeout, 0.005)
                    except queue.Empty:
                        # If we have minimum batch size, wait less
                        if not batch_too_small:
                            break
                        time.sleep(0.0005)  # Very short sleep to avoid CPU spinning
                
                wait_time = time.time() - wait_start
                self.wait_times.append(wait_time)
                
                # Log batch collection stats
                if self.verbose and len(batch) > 1:
                    logger.debug(f"Collected batch of {len(batch)} items in {wait_time*1000:.1f}ms")
                
                if not batch:
                    # No items collected, sleep briefly
                    time.sleep(0.001)
                    continue
                
                # Process batch
                batch_size = len(batch)
                self.batch_sizes.append(batch_size)
                
                # Extract states for inference
                leaves, paths = zip(*batch)
                states = [leaf.state for leaf in leaves]
                
                # Perform inference
                inference_start = time.time()
                try:
                    results = self._evaluate_batch(states)
                    inference_time = time.time() - inference_start
                    self.inference_times.append(inference_time)
                    
                    # Send results to result queue
                    for i, (leaf, path) in enumerate(zip(leaves, paths)):
                        if i < len(results):
                            result = results[i]
                        else:
                            # Fallback for mismatch
                            result = (np.ones(9)/9, 0.0)
                        
                        # Put the result in a standardized format - leaf, path, result (no virtual_visits)
                        self.result_queue.put((leaf, path, result))
                    
                    self.leaves_evaluated += batch_size
                    self.batches_evaluated += 1
                    
                    # Log successful batch processing
                    if self.verbose:
                        avg_batch = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
                        if batch_size > 1 or self.batches_evaluated % 10 == 0:
                            logger.debug(f"Processed batch {self.batches_evaluated}: size={batch_size}, " +
                                    f"avg_size={avg_batch:.1f}, time={inference_time*1000:.1f}ms")
                    
                except Exception as e:
                    logger.error(f"Batch evaluation error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Return default values on error
                    default_policy = np.ones(9) / 9  # Uniform policy
                    default_value = 0.0  # Neutral value
                    
                    for leaf, path in batch:
                        self.result_queue.put((leaf, path, (default_policy, default_value)))
                    
            except Exception as e:
                logger.error(f"Error in evaluator: {e}")
                import traceback
                traceback.print_exc()
                
                # Sleep to avoid error flooding
                time.sleep(0.01)
    
    def _collect_batch_from_queue(self):
        """Collect batch with staggered timing to reduce contention"""
        batch = []
        
        # Use progressive timeouts
        if self.eval_queue.qsize() >= self.batch_size:
            # Queue has enough items, collect quickly
            timeout = 0.001  # 1ms
            target_size = self.batch_size
        else:
            # Queue is building up, wait longer
            timeout = 0.01  # 10ms
            target_size = max(self.min_batch_size, self.eval_queue.qsize())
        
        # Try to collect the batch
        start_time = time.time()
        max_wait = 0.05  # 50ms max wait
        
        while len(batch) < target_size and time.time() - start_time < max_wait:
            try:
                item = self.eval_queue.get(timeout=timeout)
                batch.append(item)
                # Reduce timeout as we collect more items
                timeout = max(0.0005, timeout * 0.8)
            except queue.Empty:
                # If we have minimum batch size, proceed
                if len(batch) >= self.min_batch_size:
                    break
                # Shorter timeouts for subsequent attempts
                timeout = max(0.0005, timeout * 0.5)
        
        return batch
    
    def _evaluate_batch(self, states):
        """Perform neural network inference with improved error handling"""
        try:
            # Handle empty batch (shouldn't happen)
            if not states:
                return []
                
            # Special case for batch size 1
            if len(states) == 1:
                # Single state evaluation
                try:
                    result = self.inference_fn(states[0])
                    return [result]
                except Exception as e:
                    logger.error(f"Error evaluating single state: {e}")
                    return [(np.ones(9)/9, 0.0)]  # Return default
            
            # Batch inference
            try:
                batch_results = self.inference_fn(states)
                
                # Log successful batch processing
                if self.verbose and len(states) > 1:
                    logger.debug(f"Successful inference for batch of {len(states)} states")
                
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
            # "avg_evaluation_time": np.mean(self.evaluation_times) if self.evaluation_times else 0,
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
                    queue_item = self.result_queue.get(timeout=0.01)  # 10ms timeout
                    self.wait_times.append(time.time() - wait_start)
                    
                    # Unpack the queue item - handle different formats
                    if len(queue_item) == 3:
                        leaf, path, result = queue_item
                        virtual_visits = None
                    elif len(queue_item) == 4:
                        leaf, path, result, virtual_visits = queue_item
                    else:
                        # Unexpected format - log and skip
                        logger.error(f"Unexpected result queue item format: {queue_item}")
                        continue
                        
                except queue.Empty:
                    continue
                
                # Process result
                processing_start = time.time()
                success = self._process_result(leaf, path, result, virtual_visits)
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
    
    def _process_result(self, leaf, path, result, virtual_visits=None):
        """Process evaluation result with proper virtual loss handling"""
        from mcts.core import expand_node, backpropagate
        
        virtual_visits = virtual_visits or {}
        leaf_id = id(leaf)
        
        with self.lock:
            # Always remove from pending
            self.pending_nodes.discard(leaf_id)
            
            # Skip if already expanded, but still backpropagate
            if leaf.is_expanded or leaf_id in self.expanded_nodes:
                if isinstance(result, tuple) and len(result) == 2:
                    _, value = result
                    backpropagate(path, value)  # Use simplified backpropagation
                return True
            
            # Handle terminal states
            if leaf.state.is_terminal():
                value = leaf.state.get_winner()
                backpropagate(path, value)
                return True
                
            # Handle normal evaluation
            try:
                policy, value = result
                
                # Expand leaf with the policy
                expand_node(leaf, policy)
                
                # Mark as expanded
                leaf.is_expanded = True
                self.expanded_nodes.add(leaf_id)
                
                # Backpropagate the value
                backpropagate(path, value)
                return True
            except Exception as e:
                logger.error(f"Error processing result: {e}")
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

class MCTSPerformanceMonitor:
    """
    Performance monitoring and diagnostics for MCTS operations.
    
    This class provides real-time monitoring of MCTS operations,
    collects performance metrics, and helps diagnose bottlenecks.
    """
    
    def __init__(self, mcts_instance):
        """Initialize with reference to MCTS instance"""
        self.mcts = mcts_instance
        self.metrics = {
            'batch_sizes': deque(maxlen=100),
            'collection_times': deque(maxlen=100),
            'inference_times': deque(maxlen=100),
            'node_selection_times': deque(maxlen=100),
            'expansion_times': deque(maxlen=100),
            'backprop_times': deque(maxlen=100),
            'progress_snapshots': []
        }
        self.start_time = None
        self.simulation_progress = []
        self.bottlenecks = []
        self.last_check_time = time.time()
        self.check_interval = 5.0  # Seconds between checks
    
    def start_monitoring(self):
        """Begin monitoring performance"""
        self.start_time = time.time()
        self.simulation_progress = []
        self.bottlenecks = []
    
    def record_metric(self, metric_name, value):
        """Record a metric value"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def record_progress(self, sims_completed, total_sims):
        """Record simulation progress snapshot"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        self.simulation_progress.append({
            'time': elapsed,
            'sims_completed': sims_completed,
            'percentage': sims_completed / max(1, total_sims) * 100
        })
        
        # Calculate instantaneous speed
        if len(self.simulation_progress) >= 2:
            prev = self.simulation_progress[-2]
            time_diff = elapsed - prev['time']
            sims_diff = sims_completed - prev['sims_completed']
            
            if time_diff > 0:
                speed = sims_diff / time_diff
                self.metrics['progress_snapshots'].append({
                    'time': elapsed,
                    'speed': speed
                })
    
    def check_performance(self):
        """
        Check for performance issues and bottlenecks.
        Returns True if issues were found.
        """
        current_time = time.time()
        
        # Only check periodically
        if current_time - self.last_check_time < self.check_interval:
            return False
            
        self.last_check_time = current_time
        issues_found = False
        
        # Get latest metrics
        avg_batch_size = np.mean(self.metrics['batch_sizes']) if self.metrics['batch_sizes'] else 0
        avg_inference_time = np.mean(self.metrics['inference_times']) if self.metrics['inference_times'] else 0
        
        # Check for poor GPU utilization
        if avg_batch_size < self.mcts.min_batch_size:
            self.bottlenecks.append({
                'time': current_time - self.start_time,
                'type': 'low_batch_size',
                'value': avg_batch_size,
                'message': f"Low batch size ({avg_batch_size:.1f}) indicates poor GPU utilization"
            })
            issues_found = True
        
        # Check simulation progress rate
        if len(self.simulation_progress) >= 3:
            recent_progress = self.simulation_progress[-3:]
            times = [p['time'] for p in recent_progress]
            sims = [p['sims_completed'] for p in recent_progress]
            
            # Calculate speed over recent period
            if times[-1] - times[0] > 0:
                speed = (sims[-1] - sims[0]) / (times[-1] - times[0])
                
                # Flag as issue if speed is very low
                if speed < 10 and self.mcts.total_simulations > 100:
                    self.bottlenecks.append({
                        'time': current_time - self.start_time,
                        'type': 'low_simulation_rate',
                        'value': speed,
                        'message': f"Low simulation rate ({speed:.1f} sims/s) indicates a bottleneck"
                    })
                    issues_found = True
        
        # Check queue imbalance
        eval_queue_size = self.mcts.eval_queue.qsize()
        result_queue_size = self.mcts.result_queue.qsize()
        
        if eval_queue_size > 20 * result_queue_size + 5:
            self.bottlenecks.append({
                'time': current_time - self.start_time,
                'type': 'evaluator_bottleneck',
                'value': eval_queue_size,
                'message': f"Evaluator bottleneck: {eval_queue_size} items waiting for evaluation"
            })
            issues_found = True
        
        elif result_queue_size > 20 * eval_queue_size + 5:
            self.bottlenecks.append({
                'time': current_time - self.start_time,
                'type': 'processor_bottleneck',
                'value': result_queue_size,
                'message': f"Result processor bottleneck: {result_queue_size} items waiting for processing"
            })
            issues_found = True
        
        return issues_found
    
    def generate_report(self):
        """Generate a comprehensive performance report"""
        if not self.start_time:
            return {"error": "No monitoring data available"}
            
        elapsed_time = time.time() - self.start_time
        
        # Calculate key metrics
        avg_batch_size = np.mean(self.metrics['batch_sizes']) if self.metrics['batch_sizes'] else 0
        avg_inference_time = np.mean(self.metrics['inference_times']) if self.metrics['inference_times'] else 0
        avg_node_selection = np.mean(self.metrics['node_selection_times']) if self.metrics['node_selection_times'] else 0
        avg_expansion = np.mean(self.metrics['expansion_times']) if self.metrics['expansion_times'] else 0
        avg_backprop = np.mean(self.metrics['backprop_times']) if self.metrics['backprop_times'] else 0
        
        # Calculate overall simulation speed
        if self.simulation_progress:
            final_sims = self.simulation_progress[-1]['sims_completed']
            overall_speed = final_sims / elapsed_time if elapsed_time > 0 else 0
        else:
            final_sims = 0
            overall_speed = 0
        
        # Collect node statistics
        expanded_nodes = len(self.mcts.expanded_nodes)
        pending_nodes = len(self.mcts.pending_nodes)
        total_nodes = getattr(self.mcts, 'total_nodes', 0)
        
        # Generate phase breakdown
        phase_times = {
            'selection': avg_node_selection,
            'inference': avg_inference_time,
            'expansion': avg_expansion,
            'backpropagation': avg_backprop
        }
        
        # Identify bottleneck phase
        if sum(phase_times.values()) > 0:
            bottleneck_phase = max(phase_times.items(), key=lambda x: x[1])
        else:
            bottleneck_phase = ('unknown', 0)
        
        # Generate performance recommendations
        recommendations = []
        
        if avg_batch_size < self.mcts.min_batch_size:
            recommendations.append({
                'focus': 'batching',
                'message': f"Increase batching efficiency (current avg: {avg_batch_size:.1f})",
                'actions': [
                    "Increase min_batch_size",
                    "Increase evaluator_wait_time",
                    "Reduce lock contention in node selection"
                ]
            })
        
        if bottleneck_phase[0] == 'inference' and avg_batch_size < self.mcts.batch_size / 2:
            recommendations.append({
                'focus': 'gpu_utilization',
                'message': "Improve GPU utilization",
                'actions': [
                    "Use mixed precision inference",
                    "Optimize neural network architecture for inference",
                    "Ensure sufficient batch sizes for efficient GPU utilization"
                ]
            })
        
        if bottleneck_phase[0] == 'selection' and avg_node_selection > 0.005:
            recommendations.append({
                'focus': 'tree_traversal',
                'message': "Optimize tree traversal",
                'actions': [
                    "Use Numba-accelerated traversal",
                    "Reduce lock contention during selection",
                    "Consider vectorized node selection algorithms"
                ]
            })
        
        # Compile the report
        report = {
            'duration': elapsed_time,
            'simulations_completed': final_sims,
            'overall_speed': overall_speed,
            'batch_statistics': {
                'average_size': avg_batch_size,
                'target_size': self.mcts.batch_size,
                'min_size': self.mcts.min_batch_size,
                'utilization': avg_batch_size / self.mcts.batch_size if self.mcts.batch_size > 0 else 0
            },
            'timing_breakdown': {
                phase: time_ms * 1000 for phase, time_ms in phase_times.items()
            },
            'bottleneck_phase': bottleneck_phase[0],
            'node_statistics': {
                'expanded': expanded_nodes,
                'pending': pending_nodes,
                'total': total_nodes,
                'expansion_ratio': expanded_nodes / max(1, total_nodes)
            },
            'detected_bottlenecks': self.bottlenecks,
            'recommendations': recommendations
        }
        
        return report

def get_optimal_collector_count():
    """Determine optimal number of collector threads based on CPU cores"""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    # For Ryzen 9 5900X with 12 cores / 24 threads:
    # Use 75% of logical cores, leaving some for the main process and system
    return max(2, min(int(cpu_count * 0.75), 16))

def leaf_parallel_search(root_state, inference_fn, num_simulations=800, 
                        num_collectors=None,  # Default to None for auto-detection
                        batch_size=64, exploration_weight=1.4,
                        add_dirichlet_noise=True, collect_stats=True,
                        collector_timeout=0.01, min_batch_size=16,
                        evaluator_wait_time=0.02, verbose=False,
                        adaptive_parameters=True):
    """
    Run leaf-parallel MCTS search with adaptive parameters and optimal thread usage.
    
    Args:
        root_state: Initial game state
        inference_fn: Function to perform neural network inference
        num_simulations: Number of simulations to run
        num_collectors: Number of collector threads (None = auto-detect based on CPU cores)
        batch_size: Maximum batch size for leaf evaluation
        exploration_weight: Controls exploration vs exploitation tradeoff
        add_dirichlet_noise: Whether to add Dirichlet noise at root
        collect_stats: Whether to collect performance statistics
        collector_timeout: Maximum collection time per batch
        min_batch_size: Minimum batch size for efficient inference
        evaluator_wait_time: Maximum wait time for batch formation
        verbose: Whether to print detailed logging
        adaptive_parameters: Whether to dynamically adjust parameters
        
    Returns:
        tuple: (root_node, statistics dictionary)
    """
    # Record start time for diagnostics
    search_start_time = time.time()
    
    # Auto-detect collector count if not specified
    if num_collectors is None:
        num_collectors = get_optimal_collector_count()
        logger.info(f"Auto-detected optimal collector count: {num_collectors}")
    
    try:
        # Print parameter information
        logger.info(f"Starting leaf parallel search with {num_collectors} collectors, " 
                  f"batch_size={batch_size}, min_batch_size={min_batch_size}, "
                  f"wait_time={evaluator_wait_time*1000:.1f}ms")
        
        # Enable debug logging if verbose
        if verbose:
            # Store original level for restoration
            original_level = logger.level
            logger.setLevel(logging.DEBUG)
        
        # Create MCTS instance with all parameters
        mcts = LeafParallelMCTS(
            inference_fn=inference_fn,
            num_collectors=num_collectors,
            batch_size=batch_size,
            exploration_weight=exploration_weight,
            collect_stats=collect_stats,
            collector_timeout=collector_timeout,
            min_batch_size=min_batch_size,
            evaluator_wait_time=evaluator_wait_time,
            verbose=verbose,
            adaptive_parameters=adaptive_parameters
        )
        
        # Run search
        root, stats = mcts.search(
            root_state=root_state,
            num_simulations=num_simulations,
            add_dirichlet_noise=add_dirichlet_noise
        )
        
        # Calculate actual search time
        total_search_time = time.time() - search_start_time
        logger.info(f"Search completed in {total_search_time:.3f}s for {num_simulations} simulations "
                  f"({num_simulations/total_search_time:.1f} sims/s)")
        
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
            policy = np.zeros(root_state.policy_size if hasattr(root_state, 'policy_size') else 9)
            for action in legal_actions:
                policy[action] = 1.0 / len(legal_actions)
                
            # Expand root with uniform policy
            expand_node(root, policy)
            root.visits = 1
        
        # Return root and minimal stats
        stats = {
            "error": str(e),
            "search_time": time.time() - search_start_time,
            "total_simulations": 0,
            "total_nodes": 1,
            "sims_per_second": 0
        }
        return root, stats
    finally:
        # Restore original log level if changed
        if verbose and 'original_level' in locals():
            logger.setLevel(original_level)