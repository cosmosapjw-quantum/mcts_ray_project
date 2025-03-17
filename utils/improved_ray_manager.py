# utils/improved_ray_manager.py
"""
Enhanced Ray configuration and actor management utilities for stable distributed processing.
"""
import os
import ray
import time
import logging
import threading
import psutil
from typing import Dict, List, Any, Optional, Callable, TypeVar, Type, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RayActorManager")

# Type variable for actor classes
T = TypeVar('T')

class RayActorManager:
    """
    Manager for Ray initialization and actor lifecycle management with enhanced fault tolerance.
    
    This class provides utilities for:
    1. Properly configuring Ray based on available system resources
    2. Creating and managing actor pools with automatic recovery
    3. Health monitoring with proactive failure detection
    4. Graceful shutdown and cleanup
    """
    
    def __init__(self, use_gpu=True, cpu_limit=None, memory_limit=None, object_store_limit=None):
        """
        Initialize Ray with proper resource configuration.
        
        Args:
            use_gpu: Whether to allocate GPU resources
            cpu_limit: Maximum number of CPUs to use (None = auto-detect)
            memory_limit: Memory limit in GB (None = auto-detect)
            object_store_limit: Object store memory limit in GB (None = auto-detect)
        """
        self.use_gpu = use_gpu
        self.initialized = False
        self.actor_pools = {}
        self.health_threads = {}
        self.shutdown_flag = threading.Event()
        
        # Auto-detect resource limits if not specified
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.object_store_limit = object_store_limit
        
        # Track recovery statistics
        self.recovery_attempts = {}
        self.actor_failures = {}
        self.last_health_check = {}
        
        # Initialize Ray with proper resource configuration
        self.init_ray()
    
    def init_ray(self, force_restart=False):
        """
        Initialize Ray with properly configured resources.
        
        Args:
            force_restart: Whether to force restart Ray if already initialized
        """
        if ray.is_initialized() and not force_restart:
            logger.info("Ray already initialized")
            self.initialized = True
            return
        
        if ray.is_initialized():
            logger.info("Shutting down existing Ray instance")
            ray.shutdown()
        
        # Auto-detect CPU resources if not specified, with reasonable limits
        if self.cpu_limit is None:
            # Use 3/4 of available CPUs to leave resources for system
            available_cpus = psutil.cpu_count(logical=True)
            self.cpu_limit = max(1, int(available_cpus * 0.75))
            # Reserve at least one CPU for system operations
            if self.cpu_limit >= available_cpus:
                self.cpu_limit = max(1, available_cpus - 1)
        
        # Auto-detect memory if not specified (in GB)
        if self.memory_limit is None:
            # Use 2/3 of system memory to leave room for OS and other processes
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            self.memory_limit = max(1, int(total_memory_gb * 0.66))
        
        # Configure object store memory (1/3 of allocated memory by default)
        if self.object_store_limit is None:
            self.object_store_limit = max(1, int(self.memory_limit / 3))
        
        # Find available GPUs if enabled
        num_gpus = 0
        if self.use_gpu:
            try:
                import torch
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    logger.info(f"Detected {num_gpus} GPUs for Ray allocation")
                else:
                    logger.warning("GPU requested but no CUDA devices found")
            except:
                logger.warning("Failed to detect GPUs, using CPU only")
        
        # Convert memory limits to bytes for Ray
        memory_bytes = int(self.memory_limit * 1024 * 1024 * 1024)
        object_store_bytes = int(self.object_store_limit * 1024 * 1024 * 1024)
        
        # Configure Ray with better memory management
        system_config = {
            "object_spilling_threshold": 0.7,  # More aggressive spilling (down from 0.8)
            "object_store_full_delay_ms": 50,   # Shorter delay (down from 100ms)
            "local_gc_interval_s": 20,         # More frequent garbage collection
            "memory_monitor_refresh_ms": 500   # Monitor memory more frequently
        }

        # Configure log limits to prevent excessive logging
        logging_config = {
            "logging_level": "warning",
            "log_to_driver": False
        }
        
        # Initialize Ray with configured resources
        try:
            ray.init(
                num_cpus=self.cpu_limit,
                num_gpus=num_gpus,
                _memory=memory_bytes,
                object_store_memory=object_store_bytes,
                include_dashboard=False,  # Disable dashboard for better stability
                ignore_reinit_error=True,
                _system_config=system_config,
                **logging_config
            )
            
            self.initialized = True
            
            logger.info(f"Ray initialized with {self.cpu_limit} CPUs, {num_gpus} GPUs, " +
                       f"{self.memory_limit}GB memory, {self.object_store_limit}GB object store")
            
            # Print runtime env info for debugging
            runtime_env = ray.get_runtime_context().runtime_env
            logger.debug(f"Ray runtime environment: {runtime_env}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            # Fall back to minimal configuration if initial attempt fails
            try:
                ray.init(
                    num_cpus=2,
                    num_gpus=1 if num_gpus > 0 else 0,
                    ignore_reinit_error=True
                    )
                self.initialized = True
                logger.warning("Ray initialized with minimal configuration after failure")
            except Exception as e2:
                logger.error(f"Failed to initialize Ray with fallback configuration: {e2}")
                raise
    
    def create_actor_pool(self, 
                          actor_class: Type[T], 
                          pool_name: str, 
                          num_actors: int = 1, 
                          actor_args: dict = None, 
                          per_actor_cpus: float = 1.0,
                          per_actor_gpus: float = 0.0,
                          health_check_method: str = None,
                          health_check_interval: float = 30.0,
                          max_restarts: int = 5) -> List[T]:
        """
        Create a pool of actors with health monitoring and automatic recovery.
        
        Args:
            actor_class: Ray actor class to instantiate
            pool_name: Name for this actor pool
            num_actors: Number of actors to create in the pool
            actor_args: Arguments to pass to each actor constructor
            per_actor_cpus: CPUs to allocate per actor
            per_actor_gpus: GPUs to allocate per actor
            health_check_method: Function to check actor health (Optional)
            health_check_interval: Seconds between health checks
            max_restarts: Maximum number of restart attempts per actor
            
        Returns:
            List of actor handles
        """
        if not self.initialized:
            self.init_ray()
        
        # Use empty dict if no actor_args provided
        actor_args = actor_args or {}
        
        # Create the actor options with resource requirements
        actor_options = {
            "num_cpus": per_actor_cpus,
            "num_gpus": per_actor_gpus,
            "max_restarts": max_restarts,  # Automatic restarts on failure
            "max_task_retries": 3,  # Retry tasks on failure
        }
        
        # Create actor configurations with IDs for tracking
        actor_configs = []
        for i in range(num_actors):
            actor_id = f"{pool_name}_{i}"
            actor_configs.append({
                "id": actor_id,
                "actor": None,
                "options": dict(actor_options),
                "args": dict(actor_args),
                "status": "initializing",
                "restart_count": 0,
                "last_health_check": datetime.now()
            })
        
        # Create actors
        actors = []
        for config in actor_configs:
            actor = self._create_single_actor(
                actor_class=actor_class, 
                actor_id=config["id"], 
                options=config["options"], 
                args=config["args"]
            )
            
            if actor:
                config["actor"] = actor
                config["status"] = "running"
                actors.append(actor)
            else:
                config["status"] = "failed"
                logger.error(f"Failed to create actor {config['id']}")
        
        # Store in actor pools dictionary
        self.actor_pools[pool_name] = {
            "configs": actor_configs,
            "actors": actors,
            "actor_class": actor_class,
            "health_check_method": health_check_method,
            "health_check_interval": health_check_interval,
            "max_restarts": max_restarts,
            "active": True
        }
        
        # Initialize recovery statistics
        self.recovery_attempts[pool_name] = 0
        self.actor_failures[pool_name] = 0
        self.last_health_check[pool_name] = datetime.now()
        
        # Start health monitoring if method provided
        if health_check_method and actors:
            self._start_health_monitoring(pool_name, health_check_interval)
        
        return actors
    
    def _create_single_actor(self, actor_class, actor_id, options, args):
        """Create a single actor with error handling"""
        try:
            # For GPU actors, ensure memory is clean
            if options.get('num_gpus', 0) > 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info(f"Cleaned CUDA memory before creating actor {actor_id}")
                        
                        # Add a short wait to ensure memory is released
                        time.sleep(2.0)
                except ImportError:
                    pass
            
            # Create unique name for this actor
            logger.info(f"Creating actor {actor_id} with {options.get('num_cpus', 1)} CPUs and {options.get('num_gpus', 0)} GPUs")
            
            # Create the actor with specified options
            actor = actor_class.options(**options).remote(**args)
            logger.info(f"Actor {actor_id} created successfully")
            
            return actor
        except Exception as e:
            logger.error(f"Failed to create actor {actor_id}: {e}")
            return None
    
    def _start_health_monitoring(self, pool_name: str, interval: float = 30.0):
        """Start health monitoring thread for an actor pool"""
        if pool_name not in self.actor_pools:
            logger.error(f"Cannot monitor non-existent pool: {pool_name}")
            return
            
        if pool_name in self.health_threads and self.health_threads[pool_name].is_alive():
            logger.warning(f"Health monitoring already active for pool: {pool_name}")
            return
            
        # Create and start the monitoring thread
        thread = threading.Thread(
            target=self._health_monitoring_worker,
            args=(pool_name, interval),
            daemon=True,
            name=f"health_monitor_{pool_name}"
        )
        self.health_threads[pool_name] = thread
        thread.start()
        
        logger.info(f"Started health monitoring for actor pool: {pool_name} (interval: {interval}s)")
    
    def _health_monitoring_worker(self, pool_name: str, interval: float):
        """
        Worker thread for monitoring actor health.
        
        This method continuously monitors the health of all actors in a pool,
        recreating any actors that fail health checks or become unresponsive.
        
        Args:
            pool_name: Name of the actor pool to monitor
            interval: Time between health checks in seconds
        """
        logger.info(f"Starting health monitoring for pool: {pool_name} with interval {interval}s")
        
        # Add initial delay to give actors time to initialize
        time.sleep(10.0)  # Give actors 10 seconds to initialize
        
        while not self.shutdown_flag.is_set() and pool_name in self.actor_pools:
            try:
                # Get pool info
                pool_info = self.actor_pools[pool_name]
                
                # Check if pool is still active
                if not pool_info["active"]:
                    logger.info(f"Pool {pool_name} marked inactive, stopping health monitoring")
                    break
                
                # Update last health check time
                self.last_health_check[pool_name] = time.time()
                
                # Check each actor in the pool
                for i, config in enumerate(pool_info["configs"]):
                    # Skip actors that are not running
                    if config["status"] != "running" or not config["actor"]:
                        continue
                    
                    actor = config["actor"]
                    actor_id = config["id"]
                    
                    try:
                        # Get the health check method name
                        health_check_method = pool_info.get("health_check_method")
                        
                        # If no specific method is provided, look for common health methods
                        if not health_check_method:
                            if hasattr(actor, "get_health_status"):
                                health_check_method = "get_health_status"
                            elif hasattr(actor, "ping"):
                                health_check_method = "ping"
                        
                        # If we have a method to call, perform the health check
                        if health_check_method:
                            # Call the remote method and get result with timeout
                            health_future = getattr(actor, health_check_method).remote()
                            health_result = ray.get(health_future, timeout=5.0)
                            
                            # Process the result based on its type
                            is_healthy = False
                            if isinstance(health_result, bool):
                                is_healthy = health_result
                            elif isinstance(health_result, dict) and "status" in health_result:
                                status = health_result["status"]
                                # Consider "initializing" as HEALTHY to avoid premature recreation
                                is_healthy = (status in ["ready", "healthy", "ok", "initializing"] and 
                                            "error" not in status.lower())
                                
                                # Only recreate if explicitly failed (not just initializing)
                                if not is_healthy and status == "initializing":
                                    # Check how long it's been initializing
                                    uptime = health_result.get("uptime", 0)
                                    if uptime < 30.0:  # Give 30 seconds for initialization
                                        logger.info(f"Actor {actor_id} still initializing (uptime: {uptime:.1f}s), waiting...")
                                        is_healthy = True  # Consider it healthy for now
                            else:
                                is_healthy = health_result is not None
                            
                            # Handle unhealthy actors
                            if not is_healthy:
                                logger.warning(f"Actor {actor_id} failed health check with result: {health_result}")
                                self._recreate_actor(pool_name, i)
                        else:
                            # If no health check method, just check if actor is responding
                            try:
                                # Try to ping the actor by calling a simple method
                                # Most actors should have __ray_terminate__ method
                                ray.get(actor.__ray_ping__.remote(), timeout=2.0)
                                # If we get here, actor is responding
                            except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                                logger.warning(f"Actor {actor_id} is not responding to ping")
                                self._recreate_actor(pool_name, i)
                    
                    except ray.exceptions.RayActorError as e:
                        logger.warning(f"Actor {actor_id} is dead: {e}")
                        self._recreate_actor(pool_name, i)
                    except ray.exceptions.GetTimeoutError as e:
                        logger.warning(f"Health check timeout for actor {actor_id}: {e}")
                        self._recreate_actor(pool_name, i)
                    except AttributeError as e:
                        logger.warning(f"Health check method not found for actor {actor_id}: {e}")
                        # Only recreate if the actor doesn't exist at all
                        if "object has no attribute '__ray_ping__'" in str(e):
                            self._recreate_actor(pool_name, i)
                    except Exception as e:
                        logger.error(f"Unexpected error checking health of actor {actor_id}: {e}")
                        # Only recreate for serious errors
                        if any(err in str(e).lower() for err in ["dead", "died", "killed", "terminated"]):
                            self._recreate_actor(pool_name, i)
                
                # Log statistics periodically (every 5 checks)
                check_count = getattr(self, '_health_check_count', 0) + 1
                setattr(self, '_health_check_count', check_count)
                
                if check_count % 5 == 0:
                    # Count actors by status
                    status_counts = {}
                    for config in pool_info["configs"]:
                        status = config["status"]
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    # Log current pool status
                    logger.info(f"Pool {pool_name} health status: {status_counts}")
                    logger.info(f"Recovery attempts: {self.recovery_attempts.get(pool_name, 0)}")
                    logger.info(f"Actor failures: {self.actor_failures.get(pool_name, 0)}")
                
                # Sleep until next check
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring for pool {pool_name}: {e}")
                
                # Don't exit the monitoring loop on error, just sleep and try again
                time.sleep(max(1.0, interval / 2))
        
        logger.info(f"Health monitoring stopped for pool: {pool_name}")
    
    def _recreate_actor(self, pool_name: str, actor_index: int):
        """
        Recreate a failed actor with exponential backoff and improved error handling.
        
        Args:
            pool_name: Name of the actor pool
            actor_index: Index of the actor in the pool
            
        Returns:
            bool: True if actor was successfully recreated, False otherwise
        """
        if pool_name not in self.actor_pools:
            logger.error(f"Cannot recreate actor in non-existent pool: {pool_name}")
            return False
            
        pool_info = self.actor_pools[pool_name]
        configs = pool_info["configs"]
        
        if actor_index >= len(configs):
            logger.error(f"Invalid actor index {actor_index} for pool {pool_name}")
            return False
            
        # Get actor configuration
        config = configs[actor_index]
        actor_id = config["id"]
        
        # Check if we've hit the restart limit
        if config["restart_count"] >= pool_info["max_restarts"]:
            logger.error(f"Actor {actor_id} has reached restart limit ({config['restart_count']}), not recreating")
            config["status"] = "failed_permanent"
            return False
        
        # Update failure statistics
        config["restart_count"] += 1
        self.actor_failures[pool_name] = self.actor_failures.get(pool_name, 0) + 1
        self.recovery_attempts[pool_name] = self.recovery_attempts.get(pool_name, 0) + 1
        
        # Get creation parameters
        actor_class = pool_info["actor_class"]
        
        try:
            # Kill the old actor if it exists (ignoring errors)
            if config["actor"]:
                try:
                    ray.kill(config["actor"])
                    logger.info(f"Killed actor {actor_id}")
                    
                    # Important: Wait longer for GPU actors to free memory
                    if "num_gpus" in config["options"] and config["options"]["num_gpus"] > 0:
                        logger.info(f"Waiting longer for GPU memory to be released")
                        time.sleep(5.0)  # Wait 5 seconds for GPU memory to be released
                except Exception as e:
                    logger.debug(f"Error killing actor {actor_id}: {e}")
            
            # Set status to recreating
            config["status"] = "recreating"
            
            # Calculate exponential backoff wait time
            # 0.5s, 1s, 2s, 4s, 8s, 16s, max 30s
            backoff = min(30.0, 0.5 * (2 ** (config["restart_count"] - 1)))
            logger.info(f"Waiting {backoff:.1f}s before recreating actor {actor_id} (attempt {config['restart_count']})")
            time.sleep(backoff)
            
            # Create a new actor with extended options for reliability
            creation_options = dict(config["options"])  # Copy the original options
            
            # For GPU actors, add special handling
            if "num_gpus" in creation_options and creation_options["num_gpus"] > 0:
                # Adjust resources slightly for better scheduling
                # This helps Ray find a clean GPU slot
                adjusted_gpu = max(0.1, creation_options["num_gpus"] * 0.95)
                creation_options["num_gpus"] = adjusted_gpu
                logger.info(f"Adjusted GPU allocation to {adjusted_gpu} for better scheduling")
            
            # Create actor with resource specifications
            actor_with_resources = actor_class.options(**creation_options)
            
            # Create with extended timeout for initialization
            new_actor = None
            creation_timeout = 30.0  # 30 second timeout for actor creation
            creation_start = time.time()
            
            # Create the actor and wait for it to be ready
            new_actor = actor_with_resources.remote(**config["args"])
            
            # Verify actor is alive with simple ping
            try:
                # Try a simple method call to verify the actor is responsive
                # Most actors support __ray_terminate__, so use that as a ping target
                ray.get(new_actor.__ray_ping__.remote(), timeout=5.0)
                logger.info(f"Successfully verified actor {actor_id} is responsive")
            except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError) as e:
                # Actor failed immediately after creation
                logger.error(f"Actor {actor_id} created but failed ping test: {e}")
                if new_actor:
                    try:
                        ray.kill(new_actor)
                    except:
                        pass
                raise ValueError(f"Actor created but failed verification")
            
            # For actors with health check methods, wait for ready state
            if "health_check_method" in pool_info and pool_info["health_check_method"]:
                health_method = pool_info["health_check_method"]
                health_timeout = 10.0  # 10 second timeout for initial health check
                
                try:
                    # Call the health check method and wait for result
                    health_result = ray.get(getattr(new_actor, health_method).remote(), timeout=health_timeout)
                    
                    # Check if result indicates the actor is healthy
                    is_healthy = False
                    if isinstance(health_result, bool):
                        is_healthy = health_result
                    elif isinstance(health_result, dict) and "status" in health_result:
                        status = health_result["status"]
                        # Consider "initializing" as healthy for new actors
                        is_healthy = status in ["ready", "healthy", "ok", "initializing"]
                    else:
                        is_healthy = health_result is not None
                    
                    if not is_healthy:
                        logger.error(f"Actor {actor_id} failed initial health check: {health_result}")
                        if new_actor:
                            try:
                                ray.kill(new_actor)
                            except:
                                pass
                        raise ValueError(f"Actor failed initial health check")
                        
                    logger.info(f"Actor {actor_id} passed initial health check")
                    
                except (ray.exceptions.GetTimeoutError, ray.exceptions.RayActorError) as e:
                    logger.error(f"Health check failed for newly created actor {actor_id}: {e}")
                    if new_actor:
                        try:
                            ray.kill(new_actor)
                        except:
                            pass
                    raise ValueError(f"Actor health check failed: {e}")
            
            if new_actor:
                # Replace in the configs and actors list
                config["actor"] = new_actor
                config["status"] = "running"
                config["last_health_check"] = datetime.now()
                
                # Update actors list
                pool_info["actors"] = [
                    c["actor"] for c in pool_info["configs"]
                    if c["actor"] is not None and c["status"] == "running"
                ]
                
                logger.info(f"Successfully recreated actor {actor_id} in pool {pool_name}")
                return True
            else:
                logger.error(f"Failed to recreate actor {actor_id}")
                config["status"] = "failed"
                return False
                
        except Exception as e:
            logger.error(f"Error recreating actor {actor_id}: {e}")
            config["status"] = "failed"
            
            # If this was an early failure attempt, try one more time with more delay
            if config["restart_count"] <= 2:
                logger.info(f"Will retry recreation once more after extended delay")
                time.sleep(10.0)  # longer delay for retry
                return self._recreate_actor(pool_name, actor_index)  # One recursive retry
                
            return False
    
    def get_actor(self, pool_name: str, strategy: str = "round_robin") -> Any:
        """
        Get an actor from a pool using the specified selection strategy.
        
        Args:
            pool_name: Name of the actor pool
            strategy: Actor selection strategy ('round_robin', 'least_loaded', 'random')
            
        Returns:
            Actor handle or None if pool doesn't exist or has no active actors
        """
        if pool_name not in self.actor_pools:
            logger.error(f"Actor pool not found: {pool_name}")
            return None
            
        pool_info = self.actor_pools[pool_name]
        actors = pool_info["actors"]
        
        if not actors:
            logger.error(f"No actors available in pool: {pool_name}")
            return None
            
        # Simple round-robin selection
        if "counter" not in pool_info:
            pool_info["counter"] = 0
            
        index = pool_info["counter"] % len(actors)
        pool_info["counter"] += 1
        
        return actors[index]
    
    def shutdown_pool(self, pool_name: str):
        """Shutdown all actors in a pool"""
        if pool_name not in self.actor_pools:
            logger.warning(f"Cannot shutdown non-existent pool: {pool_name}")
            return
            
        pool_info = self.actor_pools[pool_name]
        
        # Mark pool as inactive to stop health monitoring
        pool_info["active"] = False
        
        # Kill each actor
        for config in pool_info["configs"]:
            actor = config["actor"]
            actor_id = config["id"]
            
            if actor and config["status"] == "running":
                try:
                    ray.kill(actor)
                    logger.debug(f"Killed actor {actor_id}")
                except Exception as e:
                    logger.warning(f"Error killing actor {actor_id}: {e}")
                    
                # Mark as terminated
                config["status"] = "terminated"
        
        # Clear actors list
        pool_info["actors"] = []
        
        logger.info(f"Shutdown actor pool: {pool_name}")
    
    def get_pool_status(self, pool_name: str) -> Dict:
        """Get detailed status of an actor pool"""
        if pool_name not in self.actor_pools:
            return {"error": f"Pool {pool_name} not found"}
            
        pool_info = self.actor_pools[pool_name]
        
        # Count actors by status
        status_counts = {}
        for config in pool_info["configs"]:
            status = config["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            
        # Get health statistics
        last_check = self.last_health_check.get(pool_name, datetime.min)
        seconds_since_check = (datetime.now() - last_check).total_seconds()
        
        return {
            "name": pool_name,
            "total_actors": len(pool_info["configs"]),
            "active_actors": len(pool_info["actors"]),
            "status_counts": status_counts,
            "recovery_attempts": self.recovery_attempts.get(pool_name, 0),
            "actor_failures": self.actor_failures.get(pool_name, 0),
            "seconds_since_last_health_check": seconds_since_check,
            "active": pool_info["active"]
        }
    
    def shutdown(self):
        """Shutdown all actor pools and Ray"""
        logger.info("Shutting down RayActorManager...")
        
        # Set shutdown flag for health monitoring threads
        self.shutdown_flag.set()
        
        # Shutdown all actor pools
        for pool_name in list(self.actor_pools.keys()):
            self.shutdown_pool(pool_name)
        
        # Wait for health threads to exit
        for thread_name, thread in list(self.health_threads.items()):
            if thread.is_alive():
                thread.join(timeout=2.0)
                logger.debug(f"Health thread for {thread_name} stopped")
        
        # Shutdown Ray
        if ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shutdown complete")
            except Exception as e:
                logger.error(f"Error during Ray shutdown: {e}")
        
        self.initialized = False

# Health check functions for common actors

def inference_server_health_check(actor):
    """Health check function for enhanced batch inference server"""
    try:
        health_status = ray.get(actor.get_health_status.remote(), timeout=5.0)
        return (health_status["status"] == "ready" and 
                health_status["setup_complete"] and
                "error" not in health_status["status"].lower())
    except Exception:
        return False

def trainer_health_check(actor):
    """Health check function for trainer actors"""
    try:
        stats = ray.get(actor.get_stats.remote(), timeout=5.0)
        # Check if stats contains expected keys
        return isinstance(stats, dict) and "buffer_size" in stats
    except Exception:
        return False

def experience_collector_health_check(actor):
    """Health check function for experience collector actors"""
    try:
        stats = ray.get(actor.get_statistics.remote(), timeout=5.0)
        # Check if stats contains expected keys
        return isinstance(stats, dict) and "games_completed" in stats
    except Exception:
        return False

# Helper function to create a preconfigured manager
def create_manager_with_inference_server(use_gpu=True, batch_wait=0.001, 
                                      cache_size=20000, max_batch_size=256,
                                      cpu_limit=None, gpu_fraction=1.0, 
                                      use_mixed_precision=True, verbose=False):
    """
    Create a RayActorManager with a preconfigured inference server.
    
    Args:
        use_gpu: Whether to use GPU
        batch_wait: Initial batch wait time
        cache_size: Cache size for inference server
        max_batch_size: Maximum batch size for inference server
        cpu_limit: CPU limit for Ray
        gpu_fraction: GPU fraction for inference server
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        tuple: (manager, inference_server)
    """
    # Import necessary modules
    from inference.enhanced_batch_inference_server import EnhancedBatchInferenceServer
    
    # Create manager
    manager = RayActorManager(
        use_gpu=use_gpu,
        cpu_limit=cpu_limit
    )
    
    # Create inference server pool
    inference_servers = manager.create_actor_pool(
        actor_class=EnhancedBatchInferenceServer,
        pool_name="inference_servers",
        num_actors=1,
        actor_args={
            "initial_batch_wait": batch_wait,
            "cache_size": cache_size,
            "max_batch_size": max_batch_size,
            "adaptive_batching": True,
            "mixed_precision": use_mixed_precision,
            "verbose": verbose
        },
        per_actor_cpus=1.0,
        per_actor_gpus=gpu_fraction,
        health_check_method="get_health_status",
        health_check_interval=30.0
    )
    
    # Get primary inference server
    inference_server = inference_servers[0] if inference_servers else None
    
    # Return both the manager and server
    return manager, inference_server