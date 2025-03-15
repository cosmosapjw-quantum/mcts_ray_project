# hyperparameter_tuning.py
"""
Hyperparameter optimization for AlphaZero-style training using Ray Tune.
This module provides functionality to search for optimal hyperparameters
for the MCTS algorithm and neural network training process.
"""

import os
import ray
import time
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
import torch

# Import project components
from train.self_play import SelfPlayManager
from utils.state_utils import TicTacToeState
from inference.batch_inference_server import BatchInferenceServer
from config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_SIMULATIONS,
    MCTS_BATCH_SIZE, EXPLORATION_WEIGHT, TEMPERATURE_SCHEDULE
)

# Define the evaluation function for a specific hyperparameter configuration
def evaluate_config(config, checkpoint_dir=None):
    """
    Evaluate a specific hyperparameter configuration by training a model
    and measuring its performance.
    
    Args:
        config: Hyperparameter configuration to evaluate
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        dict: Evaluation metrics
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create inference server with config parameters
    inference_server = BatchInferenceServer.remote(
        batch_wait=config.get("INFERENCE_SERVER_BATCH_WAIT", 0.001),
        cache_size=config.get("INFERENCE_SERVER_CACHE_SIZE", 1000),
        max_batch_size=config.get("INFERENCE_SERVER_MAX_BATCH_SIZE", 64)
    )
    
    # Create manager with the configuration
    manager = SelfPlayManager(
        use_parallel_mcts=config.get("USE_PARALLEL_MCTS", False),
        enable_time_based_search=config.get("TIME_BASED_SEARCH", False),
        max_search_time=config.get("MAX_SEARCH_TIME", 1.0),
        verbose=False,
        max_workers=config.get("MAX_WORKERS", 2)
    )
    
    # Apply hyperparameters to the manager's trainer
    if "LEARNING_RATE" in config:
        for param_group in manager.optimizer.param_groups:
            param_group['lr'] = config["LEARNING_RATE"]
    
    if "WEIGHT_DECAY" in config:
        for param_group in manager.optimizer.param_groups:
            param_group['weight_decay'] = config["WEIGHT_DECAY"]
    
    # Define temperature schedule if parameters are provided
    if 'TEMP_INIT' in config and 'TEMP_FINAL' in config and 'TEMP_DECAY_MOVE' in config:
        temp_schedule = {
            0: config['TEMP_INIT'],
            config['TEMP_DECAY_MOVE']: config['TEMP_FINAL'],
        }
        manager.temperature_schedule = temp_schedule
    
    # Restore from checkpoint if provided
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        if os.path.exists(checkpoint_path):
            manager.load_checkpoint(checkpoint_path)
    
    # Training loop
    num_games = config.get("GAMES_PER_TRIAL", 50)
    checkpoint_freq = config.get("CHECKPOINT_FREQ", 5)
    
    initial_games = manager.game_count
    total_games = initial_games + num_games
    
    try:
        while manager.game_count < total_games:
            # Generate a game
            outcome, moves = manager.generate_game(manager.game_count)
            manager.game_count += 1
            
            # Train on the replay buffer
            loss = manager.trainer.train_batch()
            
            # Calculate current win rates
            total_played = sum(manager.win_rates.values())
            if total_played > 0:
                p1_win_rate = manager.win_rates.get(1, 0) / total_played * 100
                p2_win_rate = manager.win_rates.get(-1, 0) / total_played * 100
                draw_rate = manager.win_rates.get(0, 0) / total_played * 100
            else:
                p1_win_rate = p2_win_rate = draw_rate = 0
            
            # Update inference server periodically
            if manager.game_count % 5 == 0:
                manager.update_inference_server()
            
            # Report metrics to Ray Tune
            tune.report(
                loss=loss if loss is not None else float('inf'),
                games_completed=manager.game_count - initial_games,
                total_games=manager.game_count,
                p1_win_rate=p1_win_rate,
                p2_win_rate=p2_win_rate,
                draw_rate=draw_rate,
                buffer_size=len(manager.replay_buffer)
            )
            
            # Checkpoint periodically
            if checkpoint_dir and manager.game_count % checkpoint_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
                manager.save_checkpoint(checkpoint_path)
    
    except Exception as e:
        print(f"Error during training: {e}")
        # Return poor performance metrics so this config is deprioritized
        tune.report(
            loss=float('inf'),
            games_completed=0,
            total_games=0,
            p1_win_rate=0,
            p2_win_rate=0,
            draw_rate=0,
            buffer_size=0,
            error=str(e)
        )
    finally:
        # Clean up resources
        manager.trainer.close()
        ray.kill(inference_server)

def get_search_space():
    """Define the hyperparameter search space"""
    return {
        # Learning parameters
        "LEARNING_RATE": tune.loguniform(1e-4, 1e-2),
        "WEIGHT_DECAY": tune.loguniform(1e-5, 1e-3),
        "BATCH_SIZE": tune.choice([256, 512, 1024, 2048]),
        "TRAINING_EPOCHS": tune.choice([5, 10, 20]),
        
        # MCTS parameters
        "NUM_SIMULATIONS": tune.choice([200, 400, 800, 1600]),
        "MCTS_BATCH_SIZE": tune.choice([16, 32, 64, 128]),
        "EXPLORATION_WEIGHT": tune.uniform(0.8, 2.0),
        
        # Temperature parameters (to build TEMPERATURE_SCHEDULE)
        "TEMP_INIT": tune.uniform(0.5, 1.5),
        "TEMP_FINAL": tune.uniform(0.1, 0.5),
        "TEMP_DECAY_MOVE": tune.choice([10, 15, 20, 30]),
        
        # Inference server parameters
        "INFERENCE_SERVER_BATCH_WAIT": tune.loguniform(1e-4, 1e-2),
        "INFERENCE_SERVER_CACHE_SIZE": tune.choice([1000, 5000, 10000, 20000]),
        "INFERENCE_SERVER_MAX_BATCH_SIZE": tune.choice([64, 128, 256, 512]),
        
        # Search strategy parameters
        "USE_PARALLEL_MCTS": tune.choice([True, False]),
        "TIME_BASED_SEARCH": tune.choice([True, False]),
        "MAX_SEARCH_TIME": tune.uniform(0.5, 3.0),
        "MAX_WORKERS": tune.choice([1, 2, 4, 8]),
        
        # Constants for the trial
        "GAMES_PER_TRIAL": tune.choice([30, 50, 100]),
        "CHECKPOINT_FREQ": 5,
    }

def run_hyperparameter_optimization(args):
    """
    Run hyperparameter optimization using Ray Tune
    
    Args:
        args: Command-line arguments
    """
    # Initialize Ray with proper resources
    if ray.is_initialized():
        ray.shutdown()
        
    ray.init(
        num_cpus=args.cpus,
        num_gpus=args.gpus,
        include_dashboard=True,
        _memory=args.memory * 1024 * 1024 * 1024 if args.memory else None,
        object_store_memory=args.object_store_memory * 1024 * 1024 * 1024 if args.object_store_memory else None
    )
    
    # Define hyperparameter search space
    config = get_search_space()
    
    # Choose scheduler based on args
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            time_attr="games_completed",
            max_t=args.games,
            grace_period=args.grace_period,
            reduction_factor=2,
            brackets=args.brackets
        )
    elif args.scheduler == "pbt":
        scheduler = PopulationBasedTraining(
            time_attr="games_completed",
            perturbation_interval=args.games // 10,
            hyperparam_mutations={
                "LEARNING_RATE": lambda: np.random.uniform(1e-5, 1e-2),
                "EXPLORATION_WEIGHT": lambda: np.random.uniform(0.5, 2.5),
                "TEMP_INIT": lambda: np.random.uniform(0.3, 2.0),
                "TEMP_FINAL": lambda: np.random.uniform(0.05, 0.7),
            }
        )
    else:
        # Default: no scheduler
        scheduler = None
    
    # Choose search algorithm
    if args.search_algo == "bayesopt":
        search_alg = BayesOptSearch(
            metric="loss",
            mode="min",
            utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}
        )
    elif args.search_algo == "hyperopt":
        search_alg = HyperOptSearch(
            metric="loss",
            mode="min",
            n_initial_points=5
        )
    else:  # Default to random search
        search_alg = None
    
    # Apply concurrency limit if using a search algorithm
    if search_alg is not None:
        search_alg = ConcurrencyLimiter(
            search_alg,
            max_concurrent=args.concurrent_trials
        )
    
    # Define progress reporter
    reporter = CLIReporter(
        metric_columns=[
            "loss", "games_completed", "p1_win_rate", 
            "p2_win_rate", "draw_rate", "buffer_size"
        ],
        parameter_columns=[
            "LEARNING_RATE", "BATCH_SIZE", "NUM_SIMULATIONS", 
            "EXPLORATION_WEIGHT", "TEMP_INIT", "MAX_WORKERS"
        ]
    )
    
    # Run hyperparameter search
    analysis = tune.run(
        evaluate_config,
        config=config,
        num_samples=args.num_trials,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        resources_per_trial={
            "cpu": args.cpus_per_trial,
            "gpu": args.gpus_per_trial
        },
        local_dir=args.output_dir,
        checkpoint_at_end=True,
        keep_checkpoints_num=2,
        checkpoint_score_attr="min-loss",
        stop={
            "games_completed": args.games
        },
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        reuse_actors=False,
        resume=args.resume
    )
    
    # Print best hyperparameters
    best_config = analysis.get_best_config(metric="loss", mode="min")
    print("\nBest hyperparameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters to a file
    best_config_path = os.path.join(args.output_dir, "best_hyperparameters.txt")
    with open(best_config_path, "w") as f:
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nBest hyperparameters saved to: {best_config_path}")
    
    # Create optimized config file
    optimized_config_path = os.path.join(args.output_dir, "optimized_config.py")
    with open(optimized_config_path, "w") as f:
        f.write("# Auto-generated optimized configuration from hyperparameter tuning\n")
        f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for key, value in best_config.items():
            if isinstance(value, str):
                f.write(f"{key} = '{value}'\n")
            else:
                f.write(f"{key} = {value}\n")
        
        # Handle temperature schedule
        if 'TEMP_INIT' in best_config and 'TEMP_FINAL' in best_config and 'TEMP_DECAY_MOVE' in best_config:
            f.write("\n# Temperature schedule\n")
            f.write("TEMPERATURE_SCHEDULE = {\n")
            f.write(f"    0: {best_config['TEMP_INIT']},\n")
            f.write(f"    {best_config['TEMP_DECAY_MOVE']}: {best_config['TEMP_FINAL']},\n")
            f.write("}\n")
    
    print(f"Optimized config file created at: {optimized_config_path}")
    
    return best_config, analysis