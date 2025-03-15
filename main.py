# main.py
"""
Main entry point for AlphaZero-style training.
Provides a command-line interface for training, evaluation, and hyperparameter tuning.
"""
import ray
import argparse
import os
import sys
from train.self_play import SelfPlayManager
from utils.state_utils import TicTacToeState
from inference.batch_inference_server import BatchInferenceServer
from mcts import batched_mcts_search
import torch
import numpy as np

def train(args):
    """Run training with the specified parameters"""
    # Handle mutually exclusive arguments
    if args.enable_parallel_mcts:
        args.disable_parallel_mcts = False
        
    # Create manager instance with appropriate MCTS configuration
    manager = SelfPlayManager(
        use_parallel_mcts=not args.disable_parallel_mcts,
        enable_time_based_search=args.time_based_search,
        max_search_time=args.search_time,
        verbose=args.verbose,
        max_workers=args.max_workers
    )
    
    # Print configuration
    search_method = "time-based" if args.time_based_search else ("parallel" if not args.disable_parallel_mcts else "batched")
    print(f"Training with {search_method} MCTS")
    
    try:
        # Try to load checkpoint if specified
        if args.checkpoint:
            manager.load_checkpoint(args.checkpoint)
        else:
            # Try to load latest checkpoint if it exists
            try:
                manager.load_checkpoint("model_latest")
            except:
                print("No checkpoint found, starting fresh training")
            
        # Run training
        manager.train(num_games=args.games)
        
        # Save final checkpoint
        manager.save_checkpoint("model_latest")
    finally:
        # Ensure clean shutdown
        ray.shutdown()

def evaluate(args):
    """Evaluate a trained model by playing against itself"""
    # Initialize Ray with minimal resources
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create inference server
    inference_actor = BatchInferenceServer.remote(
        batch_wait=0.001,
        cache_size=1000,
        max_batch_size=64
    )
    
    # Load model from checkpoint
    checkpoint_path = os.path.join("checkpoints", f"{args.checkpoint}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model weights into the inference server
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        state_dict[key] = value.cpu().numpy()
    
    # Update server
    ray.get(inference_actor.update_model.remote(state_dict))
    
    # Play games
    print(f"Evaluating model: {args.checkpoint}")
    print(f"Playing {args.games} games with {args.simulations} simulations per move")
    
    win_counts = {1: 0, -1: 0, 0: 0}  # Player 1, Player -1, Draw
    move_counts = []
    
    for game_idx in range(args.games):
        state = TicTacToeState()
        moves = 0
        
        # Play until game is finished
        while not state.is_terminal():
            # Use a lower temperature for evaluation to get stronger play
            temperature = 0.5 if moves < 4 else 0.1
            
            # Perform search
            root = batched_mcts_search(
                state,
                inference_actor,
                args.simulations,
                batch_size=64,
                verbose=args.verbose
            )
            
            # Select action (more deterministic for evaluation)
            visits = np.array([child.visits for child in root.children])
            actions = [child.action for child in root.children]
            
            if temperature == 0 or np.random.random() < 0.9:  # 90% of the time, choose best move
                best_idx = np.argmax(visits)
                action = actions[best_idx]
            else:
                # Apply temperature
                probs = visits / np.sum(visits)
                action = np.random.choice(actions, p=probs)
            
            # Apply action
            state = state.apply_action(action)
            moves += 1
            
            # Print board occasionally
            if args.verbose and moves % 2 == 0:
                print(f"\nGame {game_idx+1}, Move {moves}:")
                print(state)
        
        # Game finished
        outcome = state.get_winner()
        win_counts[outcome] = win_counts.get(outcome, 0) + 1
        move_counts.append(moves)
        
        print(f"Game {game_idx+1}: {moves} moves, outcome={outcome}")
        print(state)
    
    # Print statistics
    total_games = sum(win_counts.values())
    avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0
    print("\nEvaluation Results:")
    print(f"  Games played: {total_games}")
    print(f"  Average moves per game: {avg_moves:.1f}")
    print(f"  Player 1 wins: {win_counts.get(1, 0)} ({win_counts.get(1, 0)/total_games*100:.1f}%)")
    print(f"  Player 2 wins: {win_counts.get(-1, 0)} ({win_counts.get(-1, 0)/total_games*100:.1f}%)")
    print(f"  Draws: {win_counts.get(0, 0)} ({win_counts.get(0, 0)/total_games*100:.1f}%)")
    
    # Clean up
    ray.shutdown()

def tune_hyperparameters(args):
    """Run hyperparameter optimization"""
    try:
        # Import hyperparameter tuning module dynamically
        from hyperparameter_tuning import run_hyperparameter_optimization
    except ImportError:
        print("Error: hyperparameter_tuning.py module not found.")
        print("Please ensure it exists in the current directory.")
        return
    
    # Convert args to the format expected by hyperparameter_tuning
    tune_args = argparse.Namespace(
        games=args.games,
        num_trials=args.trials,
        output_dir=args.output_dir,
        cpus=args.cpus,
        gpus=args.gpus,
        memory=args.memory,
        object_store_memory=args.object_store_memory,
        cpus_per_trial=args.cpus_per_trial,
        gpus_per_trial=args.gpus_per_trial,
        grace_period=args.grace_period,
        brackets=args.brackets,
        search_algo=args.search_algo,
        scheduler=args.scheduler,
        concurrent_trials=args.concurrent_trials,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        resume=args.resume
    )
    
    # Run hyperparameter optimization
    best_config, analysis = run_hyperparameter_optimization(tune_args)
    
    # Print additional information if verbose
    if args.verbose:
        print("\nAnalysis Summary:")
        print(f"Total trials: {len(analysis.trials)}")
        print(f"Completed trials: {sum(1 for t in analysis.trials.values() if t.status == 'TERMINATED')}")
        print(f"Failed trials: {sum(1 for t in analysis.trials.values() if t.status == 'ERROR')}")
    
    return best_config, analysis

def analyze_results(args):
    """Analyze hyperparameter tuning results"""
    try:
        # Import hyperparameter analysis module dynamically
        from hyperparameter_analysis import (
            load_results, plot_learning_curves, plot_parameter_importance,
            plot_pairwise_relationships, plot_parallel_coordinates,
            print_best_configurations, apply_config_to_file
        )
    except ImportError:
        print("Error: hyperparameter_analysis.py module not found.")
        print("Please ensure it exists in the current directory.")
        return
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.experiment_dir}...")
    analysis = load_results(args.experiment_dir)
    
    # Determine which analyses to run
    run_all = args.all
    
    # Perform requested analyses
    if args.learning_curves or run_all:
        print("Plotting learning curves...")
        save_path = os.path.join(args.output_dir, 'learning_curves.png') if args.output_dir else None
        plot_learning_curves(analysis, args.metric, args.mode, args.top_n, save_path)
    
    if args.parameter_importance or run_all:
        print("Plotting parameter importance...")
        save_path = os.path.join(args.output_dir, 'parameter_importance.png') if args.output_dir else None
        plot_parameter_importance(analysis, args.metric, args.mode, save_path)
    
    if args.pairwise or run_all:
        print("Plotting pairwise relationships...")
        save_path = os.path.join(args.output_dir, 'pairwise_relationships.png') if args.output_dir else None
        plot_pairwise_relationships(analysis, args.metric, None, args.top_n * 10, save_path)
        
    if args.parallel_coords or run_all:
        print("Plotting parallel coordinates...")
        save_path = os.path.join(args.output_dir, 'parallel_coordinates.png') if args.output_dir else None
        plot_parallel_coordinates(analysis, args.metric, args.mode, args.top_n, save_path)
    
    if args.print_best or run_all or not any([args.learning_curves, args.parameter_importance, 
                                             args.pairwise, args.parallel_coords, args.apply_best]):
        print_best_configurations(analysis, args.metric, args.mode, args.top_n)
    
    if args.apply_best or run_all:
        # Get best configuration
        best_config = analysis.get_best_config(args.metric, mode=args.mode)
        output_file = "tuned_config.py"
        if args.output_dir:
            output_file = os.path.join(args.output_dir, output_file)
        apply_config_to_file(best_config, output_file)

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='AlphaZero-style training, evaluation, and hyperparameter tuning')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--games', type=int, default=200, help='Number of games to play')
    train_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    train_parser.add_argument('--disable-parallel-mcts', action='store_true', default=True, 
                             help='Disable parallel MCTS (default: True for stability)')
    train_parser.add_argument('--enable-parallel-mcts', action='store_true',
                             help='Enable task-based parallel MCTS (more stable than actor-based)')
    train_parser.add_argument('--max-workers', type=int, default=2, 
                             help='Maximum number of parallel workers (lower is more stable)')
    train_parser.add_argument('--time-based-search', action='store_true', help='Use time-based search instead of fixed simulations')
    train_parser.add_argument('--search-time', type=float, default=1.0, help='Time limit for search in seconds')
    train_parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
    eval_parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to evaluate')
    eval_parser.add_argument('--simulations', type=int, default=800, help='Simulations per move')
    eval_parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser('tune', help='Optimize hyperparameters')
    
    # Trial parameters
    tune_parser.add_argument('--games', type=int, default=50, help='Number of games for each trial')
    tune_parser.add_argument('--trials', type=int, default=10, help='Number of trials to run')
    tune_parser.add_argument('--concurrent-trials', type=int, default=2, help='Number of concurrent trials')
    tune_parser.add_argument('--grace-period', type=int, default=10, help='Minimum games before early stopping')
    tune_parser.add_argument('--brackets', type=int, default=3, help='Number of brackets for ASHA scheduler')
    
    # Ray resources
    tune_parser.add_argument('--cpus', type=int, default=None, help='Total number of CPUs to use')
    tune_parser.add_argument('--gpus', type=int, default=None, help='Total number of GPUs to use')
    tune_parser.add_argument('--cpus-per-trial', type=int, default=1, help='CPUs to allocate per trial')
    tune_parser.add_argument('--gpus-per-trial', type=float, default=0.5, help='GPUs to allocate per trial')
    tune_parser.add_argument('--memory', type=float, default=None, help='Memory limit in GB')
    tune_parser.add_argument('--object-store-memory', type=float, default=None, help='Object store memory in GB')
    
    # Search configuration
    tune_parser.add_argument('--scheduler', type=str, default='asha', 
                            choices=['asha', 'pbt', 'none'], help='Scheduler to use')
    tune_parser.add_argument('--search-algo', type=str, default='bayesopt', 
                            choices=['random', 'bayesopt', 'hyperopt'], help='Search algorithm')
    
    # Output options
    tune_parser.add_argument('--output-dir', type=str, default='./ray_results', help='Directory for results')
    tune_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    tune_parser.add_argument('--fail-fast', action='store_true', help='Stop if a trial fails')
    tune_parser.add_argument('--resume', action='store_true', help='Resume previous tuning session')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze hyperparameter tuning results')
    analyze_parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    analyze_parser.add_argument('--metric', type=str, default='loss', help='Metric to analyze')
    analyze_parser.add_argument('--mode', type=str, default='min', choices=['min', 'max'], 
                               help='Whether to minimize or maximize the metric')
    analyze_parser.add_argument('--output-dir', type=str, default=None, help='Directory for analysis output')
    analyze_parser.add_argument('--top-n', type=int, default=5, help='Number of top configurations to analyze')
    
    # Analysis options
    analyze_parser.add_argument('--all', action='store_true', help='Run all analyses')
    analyze_parser.add_argument('--learning-curves', action='store_true', help='Plot learning curves')
    analyze_parser.add_argument('--parameter-importance', action='store_true', help='Plot parameter importance')
    analyze_parser.add_argument('--pairwise', action='store_true', help='Plot pairwise relationships')
    analyze_parser.add_argument('--parallel-coords', action='store_true', help='Plot parallel coordinates')
    analyze_parser.add_argument('--print-best', action='store_true', help='Print best configurations')
    analyze_parser.add_argument('--apply-best', action='store_true', help='Apply best config to a file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    elif args.command == 'tune':
        tune_hyperparameters(args)
    elif args.command == 'analyze':
        analyze_results(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()