# main_optimized.py
"""
Main entry point for optimized MCTS training with AlphaZero-style learning.
This script provides a command-line interface for running the different
optimized components.
"""

import os
import argparse
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mcts_training.log')
    ]
)
logger = logging.getLogger("MCTS")

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Optimized MCTS training with AlphaZero-style learning'
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Standard training command
    train_parser = subparsers.add_parser('train', help='Run standard training')
    train_parser.add_argument('--games', type=int, default=500, help='Number of games to play')
    train_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    train_parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    train_parser.add_argument('--batch-size', type=int, default=256, help='Batch size for MCTS')
    train_parser.add_argument('--collectors', type=int, default=10, help='Number of leaf collectors')
    train_parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    # Async training command
    async_parser = subparsers.add_parser('async', help='Run asynchronous training')
    async_parser.add_argument('--hours', type=float, default=None, help='Training duration in hours (None = indefinite)')
    async_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    async_parser.add_argument('--collectors', type=int, default=8, help='Number of experience collectors')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to evaluate')
    eval_parser.add_argument('--games', type=int, default=20, help='Number of games to play')
    eval_parser.add_argument('--simulations', type=int, default=1200, help='MCTS simulations per move')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark MCTS performance')
    bench_parser.add_argument('--simulations', type=int, default=800, help='MCTS simulations per move')
    bench_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for MCTS')
    bench_parser.add_argument('--collectors', type=int, default=2, help='Number of leaf collectors')
    bench_parser.add_argument('--games', type=int, default=5, help='Number of games to benchmark')
    bench_parser.add_argument('--trials', type=int, default=3, help='Number of benchmark trials')
    
    # System configuration
    parser.add_argument('--cpu-limit', type=int, default=None, help='Maximum CPUs to use (None = auto)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process system configuration
    use_gpu = not args.no_gpu
    use_mixed_precision = not args.no_mixed_precision and use_gpu
    
    if args.command == 'train':
        # Run standard training with enhanced components
        run_standard_training(
            games=args.games,
            checkpoint=args.checkpoint,
            simulations=args.simulations,
            batch_size=args.batch_size,
            num_collectors=args.collectors,
            verbose=args.verbose,
            use_gpu=use_gpu,
            use_mixed_precision=use_mixed_precision,
            cpu_limit=args.cpu_limit
        )
    elif args.command == 'async':
        # Run asynchronous training
        run_async_training(
            num_collectors=args.collectors,
            checkpoint=args.checkpoint,
            duration_hours=args.hours,
            use_gpu=use_gpu,
            cpu_limit=args.cpu_limit
        )
    elif args.command == 'eval':
        # Run evaluation
        run_evaluation(
            checkpoint=args.checkpoint,
            games=args.games,
            simulations=args.simulations,
            use_gpu=use_gpu,
            cpu_limit=args.cpu_limit
        )
    elif args.command == 'benchmark':
        # Run benchmark
        run_benchmark(
            simulations=args.simulations,
            batch_size=args.batch_size,
            num_collectors=args.collectors,
            games=args.games,
            trials=args.trials,
            use_gpu=use_gpu,
            cpu_limit=args.cpu_limit
        )
    else:
        parser.print_help()

def run_standard_training(games, checkpoint=None, simulations=800, batch_size=32, 
                         num_collectors=8, verbose=False, use_gpu=True, 
                         use_mixed_precision=True, cpu_limit=None):
    """
    Run standard training using the enhanced self-play manager.
    
    Args:
        games: Number of games to play
        checkpoint: Checkpoint to load
        simulations: MCTS simulations per move
        batch_size: Batch size for leaf evaluation
        num_collectors: Number of leaf collector threads
        verbose: Print detailed information
        use_gpu: Whether to use GPU
        use_mixed_precision: Whether to use mixed precision
        cpu_limit: Maximum CPUs to use
    """
    try:
        # Import enhanced self-play manager
        from train.enhanced_self_play import EnhancedSelfPlayManager
        
        # Create manager with enhanced components
        manager = EnhancedSelfPlayManager(
            # Search configuration
            num_simulations=simulations,
            num_collectors=num_collectors,
            batch_size=batch_size,
            exploration_weight=1.4,
            
            # Server configuration
            inference_server_batch_wait=0.001,
            inference_server_cache_size=20000,
            inference_server_max_batch_size=256,
            
            # System configuration
            use_gpu=use_gpu,
            cpu_limit=cpu_limit,
            use_mixed_precision=use_mixed_precision,
            verbose=verbose
        )
        
        # Load checkpoint if specified
        if checkpoint:
            manager.load_checkpoint(checkpoint)
        
        logger.info(f"Starting standard training with {games} games")
        logger.info(f"MCTS configuration: {simulations} simulations, {batch_size} batch size, {num_collectors} collectors")
        
        # Run training
        start_time = time.time()
        manager.train(num_games=games)
        
        # Log completion statistics
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed / 60:.1f} minutes")
        logger.info(f"Average speed: {games / (elapsed / 3600):.1f} games/hour")
        
        # Clean up
        manager.cleanup()
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_async_training(num_collectors=8, checkpoint=None, duration_hours=None,
                      use_gpu=True, cpu_limit=None):
    """
    Run asynchronous training with separate self-play and training processes.
    
    Args:
        num_collectors: Number of experience collectors
        checkpoint: Checkpoint to load
        duration_hours: Training duration in hours (None = indefinite)
        use_gpu: Whether to use GPU
        cpu_limit: Maximum CPUs to use
    """
    try:
        # Import asynchronous training pipeline
        from train.async_training import run_async_training
        
        logger.info(f"Starting asynchronous training with {num_collectors} collectors")
        if duration_hours:
            logger.info(f"Training will run for {duration_hours:.1f} hours")
        else:
            logger.info("Training will run indefinitely (Ctrl+C to stop)")
        
        # Initialize Ray with proper resource configuration
        import ray
        from utils.improved_ray_manager import RayActorManager, create_manager_with_inference_server
        
        # Create Ray manager for proper initialization
        ray_manager, inference_server = create_manager_with_inference_server(
            use_gpu=use_gpu, 
            batch_wait=0.001,              # Reduce wait time
            cache_size=20000,              # Increase cache
            max_batch_size=256,            # Increase batch size
            cpu_limit=cpu_limit,
            gpu_fraction=1.0,              # Use full GPU
            use_mixed_precision=True,
            verbose=False
        )
        
        try:
            # Run asynchronous training
            run_async_training(
                num_collectors=num_collectors,
                gpu=use_gpu,
                checkpoint=checkpoint,
                duration_hours=duration_hours
            )
        finally:
            # Clean up Ray
            ray_manager.shutdown()
        
    except Exception as e:
        logger.error(f"Error during async training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_evaluation(checkpoint, games=10, simulations=800, 
                  use_gpu=True, cpu_limit=None):
    """
    Evaluate a trained model by playing games.
    
    Args:
        checkpoint: Checkpoint to evaluate
        games: Number of games to play
        simulations: MCTS simulations per move
        use_gpu: Whether to use GPU
        cpu_limit: Maximum CPUs to use
    """
    try:
        # Import components
        from utils.improved_ray_manager import RayActorManager, create_manager_with_inference_server
        from inference.enhanced_batch_inference_server import EnhancedBatchInferenceServer
        from utils.optimized_state import OptimizedTicTacToeState
        from mcts.leaf_parallel_mcts import leaf_parallel_search
        
        import torch
        import numpy as np
        
        # Create Ray manager for proper initialization
        ray_manager, inference_server = create_manager_with_inference_server(
            use_gpu=use_gpu, 
            batch_wait=0.001,              # Reduce wait time
            cache_size=20000,              # Increase cache
            max_batch_size=256,            # Increase batch size
            cpu_limit=cpu_limit,
            gpu_fraction=1.0,              # Use full GPU
            use_mixed_precision=True,
            verbose=True
        )
        
        try:
            if not inference_server:
                raise RuntimeError("Failed to create inference server")
                
            # Load model
            checkpoint_path = os.path.join("checkpoints", f"{checkpoint}.pt")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
            # Load model on host
            device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
            
            from model import SmallResNet
            model = SmallResNet().to(device)
            
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Convert model weights for inference server
            state_dict = {}
            for key, value in model.state_dict().items():
                state_dict[key] = value.cpu().numpy()
                
            # Update inference server
            import ray
            ray.get(inference_server.update_model.remote(state_dict))
            
            logger.info(f"Evaluating model: {checkpoint}")
            logger.info(f"Playing {games} games with {simulations} simulations per move")
            
            # Create inference function
            def inference_function(state_or_batch):
                if isinstance(state_or_batch, list):
                    return ray.get(inference_server.batch_infer.remote(state_or_batch))
                else:
                    return ray.get(inference_server.infer.remote(state_or_batch))
            
            # Play evaluation games
            win_counts = {1: 0, -1: 0, 0: 0}  # Player 1, Player -1, Draw
            move_counts = []
            game_times = []
            
            for game_idx in range(games):
                game_start = time.time()
                
                state = OptimizedTicTacToeState()
                moves = 0
                
                # Play until game is finished
                while not state.is_terminal():
                    # Use lower temperature for evaluation
                    temperature = 0.5 if moves < 4 else 0.1
                    
                    # Perform search
                    root, _ = leaf_parallel_search(
                        root_state=state,
                        inference_fn=inference_function,
                        num_simulations=simulations,
                        num_collectors=8,
                        batch_size=32,
                        exploration_weight=1.4,
                        add_dirichlet_noise=False,  # No noise for evaluation
                        collect_stats=False
                    )
                    
                    # Select action (more deterministic for evaluation)
                    visits = np.array([child.visits for child in root.children])
                    actions = [child.action for child in root.children]
                    
                    if temperature == 0 or np.random.random() < 0.9:  # 90% best move
                        best_idx = np.argmax(visits)
                        action = actions[best_idx]
                    else:
                        # Apply temperature
                        from utils.mcts_utils import apply_temperature
                        action, _ = apply_temperature(visits, actions, temperature)
                    
                    # Apply action
                    state = state.apply_action(action)
                    moves += 1
                
                # Game finished
                game_time = time.time() - game_start
                outcome = state.get_winner()
                win_counts[outcome] = win_counts.get(outcome, 0) + 1
                move_counts.append(moves)
                game_times.append(game_time)
                
                logger.info(f"Game {game_idx+1}: {moves} moves, outcome={outcome}, time={game_time:.1f}s")
                logger.info(str(state))
            
            # Print statistics
            total_games = sum(win_counts.values())
            avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0
            avg_time = sum(game_times) / len(game_times) if game_times else 0
            
            logger.info("\nEvaluation Results:")
            logger.info(f"  Games played: {total_games}")
            logger.info(f"  Average moves per game: {avg_moves:.1f}")
            logger.info(f"  Average time per game: {avg_time:.1f}s")
            logger.info(f"  Player 1 wins: {win_counts.get(1, 0)} ({win_counts.get(1, 0)/total_games*100:.1f}%)")
            logger.info(f"  Player 2 wins: {win_counts.get(-1, 0)} ({win_counts.get(-1, 0)/total_games*100:.1f}%)")
            logger.info(f"  Draws: {win_counts.get(0, 0)} ({win_counts.get(0, 0)/total_games*100:.1f}%)")
            
        finally:
            # Clean up Ray
            ray_manager.shutdown()
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_benchmark(simulations=800, batch_size=32, num_collectors=8, 
                 games=5, trials=3, use_gpu=True, cpu_limit=None):
    """
    Benchmark MCTS performance with different configurations.
    
    Args:
        simulations: MCTS simulations per move
        batch_size: Batch size for leaf evaluation
        num_collectors: Number of leaf collector threads
        games: Number of games per trial
        trials: Number of benchmark trials
        use_gpu: Whether to use GPU
        cpu_limit: Maximum CPUs to use
    """
    try:
        # Import components
        from utils.improved_ray_manager import RayActorManager, create_manager_with_inference_server
        from inference.enhanced_batch_inference_server import EnhancedBatchInferenceServer
        from utils.optimized_state import OptimizedTicTacToeState
        from mcts.leaf_parallel_mcts import leaf_parallel_search
        
        import torch
        import numpy as np
        
        # Create Ray manager for proper initialization
        ray_manager, inference_server = create_manager_with_inference_server(
            use_gpu=use_gpu, 
            batch_wait=0.001,              # Reduce wait time
            cache_size=20000,              # Increase cache
            max_batch_size=256,            # Increase batch size
            cpu_limit=cpu_limit,
            gpu_fraction=1.0,              # Use full GPU
            use_mixed_precision=True,
            verbose=True
        )
        
        try:            
            if not inference_server:
                raise RuntimeError("Failed to create inference server")
                
            # Initialize model on inference server
            from model import SmallResNet
            model = SmallResNet()
            
            # Convert model weights for inference server
            state_dict = {}
            for key, value in model.state_dict().items():
                state_dict[key] = value.cpu().numpy()
                
            # Update inference server
            import ray
            ray.get(inference_server.update_model.remote(state_dict))
            
            logger.info(f"Running benchmark with {simulations} simulations, {batch_size} batch size, {num_collectors} collectors")
            
            # Create inference function
            def inference_function(state_or_batch):
                if isinstance(state_or_batch, list):
                    return ray.get(inference_server.batch_infer.remote(state_or_batch))
                else:
                    return ray.get(inference_server.infer.remote(state_or_batch))
            
            # Run benchmark trials
            moves_per_second = []
            nodes_per_move = []
            search_times = []
            
            for trial in range(trials):
                logger.info(f"\nTrial {trial+1}/{trials}:")
                
                game_moves = []
                game_times = []
                game_nodes = []
                
                for game_idx in range(games):
                    game_start = time.time()
                    
                    state = OptimizedTicTacToeState()
                    moves = 0
                    total_nodes = 0
                    total_search_time = 0
                    
                    # Play until game is finished or max moves reached
                    while not state.is_terminal() and moves < 20:  # Limit to 20 moves for benchmarking
                        # Use fixed temperature
                        temperature = 1.0
                        
                        # Perform search with detailed statistics
                        search_start = time.time()
                        root, stats = leaf_parallel_search(
                            root_state=state,
                            inference_fn=inference_function,
                            num_simulations=simulations,
                            num_collectors=num_collectors,
                            batch_size=batch_size,
                            exploration_weight=1.4,
                            add_dirichlet_noise=True,
                            collect_stats=True
                        )
                        search_time = time.time() - search_start
                        
                        # Track statistics
                        total_search_time += search_time
                        total_nodes += stats["total_nodes"]
                        
                        # Select action
                        visits = np.array([child.visits for child in root.children])
                        actions = [child.action for child in root.children]
                        
                        from utils.mcts_utils import apply_temperature
                        action, _ = apply_temperature(visits, actions, temperature)
                        
                        # Apply action
                        state = state.apply_action(action)
                        moves += 1
                    
                    # Game finished or max moves reached
                    game_time = time.time() - game_start
                    
                    # Record statistics
                    game_moves.append(moves)
                    game_times.append(game_time)
                    game_nodes.append(total_nodes)
                    
                    # Calculate performance metrics
                    moves_per_second_game = moves / game_time if game_time > 0 else 0
                    nodes_per_move_game = total_nodes / moves if moves > 0 else 0
                    search_time_per_move = total_search_time / moves if moves > 0 else 0
                    
                    logger.info(f"Game {game_idx+1}: {moves} moves, {total_nodes} nodes, {game_time:.1f}s")
                    logger.info(f"  Performance: {moves_per_second_game:.2f} moves/s, {nodes_per_move_game:.1f} nodes/move")
                    logger.info(f"  Search time: {search_time_per_move*1000:.1f}ms/move")
                
                # Calculate trial averages
                avg_moves = sum(game_moves) / len(game_moves) if game_moves else 0
                avg_time = sum(game_times) / len(game_times) if game_times else 0
                avg_nodes = sum(game_nodes) / len(game_nodes) if game_nodes else 0
                
                avg_moves_per_second = avg_moves / avg_time if avg_time > 0 else 0
                avg_nodes_per_move = avg_nodes / avg_moves if avg_moves > 0 else 0
                
                # Track across trials
                moves_per_second.append(avg_moves_per_second)
                nodes_per_move.append(avg_nodes_per_move)
                
                logger.info(f"Trial {trial+1} average: {avg_moves_per_second:.2f} moves/s, {avg_nodes_per_move:.1f} nodes/move")
            
            # Calculate overall averages
            overall_moves_per_second = sum(moves_per_second) / len(moves_per_second)
            overall_nodes_per_move = sum(nodes_per_move) / len(nodes_per_move)
            
            logger.info("\nBenchmark Results:")
            logger.info(f"  Configuration: {simulations} simulations, {batch_size} batch size, {num_collectors} collectors")
            logger.info(f"  Average moves per second: {overall_moves_per_second:.2f}")
            logger.info(f"  Average nodes per move: {overall_nodes_per_move:.1f}")
            logger.info(f"  Effective nodes per second: {overall_moves_per_second * overall_nodes_per_move:.1f}")
            
        finally:
            # Clean up Ray
            ray_manager.shutdown()
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()