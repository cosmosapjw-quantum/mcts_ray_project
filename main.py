# main.py
"""
Main entry point for AlphaZero-style training.
Provides a command-line interface for training and evaluation.
"""
import ray
import argparse
import os
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

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='AlphaZero-style training and evaluation')
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
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()