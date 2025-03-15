# AlphaZero-style Learning Framework

A modular, efficient implementation of AlphaZero-style reinforcement learning for board games, featuring optimized MCTS and distributed training capabilities.

## Features

- **Multiple MCTS Implementations**: Choose from basic reference, high-performance, or distributed algorithms
- **Distributed Training**: Parallel self-play and inference with Ray
- **Task-based Parallelism**: Efficient and stable parallel MCTS implementation
- **GPU Acceleration**: Batched neural network inference with PyTorch
- **Modular Design**: Clean separation of concerns for easy extension to new games

## Project Structure

```
alphazero/               # Root package folder
├── __init__.py          # Package initialization
├── main.py              # Main entry point
├── config.py            # Configuration and hyperparameters
├── model.py             # Neural network architecture
├── mcts/                # Monte Carlo Tree Search implementations
│   ├── __init__.py
│   ├── core.py          # Basic MCTS algorithms
│   ├── tree.py          # Optimized implementations with Numba
│   ├── search.py        # Distributed search with Ray
│   └── node.py          # Tree node definition
├── train/               # Training-related modules
│   ├── __init__.py
│   ├── self_play.py     # Self-play game generation
│   ├── trainer.py       # Neural network training logic
│   └── replay_buffer.py # Experience replay buffer
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── game_interface.py # Game interface abstractions
│   ├── state_utils.py    # Game state implementations
│   └── mcts_utils.py     # MCTS utilities
└── inference/           # Neural network inference
    └── batch_inference_server.py  # Batched inference server
```

## MCTS Implementations

The framework provides three MCTS implementations with increasing performance characteristics:

1. **Basic MCTS** (`mcts/core.py`): Simple, reference implementation for learning and experimentation
2. **Optimized MCTS** (`mcts/tree.py`): Performance-optimized with Numba for single-machine execution
3. **Distributed MCTS** (`mcts/search.py`): Parallel and batched search for distributed computing

### Parallel MCTS Approaches

The framework supports multiple parallelization strategies:

1. **Task-based Parallel MCTS**: Uses Ray tasks for distributed computation with minimal serialization overhead
2. **Batched MCTS**: Collects leaves in batches for efficient neural network inference
3. **Time-based MCTS**: Uses time budgets instead of simulation counts for more consistent performance

## Usage

### Training a Model

```bash
# Basic training with batched search (default)
python main.py train --games 200

# Training with task-based parallel MCTS (recommended)
python main.py train --enable-parallel-mcts --max-workers 2 --games 200

# Training with more parallelism (if your system can handle it)
python main.py train --enable-parallel-mcts --max-workers 4 --games 200

# Training with time-based search (very stable)
python main.py train --time-based-search --search-time 0.5 --games 200

# Continue training from a checkpoint
python main.py train --checkpoint model_100 --games 200 --enable-parallel-mcts
```

### Evaluating a Model

```bash
# Evaluate a trained model
python main.py eval --checkpoint model_final --games 20 --simulations 1600

# Detailed evaluation with verbose output
python main.py eval --checkpoint model_final --games 5 --verbose
```

## Performance Tuning

The performance of the MCTS implementations can be tuned based on your hardware:

### For CPU-Only Systems
- Use batched MCTS with a small batch size (~8-16)
- Disable parallel MCTS to avoid overhead
- Use Numba-optimized tree search from `mcts/tree.py`

### For GPU Systems
- Enable task-based parallel MCTS with 2-4 workers
- Use larger batch sizes (32-64) for efficient GPU utilization
- Increase simulation count for stronger play

### For Multi-GPU Systems
- Distribute inference across GPUs using Ray
- Use more workers (up to 8) for parallel MCTS
- Consider time-based search for better load balancing

### Memory Considerations
- The replay buffer size can be adjusted in `config.py`
- For systems with limited RAM, reduce `RAY_OBJECT_STORE_MEMORY`
- Monitor GPU memory usage and adjust batch sizes accordingly

## Extending to New Games

To add support for a new game:

1. Implement the `GameState` interface from `utils/game_interface.py`
2. Add game-specific state encoding in your implementation
3. Update the neural network architecture in `model.py` if needed

Example of implementing a new game:

```python
class MyGameState(GameState):
    def __init__(self):
        # Initialize game state
        
    def is_terminal(self):
        # Check if game has ended
        
    def get_legal_actions(self):
        # Return list of legal actions
        
    def apply_action(self, action):
        # Return new state after applying action
        
    def get_current_player(self):
        # Return current player
        
    def encode(self):
        # Encode state for neural network
        
    def get_winner(self):
        # Return winner if game ended
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Ray 2.0+
- NumPy
- Numba (for optimized tree search)