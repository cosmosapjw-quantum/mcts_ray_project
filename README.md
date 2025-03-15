# AlphaZero-style Learning Framework

A modular, efficient implementation of AlphaZero-style reinforcement learning for board games, featuring optimized MCTS and distributed training capabilities.

## Features

- **Optimized MCTS**: Multiple MCTS implementations from simple reference to high-performance versions
- **Distributed Training**: Parallel self-play and inference with Ray
- **GPU Acceleration**: Efficient batched neural network inference with PyTorch
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

## Usage

### Training a Model

```bash
# Basic training
python main.py train --games 200

# Training with time-based search
python main.py train --time-based-search --search-time 0.5

# Continue training from a checkpoint
python main.py train --checkpoint model_100 --games 200
```

### Evaluating a Model

```bash
# Evaluate a trained model
python main.py eval --checkpoint model_final --games 20 --simulations 1600

# Detailed evaluation with verbose output
python main.py eval --checkpoint model_final --games 5 --verbose
```

## Extending to New Games

To add support for a new game:

1. Implement the `GameState` interface from `utils/game_interface.py`
2. Add game-specific state encoding in your implementation
3. Update the neural network architecture in `model.py` if needed

## Performance Considerations

- For optimal performance, use `mcts/search.py` with a GPU for inference
- The `BatchMCTSWorker` provides efficient batched evaluation for large models
- For debugging and experimentation, the simpler implementations in `mcts/core.py` are recommended

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Ray 2.0+
- NumPy
- Numba (for optimized tree search)