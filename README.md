# MCTS AlphaZero Implementation

A high-performance Monte Carlo Tree Search (MCTS) implementation in Python with leaf parallelization and centralized batch inference. This project provides a sophisticated framework for AlphaZero-style reinforcement learning, optimized for both CPU and GPU utilization while maintaining adaptability for various board games.

## Key Features

- **Multiple optimized MCTS implementations**:
  - Standard single-threaded MCTS with configurable exploration parameters
  - Numba-accelerated tree operations for improved CPU performance
  - Leaf-parallel MCTS with multi-threaded collectors for efficient tree exploration
  - Ray-based distributed search with fault tolerance

- **Advanced batch inference architecture**:
  - Centralized inference server with adaptive batching
  - Optimized GPU utilization with configurable batch sizing
  - Efficient caching system to reduce redundant evaluations
  - Mixed precision support for faster inference

- **Comprehensive training pipeline**:
  - Asynchronous self-play with separate processes for generation and training
  - Experience replay buffer with prioritized sampling
  - Enhanced optimization with mixed precision training
  - Checkpointing and model versioning

- **Robust system design**:
  - Improved Ray manager with health monitoring and automatic recovery
  - Performance profiling and bottleneck detection
  - Memory optimization for large-scale training
  - Systematic error handling and graceful degradation

## Project Structure

```
├── mcts/                       # Core MCTS implementations
│   ├── core.py                 # Foundational MCTS algorithms
│   ├── tree.py                 # Numba-optimized implementations
│   ├── search.py               # Distributed and parallel search with Ray
│   ├── node.py                 # Enhanced tree node implementation 
│   ├── enhanced_batch_search.py# Optimized batched MCTS
│   └── leaf_parallel_mcts.py   # Multi-threaded leaf parallelization
├── inference/                  # Neural network inference components
│   ├── state_batcher.py        # Efficient state batching for GPU
│   └── enhanced_batch_inference_server.py # Advanced batching server
├── train/                      # Training and optimization modules
│   ├── async_training.py       # Asynchronous training pipeline
│   ├── efficient_data.py       # Optimized data handling
│   ├── enhanced_self_play.py   # Improved self-play manager
│   ├── optimized_trainer.py    # Training with enhanced features
│   ├── patch_self_play.py      # Mock components for testing
│   ├── replay_buffer.py        # Experience replay buffer
│   └── trainer.py              # Neural network trainer
├── utils/                      # Utility functions and interfaces
│   ├── game_interface.py       # Abstract interface for game states
│   ├── improved_ray_manager.py # Enhanced Ray configuration
│   ├── mcts_utils.py           # Utility functions for MCTS
│   ├── memory_optimization.py  # Memory usage optimization
│   ├── optimized_state.py      # Efficient state representation
│   ├── profiling.py            # Performance measurement tools
│   └── state_utils.py          # Game state implementations
├── config.py                   # Configuration and hyperparameters
├── model.py                    # Neural network architecture
└── main_optimized.py           # Enhanced entry point with optimized settings
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcts-alphazero.git
cd mcts-alphazero

# Install dependencies
pip install -r requirements.txt

# Optional: Install CUDA for GPU acceleration
# See https://pytorch.org/get-started/locally/ for PyTorch with CUDA
```

Required dependencies:
- Python 3.8+
- PyTorch 1.9+
- Ray 2.0+
- NumPy
- Numba (for optimized tree operations)

## Usage

### Optimized Training

Train a model with enhanced settings for modern hardware:

```bash
python main_optimized.py train --games 200 --batch-size 64 --collectors 8
```

Available options:
- `--games`: Number of games to play
- `--checkpoint`: Path to load existing checkpoint
- `--simulations`: MCTS simulations per move (default: 800)
- `--batch-size`: Batch size for MCTS leaf evaluation (default: 32)
- `--collectors`: Number of leaf collector threads (default: 2)
- `--verbose`: Print detailed information

### Asynchronous Training

Run training with separate self-play and training processes:

```bash
python main_optimized.py async --collectors 4 --hours 12
```

Options:
- `--hours`: Training duration in hours (None = indefinite)
- `--checkpoint`: Checkpoint to load
- `--collectors`: Number of experience collectors

### Evaluation

Evaluate a trained model:

```bash
python main_optimized.py eval --checkpoint model_latest --games 20
```

### Performance Benchmarking

Benchmark MCTS performance with different configurations:

```bash
python main_optimized.py benchmark --simulations 800 --batch-size 64 --collectors 8 --games 5
```

## Hardware Optimization

This implementation includes sophisticated optimizations for modern multi-core CPUs and GPUs:

### CPU Optimization

For optimal performance on multi-core systems:

```bash
# For 12-core CPU (e.g., Ryzen 9 5900X)
python main_optimized.py train --collectors 8 --batch-size 64
```

Key parameters to adjust:
- `collectors`: Set to approximately 75% of physical cores
- `batch-size`: Larger values reduce overhead but increase latency

### GPU Optimization

For optimal GPU utilization (NVIDIA RTX series):

```bash
# For RTX 3060 Ti or similar
python main_optimized.py train --batch-size 256 --no-mixed-precision False
```

The implementation automatically:
- Uses mixed precision (FP16) when available
- Adapts batch wait times to maximize GPU utilization
- Implements efficient cache management to reduce redundant computations

## Extending the Project

### Adding New Games

1. Implement the `GameState` interface in `utils/game_interface.py`:

```python
class MyGameState(GameState):
    def is_terminal(self) -> bool:
        # Implementation
        
    def get_legal_actions(self) -> list:
        # Implementation
        
    def apply_action(self, action) -> 'GameState':
        # Implementation
        
    def get_winner(self):
        # Implementation
        
    def encode(self):
        # Implementation for neural network input
```

2. Update the model architecture in `model.py` if necessary
3. Register your game in `config.py`

### Custom Neural Network Models

Modify the `model.py` file to implement your architecture:

```python
class MyCustomModel(nn.Module):
    def __init__(self, input_shape, action_size):
        super().__init__()
        # Define your layers
        
    def forward(self, x):
        # Implementation
        policy = ...
        value = ...
        return policy, value
```

### Distributed Training

For multi-node training:

1. Initialize Ray with cluster settings in `improved_ray_manager.py`
2. Adjust resource allocation based on your cluster configuration
3. Use the asynchronous training mode with `main_optimized.py async`

## Performance Tuning

### Batch Size Selection

For optimal performance, adjust batch sizes based on hardware:

- **CPU-constrained**: Smaller batches (16-32)
- **GPU-constrained**: Larger batches (128-512)
- **Balanced**: Medium batches (64-128)

### Memory Optimization

The implementation includes several memory optimization strategies:

- Efficient state representation with `__slots__`
- Caching with key-based lookup
- Numpy arrays with appropriate data types
- Binary serialization for state transmission

### Parallelism Control

Adjust parallelism based on your specific hardware:

- `num_collectors`: Number of tree exploration threads (typically 2-12)
- `batch_size`: Size of evaluation batches (16-512)
- `adaptive_batching`: Enables dynamic adjustment during execution

## Troubleshooting

### Common Issues

- **Out of memory errors**: Reduce `cache_size` and `replay_buffer_size`
- **Poor GPU utilization**: Increase `batch_size` and enable mixed precision
- **High CPU usage**: Reduce `num_collectors` or limit CPU with `--cpu-limit`
- **Stalled search**: Check for bottlenecks with `--verbose` option
- **Slow training**: Increase `simulations` for better model convergence

### Performance Monitoring

The implementation includes detailed performance monitoring:

```bash
python main_optimized.py benchmark --verbose
```

This provides metrics on:
- Tree exploration efficiency
- Batch size distribution
- GPU utilization
- Cache hit rates
- Search time breakdown

## Future Improvements

Based on detailed assessment, future development will focus on:

1. Further improving CPU parallelism with finer-grained locking
2. Enhancing game abstraction for greater flexibility
3. Optimizing inference for specific GPU architectures
4. Implementing subtree reuse between consecutive moves
5. Strengthening error handling and diagnostic capabilities

## License

[MIT License](LICENSE)