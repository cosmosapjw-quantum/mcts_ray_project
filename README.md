# MCTS AlphaZero Implementation

A fast and efficient Monte Carlo Tree Search (MCTS) implementation in Python with root parallelization and centralized batch inference. This project provides a semi-production level codebase using pure Python and Ray for easy debugging with minimal boilerplate.

## Features

- **Flexible architecture** for various board games and neural network models
- **Optimized MCTS implementations**:
  - Single-threaded with Numba acceleration
  - Root parallelization with Ray
  - Time-based search option
- **Batched inference server** for efficient GPU utilization
- **Comprehensive self-play training** pipeline with replay buffer
- **Automatic hyperparameter optimization** with Ray Tune
- **Visualization and analysis tools** for model performance and hyperparameters

## Project Structure

- `mcts/`: Core MCTS implementations and algorithms
  - `core.py`: Foundational MCTS algorithms
  - `tree.py`: Optimized Numba-accelerated implementations
  - `search.py`: Distributed and parallel search with Ray
  - `node.py`: Enhanced tree node implementation
- `inference/`: Neural network inference components
  - `batch_inference_server.py`: Ray-based batch inference server
- `train/`: Training and optimization modules
  - `self_play.py`: Self-play manager
  - `replay_buffer.py`: Experience replay buffer
  - `trainer.py`: Neural network trainer
- `utils/`: Utility functions and interfaces
  - `game_interface.py`: Abstract interface for game states
  - `state_utils.py`: TicTacToe state implementation
  - `mcts_utils.py`: Utility functions for MCTS
- `hyperparameter_tuning.py`: Automated hyperparameter optimization
- `hyperparameter_analysis.py`: Analysis and visualization of tuning results
- `config.py`: Configuration and hyperparameters
- `main.py`: CLI for training, evaluation, and hyperparameter tuning

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcts-alphazero.git
cd mcts-alphazero

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train a model with default parameters:

```bash
python main.py train --games 200
```

Train with specific settings:

```bash
python main.py train --games 500 --enable-parallel-mcts --max-workers 4
```

Use time-based search instead of fixed simulations:

```bash
python main.py train --time-based-search --search-time 1.5
```

### Evaluation

Evaluate a trained model:

```bash
python main.py eval --checkpoint model_latest --games 20
```

### Hyperparameter Optimization

The project includes a comprehensive hyperparameter optimization system built on Ray Tune. This allows you to systematically search for optimal hyperparameters across your entire training pipeline.

#### Running Hyperparameter Tuning

Basic tuning with default settings:

```bash
python main.py tune --games 50 --trials 10
```

Advanced tuning with resource configuration:

```bash
python main.py tune --games 50 --trials 20 --cpus 8 --gpus 1 --concurrent-trials 4 --search-algo bayesopt --scheduler asha
```

Available options:
- `--games`: Number of games for each trial
- `--trials`: Total number of configurations to try
- `--concurrent-trials`: Number of trials to run in parallel
- `--cpus`, `--gpus`: Total resources to allocate
- `--cpus-per-trial`, `--gpus-per-trial`: Resources per trial
- `--search-algo`: Search algorithm (random, bayesopt, hyperopt)
- `--scheduler`: Scheduler for early stopping (asha, pbt, none)
- `--output-dir`: Directory for results
- `--resume`: Resume previous tuning session

#### Analyzing Tuning Results

After running hyperparameter optimization, you can analyze the results:

```bash
python main.py analyze ./ray_results/latest_experiment --all
```

Generate specific visualizations:

```bash
python main.py analyze ./ray_results/latest_experiment --learning-curves --parameter-importance --output-dir ./analysis
```

Apply the best configuration to a file:

```bash
python main.py analyze ./ray_results/latest_experiment --apply-best
```

Analysis options:
- `--learning-curves`: Plot learning curves for top configurations
- `--parameter-importance`: Visualize parameter impact on performance
- `--pairwise`: Plot pairwise relationships between parameters
- `--parallel-coords`: Create parallel coordinates plots
- `--print-best`: Print details of best configurations
- `--apply-best`: Generate a new config file with best parameters
- `--all`: Run all analyses
- `--metric`: Metric to optimize (default: loss)
- `--mode`: Optimization mode (min or max)
- `--top-n`: Number of top configurations to consider

#### Hyperparameters Being Optimized

The optimization system tunes parameters across the entire pipeline:

- **Learning parameters**: Learning rates, weight decay, batch sizes
- **MCTS parameters**: Simulation counts, batch sizes, exploration weights
- **Temperature parameters**: Initial and final temperatures, decay schedule
- **Inference server parameters**: Batch wait times, cache sizes, max batch sizes
- **Search strategy parameters**: Parallel vs. batched search, time-based vs. simulation-based

## Extending the Project

### Adding New Games

1. Implement the `GameState` interface in `utils/game_interface.py`
2. Create a new state implementation similar to `TicTacToeState`
3. Update model architecture if necessary

### Customizing Hyperparameter Search

Modify `hyperparameter_tuning.py` to customize the search space:

```python
def get_search_space():
    return {
        # Add or modify parameters as needed
        "LEARNING_RATE": tune.loguniform(1e-5, 1e-2),
        "MY_NEW_PARAMETER": tune.choice([1, 2, 3]),
        # ...
    }
```

## Performance Considerations

- For maximum performance, use parallel MCTS with batch inference
- GPU acceleration provides significant speedups for model inference
- Adjust batch sizes based on your hardware capabilities
- Consider time-based search for more consistent move timing

## License

[MIT License](LICENSE)