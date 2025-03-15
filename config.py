# config.py
"""
Configuration file with hyperparameters for AlphaZero-style training.
"""

# Hardware-optimized hyperparameters
LEARNING_RATE = 3e-4            # Increased for faster convergence
WEIGHT_DECAY = 1e-4             # Slightly increased regularization
BATCH_SIZE = 1024               # Much larger batch size for GPU efficiency
REPLAY_BUFFER_SIZE = 500000     # Store many more experiences
CHECKPOINT_DIR = 'checkpoints'
NUM_SIMULATIONS = 800           # 4x more simulations for better gameplay
MCTS_BATCH_SIZE = 64            # Larger batches for MCTS evaluations
NUM_WORKERS = 16                # Utilize more CPU threads
NUM_PARALLEL_GAMES = 12         # Run multiple self-play games in parallel
TRAINING_EPOCHS = 10            # Train multiple epochs per batch
EXPLORATION_WEIGHT = 1.4        # Increased exploration vs exploitation
MIN_BUFFER_SIZE = BATCH_SIZE    # Wait until we have at least one batch
TEMPERATURE_SCHEDULE = {
    0: 1.0,     # First moves use temperature 1.0
    15: 0.5,    # After 15 moves, temperature drops to 0.5
    30: 0.25,   # After 30 moves, temperature drops to 0.25
}

# Ray configuration
RAY_OBJECT_STORE_MEMORY = 16 * 1024 * 1024 * 1024  # 16GB object store
RAY_HEAP_MEMORY = 32 * 1024 * 1024 * 1024  # 32GB heap memory

# Inference server configuration
INFERENCE_SERVER_BATCH_WAIT = 0.001
INFERENCE_SERVER_CACHE_SIZE = 10000
INFERENCE_SERVER_MAX_BATCH_SIZE = 256