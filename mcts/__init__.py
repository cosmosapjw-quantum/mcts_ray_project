# mcts/__init__.py
"""
Monte Carlo Tree Search module for AlphaZero-style learning.

This package provides three levels of MCTS implementations:

1. core.py - Basic implementations without optimizations
2. tree.py - Performance-optimized implementations with Numba
3. search.py - Distributed and batched search with Ray

Choose the appropriate implementation based on your needs:
- For learning and experimentation, use core.py
- For single-threaded performance, use tree.py
- For distributed and GPU-accelerated search, use search.py
"""
from mcts.node import Node

# Core implementations
from mcts.core import (
    select_node,
    expand_node,
    backpropagate,
    mcts_search_basic
)

# Optimized implementations
from mcts.tree import (
    select_optimized,
    expand_optimized,
    backpropagate_optimized,
    mcts_search_optimized
)

# Distributed and batched implementations
from mcts.search import (
    BatchMCTSWorker,
    parallel_mcts,
    batched_mcts_search,
    mcts_with_timeout
)

__all__ = [
    # Common classes
    'Node',
    
    # Core algorithms
    'select_node', 'expand_node', 'backpropagate', 'mcts_search_basic',
    
    # Optimized algorithms
    'select_optimized', 'expand_optimized', 'backpropagate_optimized', 
    'mcts_search_optimized',
    
    # Distributed algorithms
    'BatchMCTSWorker', 'parallel_mcts', 'batched_mcts_search', 'mcts_with_timeout'
]