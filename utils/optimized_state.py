# utils/optimized_state.py
"""
Optimized state representation for TicTacToe with efficient caching and serialization.
"""
import numpy as np
from typing import List, Tuple, Optional, Set, Dict, Any
from utils.game_interface import GameState

# Pre-computed win patterns for TicTacToe
WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6)             # Diagonals
]

# Cache common results for performance
EMPTY_BOARD = np.zeros(9, dtype=np.int8)
EMPTY_BOARD_TUPLE = tuple(EMPTY_BOARD.tolist())  # For hashing
EMPTY_BOARD_BYTES = EMPTY_BOARD.tobytes()

# Pre-compute all possible legal action sets for 3x3 board
# This avoids recomputing legal actions repeatedly
LEGAL_ACTIONS_CACHE = {}
for mask in range(1 << 9):  # 2^9 possible board states for empty/non-empty
    occupied = [bool(mask & (1 << i)) for i in range(9)]
    LEGAL_ACTIONS_CACHE[mask] = [i for i in range(9) if not occupied[i]]

class OptimizedTicTacToeState(GameState):
    """
    Memory-efficient TicTacToe state implementation with aggressive caching
    and optimized for minimal object creation during MCTS.
    
    Performance improvements:
    - Uses small int8 data type for the board
    - Caches results of expensive calculations
    - Avoids object creation during search
    - Implements fast serialization/deserialization
    - Uses pre-computed lookup tables where possible
    """
    __slots__ = ('board', 'current_player', '_winner', '_legal_actions', 
                 '_hash', '_terminal', '_board_bytes', '_mask')
    
    def __init__(self, board=None, current_player=1):
        """Initialize the state with an optional board and current player"""
        if board is None:
            self.board = EMPTY_BOARD.copy()
        elif isinstance(board, np.ndarray):
            self.board = board.astype(np.int8) if board.dtype != np.int8 else board
        else:
            self.board = np.array(board, dtype=np.int8)
        
        self.current_player = current_player
        
        # Cache computed values
        self._winner = None
        self._legal_actions = None
        self._hash = None
        self._terminal = None
        self._board_bytes = None
        
        # Compute occupied cell mask for legal action lookup
        self._mask = sum(1 << i for i in range(9) if self.board[i] != 0)
    
    def is_terminal(self) -> bool:
        """Check if the state is terminal (game ended)"""
        if self._terminal is not None:
            return self._terminal
        
        self._terminal = self._compute_winner() is not None
        return self._terminal
    
    def get_legal_actions(self) -> List[int]:
        """Returns indices of empty cells (legal moves)"""
        if self._legal_actions is not None:
            return self._legal_actions
        
        # Use pre-computed legal actions if available
        self._legal_actions = LEGAL_ACTIONS_CACHE.get(self._mask)
        
        # Fall back to computing if not in cache (shouldn't happen for 3x3)
        if self._legal_actions is None:
            self._legal_actions = np.where(self.board == 0)[0].tolist()
            
        return self._legal_actions
    
    def apply_action(self, action: int) -> 'OptimizedTicTacToeState':
        """Returns a new state after applying the action"""
        if self.board[action] != 0:
            # Use more informative error
            board_str = ' '.join(str(self.board[i]) for i in range(9))
            raise ValueError(f"Invalid action {action}, cell already occupied. Board: [{board_str}]")
        
        # Create new board with minimal copying
        new_board = self.board.copy()
        new_board[action] = self.current_player
        
        # Return new state with flipped player
        return OptimizedTicTacToeState(new_board, -self.current_player)
    
    def get_current_player(self) -> int:
        """Returns the current player (1 or -1)"""
        return self.current_player
    
    def encode(self) -> np.ndarray:
        """
        Encode the state for neural network input (minimal version)
        Returns a flat array for efficiency
        """
        # Simple flat encoding for the model
        return self.board.astype(np.float32)
    
    def encode_for_nn(self) -> np.ndarray:
        """
        Encode the state for neural network input (full version with planes)
        Returns a tensor of shape (3, 3, 3) with separate planes for players and turn
        """
        player_pieces = (self.board == self.current_player).astype(np.float32)
        opponent_pieces = (self.board == -self.current_player).astype(np.float32)
        
        # Create a tensor with 3 planes (player, opponent, turn)
        return np.stack([
            player_pieces.reshape(3, 3),
            opponent_pieces.reshape(3, 3),
            np.full((3, 3), self.current_player, dtype=np.float32)
        ])
    
    def _compute_winner(self) -> Optional[int]:
        """Compute the winner, caching the result"""
        # Check win patterns
        for i, j, k in WIN_PATTERNS:
            if self.board[i] != 0 and self.board[i] == self.board[j] == self.board[k]:
                return self.board[i]  # Return the winner (1 or -1)
        
        # Check for draw - if no empty cells left, it's a draw
        if self._mask == 0b111111111:  # All 9 bits set means no empty cells
            return 0
            
        # Game not finished
        return None
    
    def get_winner(self) -> Optional[int]:
        """Returns the winner if game ended, else None"""
        if self._winner is None:
            self._winner = self._compute_winner()
        return self._winner
    
    def __eq__(self, other: Any) -> bool:
        """Check equality with another state"""
        if not isinstance(other, OptimizedTicTacToeState):
            return False
            
        return (self._mask == other._mask and
                np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)
    
    def __hash__(self) -> int:
        """Hash function for state (for caching)"""
        if self._hash is None:
            # Convert board to a tuple for hashing
            if not hasattr(self, '_board_tuple'):
                board_tuple = tuple(self.board.tolist())
            else:
                board_tuple = self._board_tuple
            self._hash = hash((board_tuple, self.current_player))
        return self._hash
    
    def to_bytes(self) -> bytes:
        """
        Convert state to bytes for efficient serialization.
        Format: 9 bytes for board + 1 byte for current_player
        """
        if self._board_bytes is None:
            self._board_bytes = self.board.tobytes()
        
        # Combine board bytes and current_player
        return self._board_bytes + bytes([self.current_player])
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'OptimizedTicTacToeState':
        """Create state from byte representation"""
        # Extract board and current_player
        board = np.frombuffer(data[:-1], dtype=np.int8)
        current_player = int(data[-1])
        
        return cls(board, current_player)
    
    def __str__(self) -> str:
        """String representation of the state"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        rows = []
        for i in range(0, 9, 3):
            row = ' '.join(symbols[p] for p in self.board[i:i+3])
            rows.append(row)
        return '\n'.join(rows)
    
    def clone(self) -> 'OptimizedTicTacToeState':
        """Create a deep copy of this state"""
        return OptimizedTicTacToeState(self.board.copy(), self.current_player)
    
    @property
    def board_array(self) -> np.ndarray:
        """Direct access to the board array for optimized operations"""
        return self.board
    
    @property
    def occupied_mask(self) -> int:
        """Binary mask of occupied cells (for fast legal move checking)"""
        return self._mask
    
    def get_state_for_serialization(self) -> Dict[str, Any]:
        """Get a serialization-friendly dictionary representation"""
        return {
            "board": self.board.tolist(),
            "player": self.current_player
        }
    
    @classmethod
    def from_serialized(cls, data: Dict[str, Any]) -> 'OptimizedTicTacToeState':
        """Create a state from serialized dictionary"""
        return cls(
            board=np.array(data["board"], dtype=np.int8),
            current_player=data["player"]
        )