# utils/state_utils.py (Fixed TicTacToe State)
import numpy as np
from utils.game_interface import GameState

# Pre-computed win patterns for TicTacToe
WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6)             # Diagonals
]

def check_winner(board):
    """Check if there's a winner (non-Numba version)"""
    # Check win patterns
    for i, j, k in WIN_PATTERNS:
        if board[i] != 0 and board[i] == board[j] == board[k]:
            return board[i]  # Return the winner (1 or -1)
    
    # Check for draw
    if 0 not in board:
        return 0  # Draw
    
    # Game not finished
    return None

class TicTacToeState(GameState):
    """
    Memory-efficient TicTacToe state implementation
    """
    __slots__ = ('board', 'current_player', '_winner', '_legal_actions', '_hash')
    
    def __init__(self, board=None, current_player=1):
        """Initialize the state with an optional board and current player"""
        if board is None:
            self.board = np.zeros(9, dtype=np.int8)
        else:
            self.board = np.array(board, dtype=np.int8)
        
        self.current_player = current_player
        self._winner = None
        self._legal_actions = None
        self._hash = None
    
    def is_terminal(self):
        """Check if the state is terminal (game ended)"""
        if self._winner is not None:
            return self._winner is not None
        
        self._winner = check_winner(self.board)
        return self._winner is not None
    
    def get_legal_actions(self):
        """Returns indices of empty cells (legal moves)"""
        if self._legal_actions is not None:
            return self._legal_actions
        
        self._legal_actions = np.where(self.board == 0)[0].tolist()
        return self._legal_actions
    
    def apply_action(self, action):
        """Returns a new state after applying the action"""
        if self.board[action] != 0:
            raise ValueError(f"Invalid action {action}, cell already occupied")
        
        # Create new state
        new_board = self.board.copy()
        new_board[action] = self.current_player
        
        # Return new state with flipped player
        return TicTacToeState(new_board, -self.current_player)
    
    def get_current_player(self):
        """Returns the current player (1 or -1)"""
        return self.current_player
    
    def encode(self):
        """
        Encode the state for neural network input
        Returns a tuple of (current_player_pieces, opponent_pieces)
        """
        player_pieces = (self.board == self.current_player).astype(np.float32)
        opponent_pieces = (self.board == -self.current_player).astype(np.float32)
        
        # If we need full board representation for the model
        return np.concatenate([
            player_pieces.reshape(3, 3),
            opponent_pieces.reshape(3, 3),
            np.full((3, 3), self.current_player, dtype=np.float32)
        ])
    
    def encode_flat(self):
        """Return a flat encoding for the model"""
        # Simpler encoding: board values directly
        return self.board.astype(np.float32)
    
    @property
    def winner(self):
        """Compute and cache the winner"""
        if self._winner is None:
            self._winner = check_winner(self.board)
        return self._winner
    
    def get_winner(self):
        """Returns the winner if game ended, else None"""
        return self.winner
    
    def __eq__(self, other):
        """Check equality with another state"""
        if not isinstance(other, TicTacToeState):
            return False
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)
    
    def __hash__(self):
        """Hash function for state"""
        if self._hash is None:
            # Convert board to a tuple for hashing
            board_tuple = tuple(self.board.tolist())
            self._hash = hash((board_tuple, self.current_player))
        return self._hash
    
    def __str__(self):
        """String representation of the state"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        rows = []
        for i in range(0, 9, 3):
            row = ' '.join(symbols[p] for p in self.board[i:i+3])
            rows.append(row)
        return '\n'.join(rows)
    
    def clone(self):
        """Create a deep copy of this state"""
        return TicTacToeState(self.board.copy(), self.current_player)