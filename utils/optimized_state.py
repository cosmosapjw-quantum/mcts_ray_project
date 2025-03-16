# utils/optimized_state.py
"""
Optimized state representation for TicTacToe with efficient caching and serialization.
"""
import numpy as np
from utils.game_interface import GameState

# Pre-computed win patterns for TicTacToe
WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6)             # Diagonals
]

class TicTacToeState(GameState):
    """
    Enhanced TicTacToe state implementation conforming to the updated GameState interface.
    """
    __slots__ = ('board', 'current_player', '_winner', '_legal_actions', '_hash')
    
    # Class constants
    POLICY_SIZE = 9  # 3x3 board
    BOARD_SIZE = 3
    GAME_NAME = "TicTacToe"
    
    def __init__(self, board=None, current_player=1):
        """Initialize the state with an optional board and current player"""
        if board is None:
            self.board = np.zeros(self.POLICY_SIZE, dtype=np.int8)
        else:
            self.board = np.array(board, dtype=np.int8)
        
        self.current_player = current_player
        self._winner = None
        self._legal_actions = None
        self._hash = None
    
    @property
    def policy_size(self) -> int:
        """Return the size of the policy vector (9 for 3x3 TicTacToe)"""
        return self.POLICY_SIZE
    
    @property
    def game_name(self) -> str:
        """Return the game name"""
        return self.GAME_NAME
    
    def is_terminal(self) -> bool:
        """Check if the state is terminal (game ended)"""
        if self._winner is not None:
            return self._winner is not None
        
        # Check win conditions
        for i, j, k in WIN_PATTERNS:
            if self.board[i] != 0 and self.board[i] == self.board[j] == self.board[k]:
                self._winner = self.board[i]
                return True
        
        # Check for draw - if no empty cells left
        if 0 not in self.board:
            self._winner = 0  # Draw
            return True
            
        return False
    
    def get_legal_actions(self) -> list:
        """Returns indices of empty cells (legal moves)"""
        if self._legal_actions is not None:
            return self._legal_actions
        
        self._legal_actions = np.where(self.board == 0)[0].tolist()
        return self._legal_actions
    
    def apply_action(self, action) -> 'TicTacToeState':
        """Returns a new state after applying the action"""
        if self.board[action] != 0:
            raise ValueError(f"Invalid action {action}, cell already occupied")
        
        # Create new state
        new_board = self.board.copy()
        new_board[action] = self.current_player
        
        # Return new state with flipped player
        return TicTacToeState(new_board, -self.current_player)
    
    def get_current_player(self) -> int:
        """Returns the current player (1 or -1)"""
        return self.current_player
    
    def encode(self) -> np.ndarray:
        """
        Basic encoding - just return the board as a flat array
        """
        return self.board.astype(np.float32)
    
    def encode_for_inference(self) -> np.ndarray:
        """
        Enhanced encoding for neural network input with separate planes
        Returns a tensor of shape (3, 3, 3) with channels:
        - Channel 0: Current player's pieces
        - Channel 1: Opponent's pieces
        - Channel 2: Constant plane indicating current player
        """
        # Create planes for current player and opponent
        player_pieces = (self.board == self.current_player).astype(np.float32)
        opponent_pieces = (self.board == -self.current_player).astype(np.float32)
        current_player_plane = np.full(self.POLICY_SIZE, self.current_player, dtype=np.float32)
        
        # Stack into channels-first format for PyTorch (3, 3, 3)
        planes = np.stack([
            player_pieces.reshape(self.BOARD_SIZE, self.BOARD_SIZE),
            opponent_pieces.reshape(self.BOARD_SIZE, self.BOARD_SIZE),
            current_player_plane.reshape(self.BOARD_SIZE, self.BOARD_SIZE)
        ])
        
        return planes
    
    def get_winner(self) -> int:
        """Returns the winner if game ended, else None"""
        if self._winner is None and self.is_terminal():
            pass  # is_terminal already set self._winner
        return self._winner
    
    def get_canonical_state(self) -> 'TicTacToeState':
        """
        Return canonical form of the state (first player's perspective).
        For TicTacToe, this means flipping the board if current player is -1.
        """
        if self.current_player == 1:
            return self
        else:
            # Flip the perspective
            flipped_board = -self.board
            return TicTacToeState(flipped_board, 1)
    
    def get_symmetries(self) -> list:
        """
        Get all symmetries of the current board position.
        For 3x3 TicTacToe, there are 8 symmetries (4 rotations * 2 reflections).
        
        Returns:
            List of (state, action_mapping) tuples, where action_mapping[i]
            gives the action index in the original state corresponding to action i
            in the symmetric state.
        """
        # Reshape board to 2D for easier transformations
        board_2d = self.board.reshape(self.BOARD_SIZE, self.BOARD_SIZE)
        
        # Initialize results
        symmetries = []
        action_maps = []
        
        # Get all unique symmetries through rotations and reflections
        # For 3x3 board, we have 4 rotations and 1 reflection
        for i in range(4):  # 4 rotations (0, 90, 180, 270 degrees)
            rot_board = np.rot90(board_2d, i).copy()
            rot_flat = rot_board.flatten()
            
            # Calculate action mapping (original -> rotated)
            rot_map = []
            for y in range(self.BOARD_SIZE):
                for x in range(self.BOARD_SIZE):
                    # Map coordinates based on rotation
                    if i == 0:  # 0 degrees
                        orig_idx = y * self.BOARD_SIZE + x
                    elif i == 1:  # 90 degrees
                        orig_idx = (self.BOARD_SIZE - 1 - x) * self.BOARD_SIZE + y
                    elif i == 2:  # 180 degrees
                        orig_idx = (self.BOARD_SIZE - 1 - y) * self.BOARD_SIZE + (self.BOARD_SIZE - 1 - x)
                    else:  # 270 degrees
                        orig_idx = x * self.BOARD_SIZE + (self.BOARD_SIZE - 1 - y)
                    rot_map.append(orig_idx)
            
            # Add rotation
            symmetries.append(TicTacToeState(rot_flat, self.current_player))
            action_maps.append(rot_map)
            
            # Add reflection of rotation
            reflect_board = np.fliplr(rot_board).copy()
            reflect_flat = reflect_board.flatten()
            
            # Calculate action mapping for reflection
            reflect_map = []
            for y in range(self.BOARD_SIZE):
                for x in range(self.BOARD_SIZE):
                    # Map to rotated coordinates first
                    if i == 0:  # 0 degrees
                        rot_y, rot_x = y, x
                    elif i == 1:  # 90 degrees
                        rot_y, rot_x = x, self.BOARD_SIZE - 1 - y
                    elif i == 2:  # 180 degrees
                        rot_y, rot_x = self.BOARD_SIZE - 1 - y, self.BOARD_SIZE - 1 - x
                    else:  # 270 degrees
                        rot_y, rot_x = self.BOARD_SIZE - 1 - x, y
                    
                    # Then reflect across vertical axis
                    orig_idx = rot_y * self.BOARD_SIZE + (self.BOARD_SIZE - 1 - rot_x)
                    reflect_map.append(orig_idx)
            
            # Add reflection
            symmetries.append(TicTacToeState(reflect_flat, self.current_player))
            action_maps.append(reflect_map)
        
        # Return unique symmetries with their action mappings
        unique_symmetries = []
        seen = set()
        
        for i, (sym_state, action_map) in enumerate(zip(symmetries, action_maps)):
            # Use board as a key to detect duplicates
            board_tuple = tuple(sym_state.board.tolist())
            if board_tuple not in seen:
                seen.add(board_tuple)
                unique_symmetries.append((sym_state, action_map))
        
        return unique_symmetries
    
    def to_key(self) -> str:
        """Efficient key representation for caching"""
        return self.board.tobytes().hex()
    
    def get_observation_shape(self) -> tuple:
        """Return shape needed for neural network input"""
        # For AlphaZero-style networks, we return the shape with channels first
        return (3, self.BOARD_SIZE, self.BOARD_SIZE)
    
    def __eq__(self, other) -> bool:
        """Check equality with another state"""
        if not isinstance(other, TicTacToeState):
            return False
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)
    
    def __hash__(self) -> int:
        """Hash function for state"""
        if self._hash is None:
            # Convert board to a tuple for hashing
            board_tuple = tuple(self.board.tolist())
            self._hash = hash((board_tuple, self.current_player))
        return self._hash
    
    def __str__(self) -> str:
        """String representation of the state"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        rows = []
        for i in range(0, 9, 3):
            row = ' '.join(symbols[p] for p in self.board[i:i+3])
            rows.append(row)
        return '\n'.join(rows)
    
    def clone(self) -> 'TicTacToeState':
        """Create a deep copy of this state"""
        return TicTacToeState(self.board.copy(), self.current_player)