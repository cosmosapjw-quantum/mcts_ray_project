# utils/state_utils.py (Fixed TicTacToe State)
import numpy as np
from utils.game_interface import GameState
from typing import List, Tuple, Optional, Union

# Pre-computed win patterns for TicTacToe
WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6)             # Diagonals
]

class TicTacToeState(GameState):
    """
    Enhanced TicTacToe state implementation conforming to the updated GameState interface.
    Includes move history tracking and temporary move application.
    """
    __slots__ = ('board', 'current_player', '_winner', '_legal_actions', '_hash', 
                '_move_history', '_temp_moves', '_temp_player')
    
    # Class constants
    POLICY_SIZE = 9  # 3x3 board
    BOARD_SIZE = 3
    GAME_NAME = "TicTacToe"
    
    def __init__(self, board=None, current_player=1, move_history=None):
        """
        Initialize the state with an optional board, current player, and move history.
        
        Args:
            board: Optional initial board state
            current_player: Current player (1 or -1)
            move_history: Optional list of previous moves
        """
        if board is None:
            self.board = np.zeros(self.POLICY_SIZE, dtype=np.int8)
        else:
            self.board = np.array(board, dtype=np.int8)
        
        self.current_player = current_player
        self._winner = None
        self._legal_actions = None
        self._hash = None
        
        # Initialize move history
        self._move_history = move_history.copy() if move_history else []
        
        # For temporary move application
        self._temp_moves = []
        self._temp_player = current_player
    
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
        """
        Returns a new state after applying the action.
        Maintains move history.
        
        Args:
            action: Cell index to place the piece (0-8)
            
        Returns:
            TicTacToeState: New state after move is applied
        """
        if self.board[action] != 0:
            raise ValueError(f"Invalid action {action}, cell already occupied")
        
        # Create new state
        new_board = self.board.copy()
        new_board[action] = self.current_player
        
        # Create updated move history
        new_history = self._move_history.copy()
        new_history.append(action)
        
        # Return new state with flipped player and updated history
        return TicTacToeState(new_board, -self.current_player, new_history)
    
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
        """Create a deep copy of this state including move history"""
        return TicTacToeState(
            self.board.copy(), 
            self.current_player,
            self._move_history.copy()
        )
        
    def make_move(self, action: int) -> bool:
        """
        Make a temporary move on the board.
        Useful for rule checking without creating new states.
        
        Args:
            action: Cell index to place the piece (0-8)
            
        Returns:
            bool: True if move was valid and applied, False otherwise
        """
        # Check if move is valid
        if action < 0 or action >= self.POLICY_SIZE:
            return False
            
        if self.board[action] != 0:
            return False
            
        # Apply move directly to the board
        self.board[action] = self._temp_player
        
        # Store move for potential undo
        self._temp_moves.append((action, self._temp_player))
        
        # Switch player
        self._temp_player = -self._temp_player
        
        # Clear cached values
        self._winner = None
        self._legal_actions = None
        self._hash = None
        
        return True
        
    def undo_move(self) -> bool:
        """
        Undo the last temporary move.
        
        Returns:
            bool: True if a move was undone, False if no moves to undo
        """
        if not self._temp_moves:
            return False
            
        # Get last move
        action, _ = self._temp_moves.pop()
        
        # Reset board position
        self.board[action] = 0
        
        # Restore player
        self._temp_player = -self._temp_player
        
        # Clear cached values
        self._winner = None
        self._legal_actions = None
        self._hash = None
        
        return True
        
    def get_move_history(self, n: Optional[int] = None) -> List[int]:
        """
        Get the history of moves made in this game.
        
        Args:
            n: Optional number of most recent moves to return (None = all moves)
            
        Returns:
            List[int]: List of actions in chronological order
        """
        if n is None or n >= len(self._move_history):
            return self._move_history.copy()
        else:
            return self._move_history[-n:]
            
    def calculate_move_distance(self, move1: int, move2: int) -> float:
        """
        Calculate the 'distance' between two moves on the TicTacToe board.
        Uses Manhattan distance on the 3x3 grid.
        
        Args:
            move1: First move (0-8)
            move2: Second move (0-8)
            
        Returns:
            float: Distance between moves
        """
        if move1 == move2:
            return 0.0
            
        # Convert to 2D coordinates
        row1, col1 = move1 // self.BOARD_SIZE, move1 % self.BOARD_SIZE
        row2, col2 = move2 // self.BOARD_SIZE, move2 % self.BOARD_SIZE
        
        # Calculate Manhattan distance
        return abs(row1 - row2) + abs(col1 - col2)

class ConnectFourState(GameState):
    """
    Connect Four game state implementation.
    This demonstrates a more complex game with the enhanced GameState interface.
    Includes move history tracking and temporary move functionality.
    """
    
    # Class constants
    ROWS = 6
    COLS = 7
    POLICY_SIZE = COLS  # Actions are column selections
    GAME_NAME = "Connect4"
    
    def __init__(self, board=None, current_player=1, move_history=None):
        """
        Initialize Connect Four state with optional board, current player, and move history.
        
        Args:
            board: Optional initial board state
            current_player: Current player (1 or -1)
            move_history: Optional list of previous moves
        """
        if board is None:
            # Initialize empty board: 0 = empty, 1 = player 1, -1 = player 2
            self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        else:
            self.board = np.array(board, dtype=np.int8)
        
        self.current_player = current_player
        self._winner = None
        self._legal_actions = None
        self._last_move = None
        
        # Initialize move history
        self._move_history = move_history.copy() if move_history else []
        
        # For temporary move application
        self._temp_moves = []
        self._temp_player = current_player
    
    @property
    def policy_size(self) -> int:
        """Return the size of the policy vector (7 for standard Connect Four)"""
        return self.COLS
    
    @property
    def game_name(self) -> str:
        """Return the game name"""
        return self.GAME_NAME
    
    def is_terminal(self) -> bool:
        """Check if the game has ended (win or draw)"""
        # Cache win check result
        if self._winner is not None:
            return True
        
        # Check for a winner
        winner = self._check_winner()
        if winner != 0:
            self._winner = winner
            return True
        
        # Check for draw (full board)
        if np.all(self.board != 0):
            self._winner = 0  # Draw
            return True
        
        return False
    
    def _check_winner(self) -> int:
        """Check if either player has won"""
        # No need to check if no moves have been made
        if self._last_move is None and np.all(self.board == 0):
            return 0
        
        # Check horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if (self.board[r, c] != 0 and 
                    self.board[r, c] == self.board[r, c+1] == 
                    self.board[r, c+2] == self.board[r, c+3]):
                    return self.board[r, c]
        
        # Check vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if (self.board[r, c] != 0 and 
                    self.board[r, c] == self.board[r+1, c] == 
                    self.board[r+2, c] == self.board[r+3, c]):
                    return self.board[r, c]
        
        # Check diagonal (rising)
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                if (self.board[r, c] != 0 and 
                    self.board[r, c] == self.board[r-1, c+1] == 
                    self.board[r-2, c+2] == self.board[r-3, c+3]):
                    return self.board[r, c]
        
        # Check diagonal (falling)
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if (self.board[r, c] != 0 and 
                    self.board[r, c] == self.board[r+1, c+1] == 
                    self.board[r+2, c+2] == self.board[r+3, c+3]):
                    return self.board[r, c]
        
        # No winner
        return 0
    
    def get_legal_actions(self) -> List[int]:
        """Return columns that aren't full (legal moves)"""
        if self._legal_actions is not None:
            return self._legal_actions
        
        # Find columns that aren't full
        self._legal_actions = [c for c in range(self.COLS) if self.board[0, c] == 0]
        return self._legal_actions
    
    def _get_next_empty_row(self, col: int) -> Optional[int]:
        """Find the next empty row in a column (where piece will land)"""
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        return None  # Column is full
    
    def apply_action(self, action: int) -> 'ConnectFourState':
        """
        Apply a move (column selection) to the board.
        Maintains move history.
        
        Args:
            action: Column to drop a piece in (0-6 for standard board)
            
        Returns:
            New ConnectFourState after the move
        """
        if action < 0 or action >= self.COLS:
            raise ValueError(f"Invalid column: {action}")
        
        # Find where piece will land
        row = self._get_next_empty_row(action)
        if row is None:
            raise ValueError(f"Column {action} is full")
        
        # Create new board
        new_board = self.board.copy()
        new_board[row, action] = self.current_player
        
        # Create updated move history
        new_history = self._move_history.copy()
        new_history.append(action)
        
        # Create new state with the move applied
        new_state = ConnectFourState(new_board, -self.current_player, new_history)
        new_state._last_move = (row, action)  # Record last move for faster win checking
        
        return new_state
    
    def get_current_player(self) -> int:
        """Return the current player (1 or -1)"""
        return self.current_player
    
    def encode(self) -> np.ndarray:
        """
        Basic encoding for the neural network.
        Returns a flat array for simple networks.
        """
        return self.board.flatten().astype(np.float32)
    
    def encode_for_inference(self) -> np.ndarray:
        """
        Enhanced encoding for neural network with separate planes.
        Returns a tensor of shape (3, ROWS, COLS) with channels:
          - Channel 0: Current player's pieces
          - Channel 1: Opponent's pieces
          - Channel 2: Constant plane indicating current player
        """
        # Create planes
        player_pieces = (self.board == self.current_player).astype(np.float32)
        opponent_pieces = (self.board == -self.current_player).astype(np.float32)
        current_player_plane = np.full((self.ROWS, self.COLS), 
                                       self.current_player, 
                                       dtype=np.float32)
        
        # Stack planes for channels-first format (PyTorch)
        planes = np.stack([player_pieces, opponent_pieces, current_player_plane])
        return planes
    
    def get_winner(self) -> Optional[int]:
        """Return the winner if game ended, else None"""
        if not self.is_terminal():
            return None
        return self._winner
    
    def get_canonical_state(self) -> 'ConnectFourState':
        """
        Return canonical form from the first player's perspective.
        In Connect Four, this means flipping the board perspective when player is -1.
        """
        if self.current_player == 1:
            return self
        else:
            # Flip the perspective
            flipped_board = -self.board
            return ConnectFourState(flipped_board, 1)
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Return observation shape for neural network input.
        For AlphaZero-style networks, uses channels-first format.
        """
        return (3, self.ROWS, self.COLS)
    
    def to_key(self) -> str:
        """Efficient key representation for caching"""
        return self.board.tobytes().hex()
    
    def __eq__(self, other) -> bool:
        """Check equality with another state"""
        if not isinstance(other, ConnectFourState):
            return False
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)
    
    def __hash__(self) -> int:
        """Hash function for state"""
        board_tuple = tuple(self.board.flatten().tolist())
        return hash((board_tuple, self.current_player))
    
    def __str__(self) -> str:
        """String representation of the board"""
        symbols = {0: '⚪', 1: '🔴', -1: '🔵'}
        # Format board for display
        rows = []
        for r in range(self.ROWS):
            row = ' '.join(symbols[p] for p in self.board[r])
            rows.append(row)
        
        # Add column numbers at the bottom
        col_numbers = ' '.join(str(i) for i in range(self.COLS))
        rows.append('='*self.COLS*2)  # Separator
        rows.append(col_numbers)
        
        return '\n'.join(rows)
    
    def clone(self) -> 'ConnectFourState':
        """Create a deep copy of this state including move history"""
        clone = ConnectFourState(
            self.board.copy(), 
            self.current_player,
            self._move_history.copy()
        )
        clone._winner = self._winner
        clone._legal_actions = self._legal_actions.copy() if self._legal_actions else None
        clone._last_move = self._last_move
        return clone
        
    def make_move(self, action: int) -> bool:
        """
        Make a temporary move on the board.
        Useful for rule checking without creating new states.
        
        Args:
            action: Column to drop a piece in (0-6 for standard board)
            
        Returns:
            bool: True if move was valid and applied, False otherwise
        """
        # Check if move is valid
        if action < 0 or action >= self.COLS:
            return False
        
        # Find where piece will land
        row = self._get_next_empty_row(action)
        if row is None:
            return False  # Column is full
        
        # Apply move directly to the board
        self.board[row, action] = self._temp_player
        
        # Store move for potential undo
        self._temp_moves.append((row, action, self._temp_player))
        
        # Switch player
        self._temp_player = -self._temp_player
        
        # Clear cached values
        self._winner = None
        self._legal_actions = None
        
        return True
        
    def undo_move(self) -> bool:
        """
        Undo the last temporary move.
        
        Returns:
            bool: True if a move was undone, False if no moves to undo
        """
        if not self._temp_moves:
            return False
            
        # Get last move
        row, col, _ = self._temp_moves.pop()
        
        # Reset board position
        self.board[row, col] = 0
        
        # Restore player
        self._temp_player = -self._temp_player
        
        # Clear cached values
        self._winner = None
        self._legal_actions = None
        
        return True
        
    def get_move_history(self, n: Optional[int] = None) -> List[int]:
        """
        Get the history of moves made in this game.
        
        Args:
            n: Optional number of most recent moves to return (None = all moves)
            
        Returns:
            List[int]: List of column selections in chronological order
        """
        if n is None or n >= len(self._move_history):
            return self._move_history.copy()
        else:
            return self._move_history[-n:]
            
    def calculate_move_distance(self, move1: int, move2: int) -> float:
        """
        Calculate the 'distance' between two moves in Connect Four.
        For Connect Four, the distance is just the number of columns between moves.
        
        Args:
            move1: First column selection (0-6)
            move2: Second column selection (0-6)
            
        Returns:
            float: Distance between columns
        """
        return abs(move1 - move2)
    