# utils/game_interface.py

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Any, Dict
import numpy as np

class GameState(ABC):
    """
    Enhanced abstract base class for game states with improved abstraction.
    This interface provides a comprehensive, game-agnostic API for MCTS.
    
    Key features:
    - Standard methods for MCTS (is_terminal, get_legal_actions, etc.)
    - Move history tracking for improved neural network context
    - Temporary move application for rule checking
    - Game-specific metadata and parameter access
    """
    
    @property
    @abstractmethod
    def policy_size(self) -> int:
        """
        Return the size of the policy vector required for this game.
        This corresponds to the total number of possible actions in the game.
        """
        pass
    
    @property
    def game_name(self) -> str:
        """
        Return a string identifier for this game type.
        Default implementation returns the class name.
        """
        return self.__class__.__name__
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if the state is terminal (game ended)."""
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[int]:
        """Return a list of available legal actions."""
        pass

    @abstractmethod
    def apply_action(self, action: int) -> 'GameState':
        """Return the next state after applying the action."""
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        """
        Return the current player's identifier.
        Typically 1 for first player and -1 for second player.
        """
        pass

    @abstractmethod
    def encode(self) -> np.ndarray:
        """
        Encode the current state into neural network input format.
        This is a basic encoding, possibly just flattening the board.
        """
        pass
    
    def encode_for_inference(self) -> np.ndarray:
        """
        Encode the state for neural network inference with a standardized format.
        Override this method for more sophisticated game-specific encoding.
        Default implementation calls the simple encode() method.
        """
        return self.encode()
    
    @abstractmethod
    def get_winner(self) -> Optional[int]:
        """
        Return the winner identifier if the game ended, else None.
        For two-player games, typically returns 1, -1, or 0 (draw).
        """
        pass
    
    def get_canonical_state(self) -> 'GameState':
        """
        Return a canonicalized version of this state.
        This is useful for symmetry handling in board games.
        Default implementation returns self (no canonicalization).
        """
        return self
    
    def get_symmetries(self) -> List[Tuple['GameState', List[float]]]:
        """
        Return a list of equivalent states through symmetries.
        Each item is a tuple of (symmetric_state, action_mapping).
        Default implementation returns just this state with identity mapping.
        """
        return [(self, list(range(self.policy_size)))]
    
    def to_key(self) -> str:
        """
        Return a string key representation for caching.
        Default implementation uses the string representation of the state.
        Override for more efficient key generation.
        """
        return str(self)
    
    def __hash__(self) -> int:
        """
        Hash function for state. Default implementation uses to_key().
        Override for more efficient hashing.
        """
        return hash(self.to_key())
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the observation tensor for this game.
        This is useful for initializing neural network input layers.
        Default implementation derives shape from encode() result.
        """
        encoded = self.encode()
        return encoded.shape
    
    def get_action_mask(self) -> np.ndarray:
        """
        Return a binary mask for valid actions (1 = legal, 0 = illegal).
        Useful for masking illegal actions in policy outputs.
        """
        mask = np.zeros(self.policy_size, dtype=np.float32)
        for action in self.get_legal_actions():
            mask[action] = 1.0
        return mask
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return any additional game-specific metadata.
        Useful for custom game features or debugging.
        """
        return {
            "game_name": self.game_name,
            "policy_size": self.policy_size,
            "player": self.get_current_player()
        }