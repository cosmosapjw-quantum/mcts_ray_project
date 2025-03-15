# utils/game_interface.py

from abc import ABC, abstractmethod

class GameState(ABC):
    @abstractmethod
    def is_terminal(self) -> bool:
        """Checks if the state is terminal (game ended)."""
        pass

    @abstractmethod
    def get_legal_actions(self) -> list:
        """Returns a list of available legal actions."""
        pass

    @abstractmethod
    def apply_action(self, action) -> 'GameState':
        """Returns the next state after applying the action."""
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        """Returns the current player's identifier."""
        pass

    @abstractmethod
    def encode(self):
        """Encodes the current state into model input format."""
        pass

    @abstractmethod
    def get_winner(self):
        """Returns the winner if the game ended, else None."""
        pass
