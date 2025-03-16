# utils/game_registry.py
from typing import Dict, Type, Any, Optional, List, Callable, Tuple
import importlib
import logging
from utils.game_interface import GameState

logger = logging.getLogger(__name__)

class GameRegistry:
    """
    Registry for managing game implementations and their properties.
    This provides a central place to look up game-specific parameters and create instances.
    """
    
    # Registry of game implementations
    _registry: Dict[str, Type[GameState]] = {}
    
    # Game-specific parameters
    _game_params: Dict[str, Dict[str, Any]] = {}
    
    # Factory functions for creating game states
    _factories: Dict[str, Callable[..., GameState]] = {}
    
    @classmethod
    def register_game(cls, game_class: Type[GameState], game_id: Optional[str] = None) -> None:
        """
        Register a game implementation with the registry.
        
        Args:
            game_class: GameState implementation class
            game_id: Optional identifier (defaults to class name)
        """
        # Get game ID from class if not provided
        if game_id is None:
            if hasattr(game_class, 'GAME_NAME'):
                game_id = game_class.GAME_NAME
            else:
                game_id = game_class.__name__
        
        # Register the game
        cls._registry[game_id] = game_class
        
        # Initialize game parameters
        if game_id not in cls._game_params:
            cls._game_params[game_id] = {}
        
        # Add basic parameters
        params = cls._game_params[game_id]
        
        # Get policy size from class constant or create instance to determine
        if hasattr(game_class, 'POLICY_SIZE'):
            params['policy_size'] = game_class.POLICY_SIZE
        else:
            # Create temporary instance to get policy size
            try:
                temp_instance = game_class()
                params['policy_size'] = temp_instance.policy_size
            except Exception as e:
                logger.warning(f"Could not determine policy size for {game_id}: {e}")
                params['policy_size'] = None  # Will be determined later
        
        logger.info(f"Registered game: {game_id} with policy size: {params.get('policy_size')}")
    
    @classmethod
    def register_game_params(cls, game_id: str, **params) -> None:
        """
        Register game-specific parameters.
        
        Args:
            game_id: Game identifier
            **params: Parameters to register
        """
        if game_id not in cls._game_params:
            cls._game_params[game_id] = {}
        
        # Update parameters
        cls._game_params[game_id].update(params)
        logger.debug(f"Updated parameters for {game_id}: {params}")
    
    @classmethod
    def register_factory(cls, game_id: str, factory_func: Callable[..., GameState]) -> None:
        """
        Register a factory function for creating game states.
        
        Args:
            game_id: Game identifier
            factory_func: Factory function that creates game states
        """
        cls._factories[game_id] = factory_func
        logger.info(f"Registered factory for {game_id}")
    
    @classmethod
    def create_game(cls, game_id: str, **kwargs) -> GameState:
        """
        Create a game state instance using the registered class or factory.
        
        Args:
            game_id: Game identifier
            **kwargs: Parameters to pass to the constructor or factory
            
        Returns:
            GameState: New game state instance
        """
        # Check if we have a factory
        if game_id in cls._factories:
            return cls._factories[game_id](**kwargs)
        
        # Check if we have a registered class
        if game_id in cls._registry:
            return cls._registry[game_id](**kwargs)
        
        # Try import from module
        try:
            # Try common locations
            for module_path in [
                f"games.{game_id.lower()}",
                f"utils.{game_id.lower()}_state",
                f"utils.{game_id.lower()}"
            ]:
                try:
                    module = importlib.import_module(module_path)
                    # Find class that matches the game ID
                    for attr_name in dir(module):
                        if attr_name.lower() == game_id.lower() or attr_name.lower() == f"{game_id.lower()}state":
                            game_class = getattr(module, attr_name)
                            if isinstance(game_class, type) and issubclass(game_class, GameState):
                                # Register for future use
                                cls.register_game(game_class, game_id)
                                return game_class(**kwargs)
                except ImportError:
                    continue
        except Exception as e:
            logger.error(f"Error importing {game_id}: {e}")
        
        raise ValueError(f"Game '{game_id}' not found in registry or module system")
    
    @classmethod
    def get_policy_size(cls, game_id: str) -> int:
        """
        Get the policy size for a registered game.
        
        Args:
            game_id: Game identifier
            
        Returns:
            int: Policy size
        """
        # Check if we have registered parameters
        if game_id in cls._game_params and 'policy_size' in cls._game_params[game_id]:
            return cls._game_params[game_id]['policy_size']
        
        # Try to create an instance to determine
        try:
            game = cls.create_game(game_id)
            policy_size = game.policy_size
            
            # Register for future use
            cls.register_game_params(game_id, policy_size=policy_size)
            
            return policy_size
        except Exception as e:
            logger.error(f"Could not determine policy size for {game_id}: {e}")
            raise ValueError(f"Policy size unknown for game '{game_id}'")
    
    @classmethod
    def get_game_param(cls, game_id: str, param_name: str, default: Any = None) -> Any:
        """
        Get a game parameter.
        
        Args:
            game_id: Game identifier
            param_name: Parameter name
            default: Default value if parameter doesn't exist
            
        Returns:
            Parameter value or default
        """
        if game_id in cls._game_params and param_name in cls._game_params[game_id]:
            return cls._game_params[game_id][param_name]
        return default
    
    @classmethod
    def list_games(cls) -> List[str]:
        """
        Get list of registered game IDs.
        
        Returns:
            List[str]: Registered game IDs
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_game_info(cls, game_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a registered game.
        
        Args:
            game_id: Game identifier
            
        Returns:
            Dict: Game information
        """
        game_class = cls._registry.get(game_id)
        params = cls._game_params.get(game_id, {})
        
        info = {
            "id": game_id,
            "class": game_class.__name__ if game_class else None,
            "has_factory": game_id in cls._factories,
            **params
        }
        
        return info


# Register TicTacToe automatically
from utils.state_utils import TicTacToeState
GameRegistry.register_game(TicTacToeState, "TicTacToe")
GameRegistry.register_game_params("TicTacToe", 
                                 board_size=3, 
                                 default_player=1,
                                 observation_shape=(3, 3, 3))

# Register the Connect Four game with the game registry
from utils.state_utils import ConnectFourState
GameRegistry.register_game(ConnectFourState, "Connect4")
GameRegistry.register_game_params("Connect4", 
                                 rows=6,
                                 cols=7,
                                 policy_size=7,
                                 observation_shape=(3, 6, 7))