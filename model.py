# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, List, Type
import logging

logger = logging.getLogger(__name__)

class GameSpecificModel(nn.Module):
    """
    Base class for game-specific models.
    Each game type can have a specialized neural network architecture.
    """
    
    def __init__(self, game_name: str, policy_size: int, **kwargs):
        super().__init__()
        self.game_name = game_name
        self.policy_size = policy_size
    
    def get_input_shape(self) -> tuple:
        """Get expected input tensor shape"""
        raise NotImplementedError("Subclasses must implement get_input_shape")
    
    def get_output_sizes(self) -> Tuple[int, int]:
        """Get policy and value output sizes"""
        return self.policy_size, 1

class SmallResNet(GameSpecificModel):
    """
    Small ResNet model compatible with TicTacToe and similar games.
    Works with either flat input or 2D board with channels.
    """
    
    def __init__(self, game_name: str = "TicTacToe", policy_size: int = 9, 
                input_channels: int = 3, board_size: int = 3):
        super().__init__(game_name, policy_size)
        
        # Store board parameters
        self.input_channels = input_channels
        self.board_size = board_size
        self.flat_input = False  # Will be set during first forward pass
        
        # Input processing layers
        self.flat_fc = nn.Linear(policy_size, 64)  # For flat input
        self.conv_input = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)  # For 2D input
        
        # Common network body
        self.res_block = nn.Sequential(
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Output heads
        self.policy_head = nn.Linear(64, policy_size)
        self.value_head = nn.Linear(64, 1)
    
    def get_input_shape(self) -> tuple:
        """Get expected input tensor shape"""
        if self.flat_input:
            return (self.policy_size,)
        else:
            return (self.input_channels, self.board_size, self.board_size)
    
    def forward(self, x):
        """
        Forward pass with automatic input shape detection.
        
        Args:
            x: Input tensor (can be flat or 2D+channels)
            
        Returns:
            tuple: (policy, value) predictions
        """
        # Detect input shape on first forward pass
        if not hasattr(self, 'flat_input_detected'):
            if len(x.shape) == 2:
                # (batch_size, features) - flat input
                self.flat_input = True
            elif len(x.shape) == 4:
                # (batch_size, channels, height, width) - 2D input
                self.flat_input = False
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
            self.flat_input_detected = True
            logger.info(f"Detected input format for {self.game_name}: {'flat' if self.flat_input else '2D'}")
        
        # Process input according to detected shape
        if self.flat_input:
            features = F.relu(self.flat_fc(x))
        else:
            # Process 2D input with convolutional layer
            features = F.relu(self.conv_input(x))
            
            # Pool to flat features
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(x.size(0), -1)
            
            # If feature size doesn't match, add a linear layer
            if features.shape[1] != 64:
                features = F.relu(nn.Linear(features.shape[1], 64)(features))
        
        # Apply residual block
        features = features + self.res_block(features)  # Residual connection
        
        # Output heads
        policy = F.softmax(self.policy_head(features), dim=1)
        value = torch.tanh(self.value_head(features))
        
        return policy, value

class Connect4Model(GameSpecificModel):
    """
    ResNet-based model optimized for Connect Four.
    Uses a deeper architecture with more convolutional layers.
    """
    
    def __init__(self, game_name: str = "Connect4", policy_size: int = 7,
                input_channels: int = 3, rows: int = 6, cols: int = 7):
        super().__init__(game_name, policy_size)
        
        # Store board parameters
        self.input_channels = input_channels
        self.rows = rows
        self.cols = cols
        
        # Input layer
        self.conv_input = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(64, 64) for _ in range(3)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * rows * cols, policy_size)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def _make_res_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def get_input_shape(self) -> tuple:
        """Get expected input tensor shape"""
        return (self.input_channels, self.rows, self.cols)
    
    def forward(self, x):
        """
        Forward pass optimized for Connect Four.
        
        Args:
            x: Input tensor (batch_size, channels, rows, cols)
            
        Returns:
            tuple: (policy, value) predictions
        """
        # Check input shape
        if len(x.shape) != 4:
            raise ValueError(f"Connect4Model expects 4D input, got shape {x.shape}")
        
        # Input layer
        x = F.relu(self.conv_input(x))
        
        # Apply residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)  # Add residual connection
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class MultiGameModel(nn.Module):
    """
    Neural network that can handle multiple game types.
    It maintains a dictionary of game-specific models and routes
    inference requests to the appropriate model based on the game type.
    """
    
    def __init__(self):
        super().__init__()
        
        # Dictionary of game-specific models
        self.game_models = nn.ModuleDict()
        
        # Default game type
        self.default_game = None
    
    def add_game_model(self, model: GameSpecificModel, game_name: Optional[str] = None):
        """
        Add a model for a specific game type.
        
        Args:
            model: Game-specific model
            game_name: Game identifier (defaults to model.game_name)
        """
        if game_name is None:
            game_name = model.game_name
        
        # Add to module dictionary
        self.game_models[game_name] = model
        
        # Set as default if first model
        if self.default_game is None:
            self.default_game = game_name
    
    def get_game_model(self, game_name: Optional[str] = None) -> GameSpecificModel:
        """
        Get the model for a specific game.
        
        Args:
            game_name: Game identifier (defaults to default_game)
            
        Returns:
            GameSpecificModel: Model for the specified game
        """
        if game_name is None:
            game_name = self.default_game
        
        if game_name not in self.game_models:
            if self.default_game is not None:
                logger.warning(f"Game model for '{game_name}' not found, using default '{self.default_game}'")
                return self.game_models[self.default_game]
            else:
                raise ValueError(f"No model found for game '{game_name}' and no default model set")
        
        return self.game_models[game_name]
    
    def forward(self, x, game_name: Optional[str] = None):
        """
        Forward pass that routes to the appropriate game model.
        
        Args:
            x: Input tensor
            game_name: Game identifier (defaults to default_game)
            
        Returns:
            tuple: (policy, value) predictions
        """
        model = self.get_game_model(game_name)
        return model(x)
    
    def create_inference_preprocessor(self, game_name: Optional[str] = None) -> callable:
        """
        Create a preprocessor function for the specified game model.
        The preprocessor converts game states to model inputs.
        
        Args:
            game_name: Game identifier (defaults to default_game)
            
        Returns:
            callable: Function that takes a game state and returns a model input tensor
        """
        model = self.get_game_model(game_name)
        input_shape = model.get_input_shape()
        
        def preprocessor(state):
            """Convert a game state to model input format"""
            # Use game state's built-in encoding methods
            if hasattr(state, 'encode_for_inference'):
                encoded = state.encode_for_inference()
            else:
                encoded = state.encode()
            
            # Convert to tensor
            return torch.tensor(encoded, dtype=torch.float32)
        
        return preprocessor

# Factory function to create appropriate model for a game
def create_model_for_game(game_name: str, **kwargs) -> GameSpecificModel:
    """
    Create an appropriate neural network model for a specific game type.
    
    Args:
        game_name: Game identifier
        **kwargs: Additional parameters for model initialization
        
    Returns:
        GameSpecificModel: Appropriate model for the game
    """
    from utils.game_registry import GameRegistry
    
    # Get game parameters from registry
    policy_size = GameRegistry.get_game_param(game_name, 'policy_size', 9)
    
    if game_name == "Connect4":
        rows = GameRegistry.get_game_param(game_name, 'rows', 6)
        cols = GameRegistry.get_game_param(game_name, 'cols', 7)
        return Connect4Model(
            game_name=game_name,
            policy_size=policy_size,
            rows=rows,
            cols=cols,
            **kwargs
        )
    else:
        # Default to SmallResNet for other games
        board_size = GameRegistry.get_game_param(game_name, 'board_size', 3)
        return SmallResNet(
            game_name=game_name,
            policy_size=policy_size,
            board_size=board_size,
            **kwargs
        )

# Create a multi-game model with common games
def create_multi_game_model() -> MultiGameModel:
    """
    Create a multi-game model with common games pre-registered.
    
    Returns:
        MultiGameModel: Model that supports multiple games
    """
    model = MultiGameModel()
    
    # Add models for common games
    model.add_game_model(create_model_for_game("TicTacToe"))
    model.add_game_model(create_model_for_game("Connect4"))
    
    return model