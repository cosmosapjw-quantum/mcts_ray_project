# config.py
import os
import json
import yaml
import argparse
import torch
import multiprocessing
import logging
import logging.config

# Set up logging
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'training.log',
            'formatter': 'standard',
            'level': 'DEBUG',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
})

logger = logging.getLogger('config')

# Default configuration
DEFAULT_CONFIG = {
    # Ray settings
    'ray': {
        'address': 'auto',
        'num_cpus': None,  # Auto-detect
        'num_gpus': None,  # Auto-detect
        'object_store_memory': None,  # Auto-detect
        'redis_max_memory': 1024 * 1024 * 1000,  # 1GB
    },
    
    # MCTS settings
    'mcts': {
        'base_simulations': 20,
        'simulations_per_worker': 30,
        'num_workers': 8,
        'exploration_weight': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'temperature_schedule': {
            '0': 1.0,
            '500': 0.8,
            '2000': 0.5,
            '5000': 0.3,
            '10000': 0.1
        },
        'virtual_loss': 0.1,
    },
    
    # Training settings
    'training': {
        'batch_size': 1024,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'num_epochs': 1,
        'replay_buffer_size': 100000,
        'min_buffer_size': 10000,
        'games_per_generation': 1000,
        'checkpoint_interval': 100,
        'eval_interval': 500,
        'checkpoint_dir': 'checkpoints',
        'tensorboard_dir': 'runs',
        'mixed_precision': True,
    },
    
    # Model settings
    'model': {
        'type': 'small_resnet',
        'hidden_size': 64,
        'num_res_blocks': 1,
    },
    
    # Inference server settings
    'inference': {
        'batch_wait': 0.02,
        'cache_size': 10000,
        'max_batch_size': 512,
        'gpu_fraction': 0.5,
    },
    
    # General settings
    'general': {
        'verbose': False,
        'random_seed': 42,
        'num_games': 10000,
        'max_game_length': 100,
    }
}

class Config:
    """Configuration manager that handles loading from files, CLI, and environment"""
    
    def __init__(self):
        self._config = DEFAULT_CONFIG.copy()
        self._parsed = False
        
    def parse_args(self):
        """Parse command line arguments"""
        if self._parsed:
            return
            
        parser = argparse.ArgumentParser(description='AlphaZero-style Training')
        
        # General arguments
        parser.add_argument('--config', type=str, help='Path to config file (yaml or json)')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        parser.add_argument('--seed', type=int, help='Random seed')
        parser.add_argument('--num-games', type=int, help='Number of games to play')
        
        # Ray arguments
        parser.add_argument('--ray-address', type=str, help='Ray address')
        parser.add_argument('--num-cpus', type=int, help='Number of CPUs to use')
        parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use')
        
        # MCTS arguments
        parser.add_argument('--simulations', type=int, help='Base simulations per batch')
        parser.add_argument('--sims-per-worker', type=int, help='Simulations per worker')
        parser.add_argument('--num-workers', type=int, help='Number of workers')
        parser.add_argument('--exploration', type=float, help='Exploration weight')
        
        # Training arguments
        parser.add_argument('--batch-size', type=int, help='Training batch size')
        parser.add_argument('--lr', type=float, help='Learning rate')
        parser.add_argument('--epochs', type=int, help='Epochs per batch')
        
        # Inference arguments
        parser.add_argument('--batch-wait', type=float, help='Inference batch wait time')
        parser.add_argument('--gpu-fraction', type=float, help='GPU fraction for inference')
        
        # Parse arguments
        args = parser.parse_args()
        
        # Load config from file if specified
        if args.config:
            self.load_from_file(args.config)
        
        # Update config with command line arguments
        if args.verbose is not None:
            self._config['general']['verbose'] = args.verbose
            
        if args.seed is not None:
            self._config['general']['random_seed'] = args.seed
            
        if args.num_games is not None:
            self._config['general']['num_games'] = args.num_games
            
        if args.ray_address is not None:
            self._config['ray']['address'] = args.ray_address
            
        if args.num_cpus is not None:
            self._config['ray']['num_cpus'] = args.num_cpus
            
        if args.num_gpus is not None:
            self._config['ray']['num_gpus'] = args.num_gpus
            
        if args.simulations is not None:
            self._config['mcts']['base_simulations'] = args.simulations
            
        if args.sims_per_worker is not None:
            self._config['mcts']['simulations_per_worker'] = args.sims_per_worker
            
        if args.num_workers is not None:
            self._config['mcts']['num_workers'] = args.num_workers
            
        if args.exploration is not None:
            self._config['mcts']['exploration_weight'] = args.exploration
            
        if args.batch_size is not None:
            self._config['training']['batch_size'] = args.batch_size
            
        if args.lr is not None:
            self._config['training']['learning_rate'] = args.lr
            
        if args.epochs is not None:
            self._config['training']['num_epochs'] = args.epochs
            
        if args.batch_wait is not None:
            self._config['inference']['batch_wait'] = args.batch_wait
            
        if args.gpu_fraction is not None:
            self._config['inference']['gpu_fraction'] = args.gpu_fraction
            
        # Load from environment variables
        self.load_from_env()
        
        # Auto-detect values for unspecified parameters
        self.auto_detect()
        
        self._parsed = True
        logger.info("Configuration loaded and parsed")
        
    def load_from_file(self, file_path):
        """Load configuration from a file (YAML or JSON)"""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
                    
                # Update config with file values
                self._update_nested_dict(self._config, file_config)
                logger.info(f"Loaded configuration from {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Environment variables should be in the format AZ_SECTION_KEY=value
        # For example: AZ_MCTS_NUM_WORKERS=12
        
        prefix = "AZ_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                parts = key[len(prefix):].lower().split('_')
                if len(parts) >= 2:
                    section = parts[0]
                    param = '_'.join(parts[1:])
                    
                    if section in self._config and param in self._config[section]:
                        # Convert value to the appropriate type
                        current_value = self._config[section][param]
                        if isinstance(current_value, bool):
                            self._config[section][param] = value.lower() in ('true', '1', 'yes')
                        elif isinstance(current_value, int):
                            self._config[section][param] = int(value)
                        elif isinstance(current_value, float):
                            self._config[section][param] = float(value)
                        elif isinstance(current_value, dict):
                            try:
                                self._config[section][param] = json.loads(value)
                            except:
                                logger.warning(f"Could not parse environment variable {key} as JSON")
                        else:
                            self._config[section][param] = value
                            
                        logger.debug(f"Set {section}.{param} = {self._config[section][param]} from environment")
                        
    def auto_detect(self):
        """Auto-detect values for unspecified parameters"""
        # Detect CPU count if not specified
        if self._config['ray']['num_cpus'] is None:
            self._config['ray']['num_cpus'] = max(1, multiprocessing.cpu_count() - 1)
            logger.info(f"Auto-detected {self._config['ray']['num_cpus']} CPUs")
            
        # Detect GPU count if not specified
        if self._config['ray']['num_gpus'] is None and torch.cuda.is_available():
            self._config['ray']['num_gpus'] = torch.cuda.device_count()
            logger.info(f"Auto-detected {self._config['ray']['num_gpus']} GPUs")
        elif self._config['ray']['num_gpus'] is None:
            self._config['ray']['num_gpus'] = 0
            logger.info("No GPUs detected")
            
        # Adjust object store size based on system memory
        if self._config['ray']['object_store_memory'] is None:
            try:
                import psutil
                system_memory = psutil.virtual_memory().total
                self._config['ray']['object_store_memory'] = int(system_memory * 0.3)  # 30% of system memory
                logger.info(f"Auto-configured object store memory: {self._config['ray']['object_store_memory'] / (1024**3):.1f} GB")
            except ImportError:
                # Default to 4GB if psutil is not available
                self._config['ray']['object_store_memory'] = 4 * 1024 * 1024 * 1024
                logger.info("psutil not available, defaulting to 4GB object store")
                
        # Adjust worker count based on CPU count
        cpu_count = self._config['ray']['num_cpus']
        if self._config['mcts']['num_workers'] > cpu_count:
            self._config['mcts']['num_workers'] = max(1, cpu_count - 1)
            logger.info(f"Adjusted worker count to {self._config['mcts']['num_workers']} based on CPU count")
            
        # Set mixed precision based on GPU availability
        self._config['training']['mixed_precision'] = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
        
    def _update_nested_dict(self, d, u):
        """Recursively update a nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
                
    def get(self, section, key=None):
        """Get a configuration value"""
        if not self._parsed:
            self.parse_args()
            
        if key is None:
            return self._config.get(section, {})
        else:
            return self._config.get(section, {}).get(key)
            
    def set(self, section, key, value):
        """Set a configuration value"""
        if section not in self._config:
            self._config[section] = {}
            
        self._config[section][key] = value
        
    def save(self, file_path):
        """Save configuration to a file"""
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(file_path, 'w') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(self._config, f, default_flow_style=False)
                elif file_path.endswith('.json'):
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
                    
            logger.info(f"Saved configuration to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")
            
    def __str__(self):
        """String representation of the configuration"""
        return yaml.dump(self._config, default_flow_style=False)
        
# Create a singleton config instance
CONFIG = Config()

# For backward compatibility with existing code
RAY_ADDRESS = CONFIG.get('ray', 'address')
VERBOSE = CONFIG.get('general', 'verbose')
NUM_SIMULATIONS = CONFIG.get('mcts', 'base_simulations')
SIMULATIONS_PER_WORKER = CONFIG.get('mcts', 'simulations_per_worker')
NUM_WORKERS = CONFIG.get('mcts', 'num_workers')

# Return the config object when the module is imported
def get_config():
    return CONFIG