import os
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from inference.inference_server import InferenceServer
from utils.state_utils import TicTacToeState
from mcts.search import mcts_worker
from model import SmallResNet
from config import NUM_SIMULATIONS, NUM_WORKERS
import torch.optim as optim
import torch

# Create experiment directory if it doesn't exist
experiment_dir = os.path.abspath("./experiment")
os.makedirs(experiment_dir, exist_ok=True)

# Disable TensorBoard logging via environment variable
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

# Initialize Ray
ray.init(
    num_cpus=os.cpu_count() - 2,   # Reserve 2 cores for system processes
    num_gpus=1 if torch.cuda.is_available() else 0,
    object_store_memory=4 * 10**9  # 4GB for object store
)

def training_loop(config):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SmallResNet()
    model = model.to(device)  # Move model to GPU if available
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Create inference server actor with GPU if needed
    inference_actor = InferenceServer.remote(
        batch_wait=config["batch_wait"]
    )

    total_loss = 0
    for episode in range(config["episodes"]):
        state = TicTacToeState()
        memory = []

        while not state.is_terminal():
            # Use mcts_worker with resource management
            root = ray.get(mcts_worker.remote(
                state, 
                inference_actor, 
                config["simulations_per_worker"]
            ))
            visits = torch.tensor([child.visits for child in root.children], device=device)

            full_policy = torch.zeros(9, device=device)
            legal_actions = state.get_legal_actions()
            full_policy[legal_actions] = visits.float() / visits.sum()
            policy = full_policy

            action = torch.multinomial(policy, 1).item()
            state_tensor = torch.tensor(state.board, device=device).float()

            memory.append((state_tensor, policy, state.current_player))
            state = state.apply_action(action)

        outcome = state.winner
        episode_loss = 0
        for state_tensor, policy, player in memory:
            target_value = torch.tensor([1.0 if outcome == player else -1.0 if outcome else 0], device=device)
            predicted_policy, predicted_value = model(state_tensor)

            loss_policy = -(policy * predicted_policy.log()).sum()
            loss_value = (predicted_value - target_value) ** 2
            loss = loss_policy + loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()
        
        total_loss += episode_loss
        
        # Instead of tune.report, use the trainable API
        result = {"mean_loss": total_loss / len(memory)}
        
        # Return the result to Ray Tune (compatible with older Ray versions)
        return result

search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_wait": tune.uniform(0.005, 0.05),
    "simulations_per_worker": tune.choice([10, 20, 30]),
    "episodes": 5
    }

scheduler = ASHAScheduler(
    max_t=10,
    grace_period=1,
    reduction_factor=2,
    metric="mean_loss",
    mode="min"
)

reporter = CLIReporter(
    metric_columns=["mean_loss", "training_iteration"]
)

# Calculate resource allocation with GPU

# Number of concurrent trials (using about half your logical cores)
num_concurrent_trials = 5  # Run 5 trials simultaneously

# For a GPU setup, we might allocate the GPU to the inference server
has_gpu = torch.cuda.is_available()

# Resource allocation per trial
pg_factory = tune.PlacementGroupFactory([
    {"CPU": 1},               # Trial runner process
    {"CPU": 1, "GPU": 0.2},   # Inference server with partial GPU
    {"CPU": 2}                # Multiple MCTS workers sharing cores
])

# Create a Trainable class to avoid using tune.report()
class MCTSTrainable(tune.Trainable):
    def setup(self, config):
        self.config = config
        
    def step(self):
        return training_loop(self.config)

result = tune.run(
    MCTSTrainable,
    storage_path=experiment_dir,
    name="my_experiment",
    config=search_space,
    num_samples=50,
    scheduler=scheduler,
    progress_reporter=reporter,
    log_to_file=True,
    resources_per_trial=pg_factory,
    max_concurrent_trials=num_concurrent_trials,  # Control parallelism
)

best_trial = result.get_best_trial("mean_loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
ray.shutdown()