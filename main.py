# main.py
import ray
import torch
import torch.optim as optim
from inference.inference_server import InferenceServer
from utils.state_utils import TicTacToeState
from mcts.search import mcts_worker
from model import SmallResNet
from config import NUM_SIMULATIONS, NUM_WORKERS, VERBOSE

ray.init()

# Initialize model
model = SmallResNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Launch inference server
inference_actor = InferenceServer.remote()

# Self-play loop
def self_play_episode():
    state = TicTacToeState()
    memory = []

    while not state.is_terminal():
        root = ray.get(mcts_worker.remote(state, inference_actor, NUM_SIMULATIONS))
        visits = torch.tensor([child.visits for child in root.children])

        full_policy = torch.zeros(9)
        legal_actions = state.get_legal_actions()
        full_policy[legal_actions] = visits.float() / visits.sum()
        policy = full_policy

        if VERBOSE:
            print(f"State: {state.board}")
            print(f"Policy distribution: {policy}")

        action = torch.multinomial(policy, 1).item()
        state_tensor = torch.tensor(state.board).float()

        memory.append((state_tensor, policy, state.current_player))

        if VERBOSE:
            print(f"Chosen action: {action}")

        state = state.apply_action(action)

    outcome = state.winner

    if VERBOSE:
        print(f"Game outcome: {outcome}")

    for state_tensor, policy, player in memory:
        target_value = torch.tensor([1.0 if outcome == player else -1.0 if outcome else 0])
        predicted_policy, predicted_value = model(state_tensor)

        loss_policy = -(policy * predicted_policy.log()).sum()
        loss_value = (predicted_value - target_value) ** 2
        loss = loss_policy + loss_value

        if VERBOSE:
            print(f"Policy loss: {loss_policy.item()}, Value loss: {loss_value.item()}, Total loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Run self-play episodes
for episode in range(10):
    if VERBOSE:
        print(f"Starting episode {episode + 1}")
    self_play_episode()

ray.shutdown()