# main.py
import ray
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from inference.inference_server import InferenceServer
from utils.state_utils import TicTacToeState
from mcts.search import mcts_worker
from model import SmallResNet
from config import NUM_SIMULATIONS, NUM_WORKERS, SIMULATIONS_PER_WORKER, VERBOSE

ray.init()

# Initialize model
model = SmallResNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Launch inference server
inference_actor = InferenceServer.remote(batch_wait=0.05)

writer = SummaryWriter(log_dir='runs/experiment1')

def prev_board_differs(old_board, new_board, index):
    return old_board[index] != new_board[index]

# Self-play loop
def self_play_episode(episode_num):
    state = TicTacToeState()
    memory = []

    while not state.is_terminal():
        roots = ray.get([
            mcts_worker.remote(state, inference_actor, NUM_SIMULATIONS * SIMULATIONS_PER_WORKER)
            for _ in range(NUM_WORKERS)
        ])

        # Aggregate visit counts across workers
        combined_visits = torch.zeros(9)
        for root in roots:
            for child in root.children:
                action = [i for i, (prev, new) in enumerate(zip(state.board, child.state.board)) if prev_board_differs(state.board, child.state.board, i)]
                combined_action = action[0] if action else None
                if combined_action is not None:
                    combined_visits[combined_action] += child.visits

        policy = combined_visits / combined_visits.sum()

        if VERBOSE:
            print(f"State: {state.board}")
            print(f"Batched policy distribution: {policy}")

        action = torch.multinomial(policy, 1).item()
        state_tensor = torch.tensor(state.board).float()

        memory.append((state_tensor, policy, state.current_player))

        if VERBOSE:
            print(f"Chosen action: {action}")

        state = state.apply_action(action)

    outcome = state.winner

    if VERBOSE:
        print(f"Game outcome: {outcome}")

    total_loss = 0
    for state_tensor, policy, player in memory:
        target_value = torch.tensor([1.0 if outcome == player else -1.0 if outcome else 0])
        predicted_policy, predicted_value = model(state_tensor)

        loss_policy = -(policy * predicted_policy.log()).sum()
        loss_value = (predicted_value - target_value) ** 2
        loss = loss_policy + loss_value
        total_loss += loss.item()

        if VERBOSE:
            print(f"Policy loss: {loss_policy.item()}, Value loss: {loss_value.item()}, Total loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    writer.add_scalar('Loss/Total', total_loss / len(memory), episode_num)
    writer.add_scalar('Game Outcome', outcome, episode_num)

# Run self-play episodes
for episode in range(5):
    if VERBOSE:
        print(f"Starting episode {episode + 1}")
    self_play_episode(episode)
    
writer.close()

ray.shutdown()