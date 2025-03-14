import ray
from utils.state_utils import TicTacToeState
from mcts.search import mcts_worker
from inference.inference_server import InferenceServer
from config import NUM_SIMULATIONS, NUM_WORKERS

ray.init()
inference_actor = InferenceServer.remote()

def evaluate_model(num_games=50):
    wins, losses, draws = 0, 0, 0
    for _ in range(num_games):
        state = TicTacToeState()

        while not state.is_terminal():
            root = ray.get(mcts_worker.remote(state, inference_actor, NUM_SIMULATIONS))
            visits = [child.visits for child in root.children]
            action = root.children[visits.index(max(visits))].state
            state = action

        if state.winner == 1:
            wins += 1
        elif state.winner == -1:
            losses += 1
        else:
            draws += 1

    print(f"Evaluation over {num_games} games:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win rate: {wins / num_games:.2%}")

evaluate_model()
ray.shutdown()
