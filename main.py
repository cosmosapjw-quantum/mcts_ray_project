# main.py
import ray
from config import *
from utils.state_utils import TicTacToeState
from inference.inference_server import InferenceServer
from mcts.search import mcts_worker

ray.init()
inference_actor = InferenceServer.remote()

initial_state = TicTacToeState()

futures = [mcts_worker.remote(initial_state, inference_actor, NUM_SIMULATIONS) for _ in range(NUM_WORKERS)]
roots = ray.get(futures)

for i, root in enumerate(roots):
    print(f'Worker {i}: Child visit counts:', [child.visits for child in root.children])

ray.shutdown()