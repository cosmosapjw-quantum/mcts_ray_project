# profile_mcts.py
import cProfile
import ray
from inference.inference_server import InferenceServer
from utils.state_utils import TicTacToeState
from mcts.search import mcts_worker
from config import NUM_SIMULATIONS, NUM_WORKERS

ray.init()
inference_actor = InferenceServer.remote(batch_wait=0.02)
state = TicTacToeState()

profiler = cProfile.Profile()
profiler.enable()

ray.get([
            mcts_worker.remote(state, inference_actor, NUM_SIMULATIONS)
            for _ in range(NUM_WORKERS)
        ])

profiler.disable()
profiler.dump_stats("mcts_profile.stats")
ray.shutdown()