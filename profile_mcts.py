# profile_mcts.py
import cProfile
import ray
from inference.inference_server import InferenceServer
from utils.state_utils import TicTacToeState
from mcts.search import mcts_worker
from config import NUM_SIMULATIONS

ray.init()
inference_actor = InferenceServer.remote()
state = TicTacToeState()

profiler = cProfile.Profile()
profiler.enable()

ray.get(mcts_worker.remote(state, inference_actor, NUM_SIMULATIONS))

profiler.disable()
profiler.dump_stats("mcts_profile.stats")
ray.shutdown()