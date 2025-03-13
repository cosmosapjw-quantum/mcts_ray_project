# mcts/search.py
import ray
from mcts.tree import select, expand, backpropagate
from mcts.node import Node

@ray.remote
def mcts_worker(root_state, inference_actor, num_simulations):
    root = Node(root_state)
    for _ in range(num_simulations):
        node = select(root)
        if node.state.is_terminal():
            value = node.state.winner
        else:
            priors, value = ray.get(inference_actor.infer.remote(node.state))
            expand(node, priors)  # pass priors directly
        backpropagate(node, value)
    return root