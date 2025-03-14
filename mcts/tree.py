# mcts/tree.py
from numba import jit
import numpy as np
from mcts.node import Node

@jit(nopython=True, fastmath=True)
def compute_uct(values, visits, total_visits, c=1.4):
    uct_values = np.empty(len(values))
    for i in range(len(values)):
        if visits[i] == 0:
            uct_values[i] = float('inf')
        else:
            uct_values[i] = (values[i] / visits[i]) + c * np.sqrt(np.log(total_visits) / visits[i])
    return uct_values

def select(node):
    while node.children:
        # Ensure values and visits are explicitly scalar
        values = np.array([float(child.value) for child in node.children], dtype=np.float32)
        visits = np.array([child.visits for child in node.children], dtype=np.int32)
        total_visits = sum(visits)

        uct_values = compute_uct(values, visits, total_visits)
        best_idx = np.argmax(uct_values)
        node = node.children[best_idx]
    return node

def expand(node, priors):
    actions = node.state.get_legal_actions()
    priors_for_actions = [priors[a] for a in actions]  # align priors with actions
    for action, prior in zip(actions, priors):
        child = Node(node.state.apply_action(action), node)
        child.prior = prior
        node.children.append(child)

def backpropagate(node, value):
    while node:
        node.visits += 1
        node.value += value
        value = -value
        node = node.parent