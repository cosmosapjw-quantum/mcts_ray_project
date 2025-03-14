# mcts/tree.py
from numba import jit
import numpy as np
from mcts.node import Node

@jit(nopython=True, fastmath=True)
def compute_puct(values, visits, priors, total_visits, c=1.0):
    puct_values = np.empty(len(values))
    for i in range(len(values)):
        q_value = values[i] / visits[i] if visits[i] > 0 else 0
        u_value = c * priors[i] * np.sqrt(total_visits) / (1 + visits[i])
        puct_values[i] = q_value + u_value
    return puct_values

def select(node):
    while node.children:
        # Ensure values and visits are explicitly scalar
        values = np.array([float(child.value) for child in node.children], dtype=np.float32)
        visits = np.array([child.visits for child in node.children], dtype=np.int32)
        priors = np.array([child.prior for child in node.children], dtype=np.int32)
        total_visits = sum(visits)

        uct_values = compute_puct(values, visits, priors, total_visits)
        best_idx = np.argmax(uct_values)
        node = node.children[best_idx]
    return node

def expand(node, priors, alpha=0.3, epsilon=0.25):
    actions = node.state.get_legal_actions()
    noise = np.random.dirichlet([alpha] * len(actions))
    for idx, action in enumerate(actions):
        noisy_prior = (1 - epsilon) * priors[idx] + epsilon * noise[idx]
        child_state = node.state.apply_action(action)
        child_node = Node(child_state, node)
        child_node.prior = noisy_prior
        node.children.append(child_node)

def backpropagate(node, value):
    while node:
        node.visits += 1
        node.value += value
        value = -value
        node = node.parent