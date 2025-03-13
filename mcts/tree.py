# mcts/tree.py
from mcts.node import Node

def select(node):
    while node.children:
        node = max(node.children, key=lambda n: n.value/(1+n.visits))
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