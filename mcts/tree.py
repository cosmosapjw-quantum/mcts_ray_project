# mcts/tree.py
from mcts.node import Node

def select(node):
    while node.children:
        node = max(node.children, key=lambda n: n.value/(1+n.visits))
    return node

def expand(node, priors):
    for action, prior in zip(node.state.get_legal_actions(), priors):
        child = Node(node.state.apply_action(action), node)
        child.prior = prior
        node.children.append(child)

def backpropagate(node, value):
    while node:
        node.visits += 1
        node.value += value
        value = -value
        node = node.parent