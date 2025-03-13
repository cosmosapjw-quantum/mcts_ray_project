# mcts/node.py
class Node:
    def __init__(self, state, parent=None):
        self.state, self.parent = state, parent
        self.children, self.visits, self.value, self.prior = [], 0, 0, 0