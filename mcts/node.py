# mcts/node.py
import numpy as np

class Node:
    """Enhanced tree node with improved memory efficiency"""
    __slots__ = ('state', 'parent', 'children', 'visits', 'value', 'prior', 'action', 'is_expanded')
    
    def __init__(self, state, parent=None, prior=0.0, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.action = action  # Action that led to this state from parent
        self.is_expanded = False
        
    def get_mean_value(self):
        """Get the mean value of this node"""
        return self.value / max(1, self.visits)
    
    def is_leaf(self):
        """Check if this node is a leaf node (unexpanded)"""
        return len(self.children) == 0
    
    def get_ucb_score(self, exploration_weight=1.0):
        """Calculate the UCB score for this node"""
        if self.visits == 0:
            return float('inf')
        
        parent_visits = self.parent.visits if self.parent else 1
        
        # Q-value (exploitation)
        q_value = self.value / self.visits
        
        # U-value (exploration)
        u_value = exploration_weight * self.prior * np.sqrt(parent_visits) / (1 + self.visits)
        
        return q_value + u_value
        
    def best_child(self, temperature=1.0):
        """Select the best child based on visit counts and optional temperature"""
        if not self.children:
            return None
            
        if temperature == 0:
            # Select most visited child deterministically
            return max(self.children, key=lambda c: c.visits)
        
        # Convert visit counts to policy using temperature
        visits = np.array([child.visits for child in self.children])
        if np.sum(visits) == 0:
            # If no visits, use priors
            probs = np.array([child.prior for child in self.children])
        else:
            # Apply temperature
            if temperature == 1.0:
                probs = visits / np.sum(visits)
            else:
                visits_temp = visits ** (1.0 / temperature) 
                probs = visits_temp / np.sum(visits_temp)
        
        # Sample from the policy
        idx = np.random.choice(len(self.children), p=probs)
        return self.children[idx]