# mcts/node.py
import numpy as np

class Node:
    """Enhanced tree node with improved memory efficiency"""
    __slots__ = ('state', 'parent', 'children', 'visits', 'value', 'prior', 'action', 
                'is_expanded', '_in_progress', '_has_legal_actions', '_legal_actions_cache')
    
    def __init__(self, state, parent=None, prior=0.0, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.action = action
        self.is_expanded = False
        self._in_progress = False  # For thread coordination
        
        # Cache legal actions calculation for efficiency
        self._has_legal_actions = None
        self._legal_actions_cache = None
        
    def get_mean_value(self):
        """Get the mean value of this node"""
        return self.value / max(1, self.visits)
    
    def is_leaf(self):
        """Check if this node is a leaf node (unexpanded)"""
        return not self.is_expanded
    
    def has_legal_actions(self):
        """Check if this node has any legal actions (cached)"""
        if self._has_legal_actions is None:
            # Cache result to avoid repeated expensive calls
            legal_actions = self.get_legal_actions()
            self._has_legal_actions = len(legal_actions) > 0
            self._legal_actions_cache = legal_actions
        return self._has_legal_actions
    
    def get_legal_actions(self):
        """Get legal actions with caching"""
        if self._legal_actions_cache is not None:
            return self._legal_actions_cache
        
        # Compute and cache legal actions
        self._legal_actions_cache = self.state.get_legal_actions()
        self._has_legal_actions = len(self._legal_actions_cache) > 0
        return self._legal_actions_cache
    
    def get_ucb_score(self, exploration_weight=1.4):
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
        total_visits = np.sum(visits)
        
        if total_visits == 0:
            # If no visits, use priors
            probs = np.array([child.prior for child in self.children])
            probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs) / len(probs)
        else:
            # Apply temperature
            if temperature == 1.0:
                probs = visits / total_visits
            else:
                visits_temp = visits ** (1.0 / temperature) 
                probs = visits_temp / np.sum(visits_temp)
        
        # Handle numerical issues
        if np.isnan(probs).any() or np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)
        
        # Sample from the policy
        try:
            idx = np.random.choice(len(self.children), p=probs)
            return self.children[idx]
        except ValueError as e:
            # Fallback to most visited if sampling fails
            print(f"Error sampling from probs: {e}, falling back to argmax")
            return max(self.children, key=lambda c: c.visits)
    
    def __repr__(self):
        """String representation for debugging"""
        return f"Node(visits={self.visits}, value={self.get_mean_value():.3f}, expanded={self.is_expanded}, children={len(self.children)})"