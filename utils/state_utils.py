# utils/state_utils.py (Minimal TicTacToe)
import numpy as np

class TicTacToeState:
    def __init__(self):
        self.board = np.zeros(9)
        self.current_player = 1
        self.winner = None

    def is_terminal(self):
        b = self.board.reshape(3, 3)
        for i in range(3):
            if abs(sum(b[i,:])) == 3 or abs(sum(b[:,i])) == 3:
                self.winner = self.current_player
                return True
        if abs(b.trace()) == 3 or abs(np.fliplr(b).trace()) == 3:
            self.winner = self.current_player
            return True
        if not 0 in self.board:
            self.winner = 0  # Draw
            return True
        return False

    def get_legal_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def apply_action(self, action):
        new_state = TicTacToeState()
        new_state.board = self.board.copy()
        new_state.board[action] = self.current_player
        new_state.current_player = -self.current_player
        return new_state