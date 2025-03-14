# utils/state_utils.py (Minimal TicTacToe)
import numpy as np
from game_interface import GameState

class TicTacToeState(GameState):
    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1
        self.winner = None

    def is_terminal(self):
        b = self.board
        wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6),
                (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for (i, j, k) in wins:
            if b[i] == b[j] == b[k] != 0:
                self.winner = b[i]
                return True
        if 0 not in b:
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

    def get_current_player(self):
        return self.current_player

    def encode(self):
        return self.board  # Adjust this method for specific NN input

    def get_winner(self):
        return self.winner
