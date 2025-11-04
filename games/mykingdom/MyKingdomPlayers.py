# games/mykingdom/MyKingdomPlayers.py
import numpy as np

class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)  # 'player' 값은 여기서 의미 없음
        actions = np.nonzero(valid)[0]
        return np.random.choice(actions)
