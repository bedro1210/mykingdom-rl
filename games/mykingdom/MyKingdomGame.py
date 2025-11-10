# games/mykingdom/MyKingdomGame.py
import numpy as np
from .scorer import EMPTY, BLACK, WHITE, winner_by_plus3_rule

class MyKingdomGame:

    def __init__(self, N=9):
        self.N = N
        self.PASS = self.N * self.N

    # ---- basic sizes ----
    def getInitBoard(self):
        return np.zeros((self.N, self.N), dtype=int)

    def getBoardSize(self):
        return (self.N, self.N)

    def getActionSize(self):
        return self.N * self.N + 1  # +1 for PASS

    # ---- game dynamics ----
    def getNextState(self, board, player, action):

        nb = board.copy()
        if action != self.PASS:
            r, c = divmod(action, self.N)
            if nb[r, c] != EMPTY:
                # AZ-G expects caller not to play invalid moves; but be safe
                raise ValueError(f"Invalid move at {(r,c)}; cell not empty.")
            nb[r, c] = player
        return nb, -player

    def getValidMoves(self, board, player):

        valid = np.zeros(self.getActionSize(), dtype=int)
        for r in range(self.N):
            for c in range(self.N):
                if board[r, c] == EMPTY:
                    valid[r * self.N + c] = 1
        valid[self.PASS] = 0  # PASS always allowed
        return valid

    # ---- termination & result ----
    def getGameEnded(self, board, player):

        if np.all(board != EMPTY):
            win_color = winner_by_plus3_rule(board)
            return 1 if win_color == player else -1
        return 0

    # ---- canonicalization & symmetries (needed by AZ-G) ----
    def getCanonicalForm(self, board, player):

        return board * player

    def getSymmetries(self, board, pi):
 
        return [(board, pi)]

    def stringRepresentation(self, board):
        # bytes representation for caching in MCTS
        return board.tobytes()
