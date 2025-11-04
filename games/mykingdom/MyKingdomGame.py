# games/mykingdom/MyKingdomGame.py
import numpy as np
from .scorer import EMPTY, BLACK, WHITE, winner_by_plus3_rule

class MyKingdomGame:
    """
    AlphaZero-General Game interface implementation for a Go-like 9x9.
    Board values: BLACK=1, WHITE=-1, EMPTY=0
    Actions: 0..N*N-1 place stone at (r,c), N*N is PASS
    """
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
        """
        Returns (next_board, next_player).
        Very simple rules: placing on empty is allowed; PASS does nothing.
        Capture/ko/jigo rules not implemented (you said rules will evolve).
        """
        nb = board.copy()
        if action != self.PASS:
            r, c = divmod(action, self.N)
            if nb[r, c] != EMPTY:
                # AZ-G expects caller not to play invalid moves; but be safe
                raise ValueError(f"Invalid move at {(r,c)}; cell not empty.")
            nb[r, c] = player
        return nb, -player

    def getValidMoves(self, board, player):
        """
        Returns a binary vector of length action_size.
        Empty points are valid, PASS is always valid.
        """
        valid = np.zeros(self.getActionSize(), dtype=int)
        for r in range(self.N):
            for c in range(self.N):
                if board[r, c] == EMPTY:
                    valid[r * self.N + c] = 1
        valid[self.PASS] = 0  # PASS always allowed
        return valid

    # ---- termination & result ----
    def getGameEnded(self, board, player):
        """
        Return 0 if game not ended.
        If ended, return 1 for current 'player' win, -1 for loss (AZ-G convention).
        Here we end the game when the board is full OR (optional) max turns reached.
        You can swap to your preferred end rule later.
        """
        if np.all(board != EMPTY):
            win_color = winner_by_plus3_rule(board)
            return 1 if win_color == player else -1
        return 0

    # ---- canonicalization & symmetries (needed by AZ-G) ----
    def getCanonicalForm(self, board, player):
        """
        Return board from the perspective of 'player' (current player is always BLACK=+1).
        """
        return board * player

    def getSymmetries(self, board, pi):
        """
        Data augmentation for training. For now, return identity only.
        (You can add 8 symmetries: 4 rotations x 2 flips later; just remember to
         transform 'pi' accordingly, with PASS action staying at last index.)
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        # bytes representation for caching in MCTS
        return board.tobytes()
