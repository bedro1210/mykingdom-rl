from __future__ import print_function
import sys
sys.path.append('..')

from Game import Game
from .OthelloLogic import Board
import numpy as np


class OthelloGame(Game):
    # 외부에서 참조할 수 있어 그대로 둠
    square_content = {
        -1: "X",
        0:  "-",
        1:  "O",
    }

    @staticmethod
    def getSquarePiece(cell):
        return OthelloGame.square_content[cell]

    def __init__(self, n):
        self.n = n

    # --- Game API ---

    def getInitBoard(self):
        """초기 보드 상태 반환 (numpy array)"""
        bd = Board(self.n)
        return np.array(bd.pieces)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        # 모든 칸(n*n) + pass(1)
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
   
        # pass 액션
        if action == self.n * self.n:
            return (board, -player)

        # 일반 착수
        env = Board(self.n)
        env.pieces = np.copy(board)
        r, c = divmod(int(action), self.n)
        env.execute_move((r, c), player)
        return (env.pieces, -player)

    def getValidMoves(self, board, player):

        mask = [0] * self.getActionSize()

        env = Board(self.n)
        env.pieces = np.copy(board)
        legal = env.get_legal_moves(player)

        if len(legal) == 0:
            mask[-1] = 1
            return np.array(mask)

        for rr, cc in legal:
            mask[self.n * rr + cc] = 1
        return np.array(mask)

    def getGameEnded(self, board, player):

        env = Board(self.n)
        env.pieces = np.copy(board)

        # 양측 중 한쪽이라도 둘 수 있으면 진행
        if env.has_legal_moves(player):
            return 0
        if env.has_legal_moves(-player):
            return 0

        # 돌 개수 차 (player 관점)
        return 1 if env.countDiff(player) > 0 else -1

    def getCanonicalForm(self, board, player):
        # 항상 현재 플레이어 관점(+1)으로 정규화
        return player * board

    def getSymmetries(self, board, pi):

        assert len(pi) == self.n ** 2 + 1
        pi_grid = np.reshape(pi[:-1], (self.n, self.n))
        out = []

        for k in range(1, 5):                  # 90/180/270/360 회전
            for flip_lr in (True, False):      # 좌우 반전 유무
                b2 = np.rot90(board, k)
                p2 = np.rot90(pi_grid, k)
                if flip_lr:
                    b2 = np.fliplr(b2)
                    p2 = np.fliplr(p2)
                out.append((b2, list(p2.ravel()) + [pi[-1]]))
        return out

    def stringRepresentation(self, board):
        # bytes 형태 유지 (기존 코드 호환)
        return board.tobytes()

    def stringRepresentationReadable(self, board):
        # 사람이 읽기 쉬운 문자열
        return "".join(self.square_content[val] for row in board for val in row)

    def getScore(self, board, player):
        env = Board(self.n)
        env.pieces = np.copy(board)
        return env.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   " + " ".join(str(i) for i in range(n)))
        print("-----------------------")
        for r in range(n):
            row = " ".join(OthelloGame.square_content[board[r][c]] for c in range(n))
            print(f"{r} | {row} |")
        print("-----------------------")
