import numpy as np
import subprocess


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        mask = self.game.getValidMoves(board, 1)
        idxs = np.flatnonzero(mask)
        return int(idxs[np.random.randint(len(idxs))])


class HumanOthelloPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        mask = self.game.getValidMoves(board, 1)
        for i, ok in enumerate(mask):
            if ok:
                r, c = divmod(i, self.game.n)
                print("[", r, c, end="] ")
        while True:
            s = input()
            tok = s.split()
            if len(tok) == 2:
                try:
                    r, c = (int(tok[0]), int(tok[1]))
                    if (0 <= r < self.game.n and 0 <= c < self.game.n) or (r == self.game.n and c == 0):
                        a = self.game.n * r + c if r != -1 else self.game.n ** 2
                        if mask[a]:
                            return a
                except ValueError:
                    pass
            print("Invalid move")


class GreedyOthelloPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        mask = self.game.getValidMoves(board, 1)
        cand = []
        for a in range(self.game.getActionSize()):
            if not mask[a]:
                continue
            nxt, _ = self.game.getNextState(board, 1, a)
            sc = self.game.getScore(nxt, 1)
            cand.append((-sc, a))
        cand.sort()
        return cand[0][1]


class GTPOthelloPlayer:
    colors = {-1: "white", 1: "black"}

    def __init__(self, game, gtpClient):
        self.game = game
        self.gtpClient = gtpClient
        self._proc = None
        self._turn = 1

    def startGame(self):
        self._turn = 1
        self._proc = subprocess.Popen(
            self.gtpClient, bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self._send("boardsize " + str(self.game.n))
        self._send("clear_board")

    def endGame(self):
        if self._proc is not None:
            self._send("quit")
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def notify(self, board, action):
        col = GTPOthelloPlayer.colors[self._turn]
        mv = self._action_to_move(action)
        self._send(f"play {col} {mv}")
        self._flip_turn()

    def play(self, board):
        col = GTPOthelloPlayer.colors[self._turn]
        mv = self._send(f"genmove {col}")
        act = self._move_to_action(mv)
        self._flip_turn()
        return act

    def _flip_turn(self):
        self._turn = -self._turn

    def _action_to_move(self, a):
        if a < self.game.n ** 2:
            r, c = divmod(int(a), self.game.n)
            return f"{chr(ord('A') + c)}{r + 1}"
        return "PASS"

    def _move_to_action(self, mv):
        if mv != "PASS":
            c = ord(mv[0]) - ord("A")
            r = int(mv[1:])
            return (r - 1) * self.game.n + c
        return self.game.n ** 2

    def _send(self, cmd):
        self._proc.stdin.write(cmd.encode() + b"\n")
        resp = ""
        while True:
            line = self._proc.stdout.readline().decode()
            if line == "\n":
                if resp:
                    break
                else:
                    continue
            resp += line
        if resp.startswith("="):
            return resp[1:].strip().upper()
        raise Exception("Error calling GTP client: {}".format(resp[1:].strip()))

    def __call__(self, game):
        return self.play(game)
