import logging
import math
import numpy as np

EPS = 1e-8
log = logging.getLogger(__name__)


class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Q = {}
        self.N_sa = {}
        self.N_s = {}
        self.P_s = {}
        self.term = {}
        self.Vmask = {}

    def getActionProb(self, canonicalBoard, temp=1):
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.N_sa.get((s, a), 0) for a in range(self.game.getActionSize())]

        if temp == 0:
            mx = np.max(counts)
            cand = np.flatnonzero(np.array(counts) == mx)
            probs = [0] * len(counts)
            probs[int(np.random.choice(cand))] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        tot = float(sum(counts))
        return [x / tot for x in counts]

    def search(self, canonicalBoard):
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.term:
            self.term[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.term[s] != 0:
            return -self.term[s]

        if s not in self.P_s:
            p, v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            p = p * valids
            sm = np.sum(p)
            if sm > 0:
                p /= sm
            else:
                log.error("All valid moves were masked, doing a workaround.")
                p = p + valids
                p /= np.sum(p)
            self.P_s[s] = p
            self.Vmask[s] = valids
            self.N_s[s] = 0
            return -v

        valids = self.Vmask[s]
        best_u = -float("inf")
        best_a = -1

        for a in range(self.game.getActionSize()):
            if not valids[a]:
                continue
            if (s, a) in self.Q:
                u = self.Q[(s, a)] + self.args.cpuct * self.P_s[s][a] * math.sqrt(self.N_s[s]) / (1 + self.N_sa[(s, a)])
            else:
                u = self.args.cpuct * self.P_s[s][a] * math.sqrt(self.N_s[s] + EPS)
            if u > best_u:
                best_u = u
                best_a = a

        a = best_a
        nxt, ply = self.game.getNextState(canonicalBoard, 1, a)
        nxt = self.game.getCanonicalForm(nxt, ply)

        v = self.search(nxt)

        if (s, a) in self.Q:
            self.Q[(s, a)] = (self.N_sa[(s, a)] * self.Q[(s, a)] + v) / (self.N_sa[(s, a)] + 1)
            self.N_sa[(s, a)] += 1
        else:
            self.Q[(s, a)] = v
            self.N_sa[(s, a)] = 1

        self.N_s[s] += 1
        return -v
