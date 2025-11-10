import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from match_simulator import Arena
from tree_search import MCTS

log = logging.getLogger(__name__)


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.ex_hist = []
        self.skip_first = False

    def executeEpisode(self):
        buf = []
        board = self.game.getInitBoard()
        cur = 1
        step = 0

        while True:
            step += 1
            cboard = self.game.getCanonicalForm(board, cur)
            tflag = int(step < self.args.tempThreshold)

            probs = self.mcts.getActionProb(cboard, temp=tflag)
            symset = self.game.getSymmetries(cboard, probs)
            for b2, p2 in symset:
                buf.append([b2, cur, p2, None])

            aidx = np.random.choice(len(probs), p=probs)
            board, cur = self.game.getNextState(board, cur, aidx)

            res = self.game.getGameEnded(board, cur)
            if res != 0:
                return [(e[0], e[2], res * ((-1) ** (e[1] != cur))) for e in buf]

    def learn(self):
        for it_idx in range(1, self.args.numIters + 1):
            log.info(f"Starting Iteration {it_idx}")
            if not self.skip_first or it_idx > 1:
                ex_buf = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    ex_buf += self.executeEpisode()
                self.ex_hist.append(ex_buf)

            if len(self.ex_hist) > self.args.numItersForTrainExamplesHistory:
                log.warning(f"Trim oldest examples. len={len(self.ex_hist)}")
                self.ex_hist.pop(0)

            self.saveTrainExamples(it_idx - 1)

            flat_examples = []
            for e in self.ex_hist:
                flat_examples.extend(e)
            shuffle(flat_examples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")

            pmcts = MCTS(self.game, self.pnet, self.args)
            self.nnet.train(flat_examples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("Head-to-head vs previous snapshot")
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            prev_w, new_w, d = arena.playGames(self.args.arenaCompare)

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d", new_w, prev_w, d)
            if prev_w + new_w == 0 or float(new_w) / (prev_w + new_w) < self.args.updateThreshold:
                log.info("Discard new snapshot")
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            else:
                log.info("Promote new snapshot")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(it_idx))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")

    def getCheckpointFile(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        fname = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(fname, "wb+") as f:
            Pickler(f).dump(self.ex_hist)

    def loadTrainExamples(self):
        model_path = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        ex_path = model_path + ".examples"
        if not os.path.isfile(ex_path):
            log.warning(f'Examples file "{ex_path}" not found')
            ans = input("Continue? [y|n]")
            if ans != "y":
                sys.exit()
        else:
            log.info("Loading stored examples")
            with open(ex_path, "rb") as f:
                self.ex_hist = Unpickler(f).load()
            log.info("Loaded examples")
            self.skip_first = True
