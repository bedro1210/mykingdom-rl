import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena:
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        players = [self.player2, None, self.player1]
        turn = 1
        board = self.game.getInitBoard()
        step = 0

        for agent in (players[0], players[2]):
            if hasattr(agent, "startGame"):
                agent.startGame()

        while self.game.getGameEnded(board, turn) == 0:
            step += 1

            if verbose and self.display is not None:
                print(f"Turn {step} | Player {turn}")
                self.display(board)

            view = self.game.getCanonicalForm(board, turn)
            action = players[turn + 1](view)
            valid = self.game.getValidMoves(view, 1)

            if valid[action] == 0:
                log.error(f"Invalid action attempted: {action}")
                log.debug(f"Valid mask: {valid}")
                raise ValueError("Invalid move encountered during play.")

            opp = players[-turn + 1]
            if hasattr(opp, "notify"):
                opp.notify(board, action)

            board, turn = self.game.getNextState(board, turn, action)

        for agent in (players[0], players[2]):
            if hasattr(agent, "endGame"):
                agent.endGame()

        if verbose and self.display is not None:
            print(f"Game Over at Turn {step} | Result = {self.game.getGameEnded(board, 1)}")
            self.display(board)

        return turn * self.game.getGameEnded(board, turn)

    def playGames(self, num, verbose=False):
        half = int(num / 2)
        w1 = w2 = dr = 0

        for _ in tqdm(range(half), desc="Arena.playGames (P1 first)"):
            r = self.playGame(verbose)
            if r == 1:
                w1 += 1
            elif r == -1:
                w2 += 1
            else:
                dr += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(half), desc="Arena.playGames (P2 first)"):
            r = self.playGame(verbose)
            if r == -1:
                w1 += 1
            elif r == 1:
                w2 += 1
            else:
                dr += 1

        return w1, w2, dr
