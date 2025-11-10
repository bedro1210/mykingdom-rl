import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena:


    def __init__(self, p1, p2, env, visualize=None):

        self.p1 = p1
        self.p2 = p2
        self.env = env
        self.visualize = visualize



    def runSingleGame(self, verbose=False):

 
        agents = [self.p2, None, self.p1]
        turn = 1
        board = self.env.getInitBoard()
        step = 0

   
        for agent in (agents[0], agents[2]):
            if hasattr(agent, "startGame"):
                agent.startGame()

     
        while self.env.getGameEnded(board, turn) == 0:
            step += 1

            if verbose:
                assert self.visualize
                print(f"Turn {step} | Player {turn}")
                self.visualize(board)

            current_view = self.env.getCanonicalForm(board, turn)
            chosen_action = agents[turn + 1](current_view)
            valid_moves = self.env.getValidMoves(current_view, 1)

            if valid_moves[chosen_action] == 0:
                log.error(f"Invalid action attempted: {chosen_action}")
                log.debug(f"Valid mask: {valid_moves}")
                raise ValueError("Invalid move encountered during play.")

       
            opponent = agents[-turn + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, chosen_action)

      
            board, turn = self.env.getNextState(board, turn, chosen_action)

 
        for agent in (agents[0], agents[2]):
            if hasattr(agent, "endGame"):
                agent.endGame()


        if verbose:
            assert self.visualize
            print(f"Game Over at Turn {step} | Result = {self.env.getGameEnded(board, 1)}")
            self.visualize(board)

        return turn * self.env.getGameEnded(board, turn)

    # ------------------------------------------------------

    def runMultipleGames(self, total_games, verbose=False):

        half_games = int(total_games / 2)
        wins_p1, wins_p2, draws = 0, 0, 0

 
        for _ in tqdm(range(half_games), desc="Arena.runGames (P1 First)"):
            result = self.runSingleGame(verbose)
            if result == 1:
                wins_p1 += 1
            elif result == -1:
                wins_p2 += 1
            else:
                draws += 1

   
        self.p1, self.p2 = self.p2, self.p1

        for _ in tqdm(range(half_games), desc="Arena.runGames (P2 First)"):
            result = self.runSingleGame(verbose)
            if result == -1:
                wins_p1 += 1
            elif result == 1:
                wins_p2 += 1
            else:
                draws += 1

        return wins_p1, wins_p2, draws
