# main_mykingdom.py
from Coach import Coach
from utils import *
from games.mykingdom.MyKingdomGame import MyKingdomGame as Game
from games.mykingdom.pytorch.NNet import NNetWrapper as nn  

"""
MyKingdom 강화학습 설정
"""

args = dotdict({
    'numIters': 5,                       # 전체 학습 반복 횟수 (테스트용은 5로 작게)
    'numEps': 10,                        # self-play 에피소드 수 (작게 시작)
    'tempThreshold': 15,
    'updateThreshold': 0.55,             # 새 네트워크 채택 기준
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,                   # MCTS 탐색 시뮬레이션 수 (작게)
    'arenaCompare': 20,                  # evaluation 시 비교 게임 수
    'cpuct': 1.0,

    'checkpoint': './temp_mykingdom/',
    'load_model': False,
    'numItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__":
    game = Game(N=9)
    nnet = nn(game)   # 신경망 초기화 (Othello용 구조 그대로 사용)

    coach = Coach(game, nnet, args)
    print("=== Starting MyKingdom self-play training ===")
    coach.learn()
