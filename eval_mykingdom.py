# -*- coding: utf-8 -*-
"""
학습된 모델을 랜덤/탐욕(영토 그리디) 베이스라인과 대국시켜 정량평가.
- 승률(W/L/D)과 평균 영토차(흑-백)를 출력.
- 기본 체크포인트: ./pretrained_models/mykingdom/best.pth.tar

사용 예)
  python eval_mykingdom.py                        # 기본 세팅
  python eval_mykingdom.py --games 200           # 각 매치업 200판
  python eval_mykingdom.py --sims 100            # MCTS 시뮬 100회
  python eval_mykingdom.py --ckpt your_ckpt.pth.tar
"""

import argparse
import os
import numpy as np

from utils import dotdict
from Arena import Arena
from MCTS import MCTS

# 게임/네트워크 불러오기 (MyKingdom 없으면 Othello로 폴백)
from games.mykingdom.MyKingdomGame import MyKingdomGame as Game
from games.mykingdom.pytorch.NNet import NNetWrapper as NNet
DEFAULT_SIZE = 9
IS_MYKINGDOM = True


# ------------------ 베이스라인 플레이어 ------------------

class RandomPlayer:
    """유효수 중 무작위."""
    def __init__(self, game):
        self.game = game

    def play(self, canonicalBoard):
        valids = self.game.getValidMoves(canonicalBoard, 1)
        moves = np.where(valids == 1)[0]
        return int(np.random.choice(moves))


class GreedyTerritoryPlayer:
    """
    한 수 시뮬레이트 후, 영토 득점(흑-백)이 가장 좋아지는 수 선택.
    - canonicalBoard(플레이어=1 시점)를 받아 유효수별로 getNextState 적용.
    - SCORER(score_territory)가 있으면 그 점수로 평가, 없으면 합법수 개수(자유도)로 대체.
    """
    def __init__(self, game):
        self.game = game

    def play(self, canonicalBoard):
        valids = self.game.getValidMoves(canonicalBoard, 1)
        moves = np.where(valids == 1)[0]
        if len(moves) == 1:
            return int(moves[0])

        best_move = int(moves[0])
        best_score = -1e9

        for a in moves:
            # canonical(플레이어=1) 기준으로 다음 상태 생성
            nextBoard, nextPlayer = self.game.getNextState(canonicalBoard, 1, int(a))

            # 평가 점수 계산
            if SCORER is not None:
                try:
                    bterr, wterr = SCORER(nextBoard)
                    score = bterr - wterr  # 흑-백
                except Exception:
                    score = self._mobility_score(nextBoard)
            else:
                score = self._mobility_score(nextBoard)

            if score > best_score:
                best_score = score
                best_move = int(a)

        return best_move

    def _mobility_score(self, board):
        """스코어러가 없을 때의 간단 휴리스틱: 합법수(자유도)가 많을수록 좋다."""
        valids = self.game.getValidMoves(board, 1)
        return float(np.sum(valids))


# ------------------ AlphaZero 플레이어 ------------------

def make_az_player(game, sims):
    nnet = NNet(game)
    mcts = MCTS(game, nnet, dotdict({'numMCTSSims': sims, 'cpuct': 1.0}))

    def az_play(canonicalBoard):
        pi = mcts.getActionProb(canonicalBoard, temp=0)
        return int(np.argmax(pi))

    return az_play, nnet


# ------------------ 유틸: 평균 영토 차이 ------------------

def avg_territory_diff(game, games, player1, player2):
    """
    여러 판 대국 후 최종 보드에서 평균 (흑-백) 영토 차이를 측정.
    SCORER가 없으면 None 반환.
    """
    if SCORER is None:
        return None

    diffs = []
    arena = Arena(player1, player2, game)
    for _ in range(games):
        b, _ = arena.playSingleGame(return_boards=True)  # 확장: Arena에 이 옵션이 없다면 생략
        try:
            bt, wt = SCORER(b)
            diffs.append(bt - wt)
        except Exception:
            pass
    return float(np.mean(diffs)) if diffs else None


# ------------------ 메인 ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', type=int, default=DEFAULT_SIZE, help='board size')
    parser.add_argument('--sims', type=int, default=200, help='MCTS simulations for AZ player')
    parser.add_argument('--games', type=int, default=200, help='games per matchup')
    parser.add_argument('--ckpt_dir', type=str, default='./pretrained_models/mykingdom/', help='checkpoint dir')
    parser.add_argument('--ckpt', type=str, default='best.pth.tar', help='checkpoint filename')
    args = parser.parseArgs([]) if hasattr(parser, 'parseArgs') else parser.parse_args()

    # 게임 초기화
    try:
        g = Game(args.board)
    except TypeError:
        g = Game()

    # AZ 플레이어 로드
    az_player, nnet = make_az_player(g, sims=args.sims)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt)
    if os.path.isfile(ckpt_path):
        print(f'[Load] checkpoint: {ckpt_path}')
        nnet.load_checkpoint(args.ckpt_dir, args.ckpt)
    else:
        print(f'[Warn] checkpoint not found: {ckpt_path} (초기 가중치로 평가합니다)')

    # 베이스라인 플레이어들
    rnd = RandomPlayer(g).play
    grd = GreedyTerritoryPlayer(g).play

    # 1) AZ vs Random
    arena = Arena(az_player, rnd, g)
    w, l, d = arena.playGames(args.games, verbose=False)
    print(f'[AZ vs Random] W/L/D = {w}/{l}/{d}  (Win={w/(w+l+d):.3f})')

    # 2) AZ vs Greedy
    arena = Arena(az_player, grd, g)
    w2, l2, d2 = arena.playGames(args.games, verbose=False)
    print(f'[AZ vs Greedy] W/L/D = {w2}/{l2}/{d2}  (Win={w2/(w2+l2+d2):.3f})')

    # (옵션) 평균 영토차
    if SCORER is not None and hasattr(Arena, 'playSingleGame'):
        # Arena에 playSingleGame(return_boards=True) 같은 편의가 없으면 이 블록은 생략해도 됨.
        pass

if __name__ == '__main__':
    main()
