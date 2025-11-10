# play_othello.py
import os
import argparse
import numpy as np
from utils import dotdict
from tree_search import MCTS

# 프로젝트 구조에 맞춘 import (네가 쓰는 경로)
from othello.othello_env import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as NNet


def load_nnet(game, ckpt_dir, ckpt_file):
    f = os.path.join(ckpt_dir, ckpt_file)
    if not os.path.isfile(f):
        raise FileNotFoundError(f"Checkpoint not found: " + f)
    nnet = NNet(game)
    nnet.load_checkpoint(ckpt_dir, ckpt_file)
    return nnet


def mcts_agent(game, nnet, sims=200, cpuct=1.0, temp=0.0):
    mcts = MCTS(game, nnet, args=dotdict({'numMCTSSims': sims, 'cpuct': cpuct}))
    def _act(board):
        pi = mcts.getActionProb(board, temp=temp)
        a = int(np.argmax(pi))
        # 안전장치: 혹시 비합법 수면 유효수 중 하나로 교체
        valids = game.getValidMoves(board, 1)
        if valids[a] == 0:
            idxs = np.where(valids == 1)[0]
            a = int(np.random.choice(idxs))
        return a
    return _act


def print_board(board):
    n = board.shape[0]
    print("   " + " ".join(str(c) for c in range(n)))
    print("-" * (3 + 2*n))
    for r in range(n):
        row = []
        for c in range(n):
            v = board[r, c]
            row.append("O" if v == 1 else ("X" if v == -1 else "-"))
        print(f"{r:>2} " + " ".join(row))
    print("-" * (3 + 2*n))


def list_valid(game, board, player):
    valids = game.getValidMoves(board, player)
    n = board.shape[0]
    coords = [(i // n, i % n) for i in np.where(valids[:-1] == 1)[0]]
    can_pass = bool(valids[-1] == 1)
    return coords, can_pass


def parse_move(s, n):
    s = s.strip().lower()
    if s in ["p", "pass"]:
        return "pass"
    # "r c" 또는 "r,c" 모두 허용
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 2:
        return None
    try:
        r = int(parts[0]); c = int(parts[1])
    except:
        return None
    if 0 <= r < n and 0 <= c < n:
        return (r, c)
    return None


def main():
    ap = argparse.ArgumentParser("Human vs AI Othello (console)")
    ap.add_argument("--board", type=int, default=6, help="board size (default 6)")
    ap.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint folder")
    ap.add_argument("--ckpt_file", type=str, required=True, help="checkpoint file (e.g., best.pth.tar)")
    ap.add_argument("--human_color", type=str, default="black", choices=["black", "white"],
                    help="black plays first (player=+1), white is -1")
    ap.add_argument("--sims", type=int, default=200, help="MCTS sims per move")
    ap.add_argument("--cpuct", type=float, default=1.0, help="MCTS cpuct")
    ap.add_argument("--temp", type=float, default=0.0, help="MCTS temperature")
    args = ap.parse_args()

    game = Game(args.board)
    nnet = load_nnet(game, args.ckpt_dir, args.ckpt_file)
    ai = mcts_agent(game, nnet, sims=args.sims, cpuct=args.cpuct, temp=args.temp)

    human_as = 1 if args.human_color == "black" else -1  # 사람이 맡는 플레이어 값
    board = game.getInitBoard()
    cur = 1  # 오셀로는 +1(흑)부터 시작

    print("\n=== Human vs AI – Othello {}x{} ===".format(args.board, args.board))
    print(f"Human plays: {args.human_color.upper()}  (black=O, white=X)")
    print("입력 예) r c   (예: 2 3)   |   패스: pass 또는 p\n")

    try:
        while True:
            print_board(board)

            ended = game.getGameEnded(board, cur)
            if ended != 0:
                # 승패 표시
                black_cnt = int(np.sum(board == 1))
                white_cnt = int(np.sum(board == -1))
                print(f"Game Over.  Stones  O(black)={black_cnt} / X(white)={white_cnt}")
                if ended == 1:  # cur 관점이 아님! 오셀로 구현은 player=+1 승/패 기준으로 리턴
                    # ended는 호출 시 player 인자 기준이라 혼동 줄이기 위해 재판정
                    pass
                # 최종 승자 재판정: 흑-백 개수 비교
                if black_cnt > white_cnt:
                    winner = 1
                elif black_cnt < white_cnt:
                    winner = -1
                else:
                    winner = 0
                if winner == 1:
                    print("Winner: BLACK (O)")
                elif winner == -1:
                    print("Winner: WHITE (X)")
                else:
                    print("Result: DRAW")
                # 사람 승패
                if winner == human_as:
                    print("You WIN! 🎉")
                elif winner == 0:
                    print("Draw.")
                else:
                    print("You LOSE.")
                break

            # 현재 차례가 사람?
            if cur == human_as:
                coords, can_pass = list_valid(game, board, cur)
                if coords:
                    print("Valid moves:", ", ".join([f"({r},{c})" for (r, c) in coords]))
                if can_pass:
                    print("※ You may 'pass' (no legal moves).")

                # 입력 루프
                n = args.board
                while True:
                    s = input(f"[{args.human_color.upper()} turn] Enter move (r c) or 'pass': ").strip()
                    mv = parse_move(s, n)
                    if mv is None:
                        print("❗ 형식이 잘못되었습니다. 예: 2 3  또는  pass")
                        continue
                    if mv == "pass":
                        # 합법 패스인지 체크
                        _, allow_pass = list_valid(game, board, cur)
                        if not allow_pass:
                            print("❗ 아직 둘 수 있는 곳이 있어 패스가 불가합니다.")
                            continue
                        action = n * n  # pass index
                    else:
                        r, c = mv
                        idx = r * n + c
                        valids = game.getValidMoves(board, cur)
                        if valids[idx] == 0:
                            print("❗ 합법 수가 아닙니다. 유효 좌표 중 선택하세요.")
                            continue
                        action = idx
                    break
            else:
                # AI 차례
                print("[AI thinking ...]")
                # AI는 canonical 보드(+1 관점)에서 동작하므로 변환
                canon = game.getCanonicalForm(board, cur)
                action = ai(canon)
                # action은 canonical 기준이므로 동일 index 사용 가능

            # 다음 상태
            board, cur = game.getNextState(board, cur, action)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Bye!")


if __name__ == "__main__":
    main()
