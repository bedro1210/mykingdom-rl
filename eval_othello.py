# eval_othello.py
import os
import argparse
import csv
import datetime
import numpy as np
from utils import dotdict
from match_simulator import Arena
from tree_search import MCTS

# ★ 네 프로젝트 구조 기준 import (othello.* 경로)
from othello.othello_env import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as NNet


# ------------------------- Player definitions ------------------------- #
def random_player_fn(game):
    """유효수 중 균등 랜덤(정수 action 반환)."""
    def _p(board):
        valids = game.getValidMoves(board, 1)
        idxs = np.where(valids == 1)[0]
        return int(np.random.choice(idxs))
    return _p


def greedy_player_fn(game):
    """
    1-플라이(한 수 앞) 그리디 휴리스틱.
    현재 플레이어(=1) 기준으로 착수 후 (흑-백) 돌 개수 차이가 최대가 되는 수 선택.
    """
    def _p(board):
        valids = game.getValidMoves(board, 1)
        idxs = np.where(valids == 1)[0]
        best_a, best_score = None, -1e18
        for a in idxs:
            nb, _ = game.getNextState(board, 1, int(a))
            score = int(np.sum(nb == 1) - np.sum(nb == -1))
            if score > best_score:
                best_score, best_a = score, int(a)
        return int(best_a)
    return _p


def mcts_player_fn(game, nnet, sims=200, cpuct=1.0, temp=0.0, safe=True):
    """MCTS 정책 → argmax 정수 action 반환 (Arena 호환)."""
    mcts = MCTS(game, nnet, args=dotdict({'numMCTSSims': sims, 'cpuct': cpuct}))
    def _p(board):
        pi = mcts.getActionProb(board, temp=temp)
        a = int(np.argmax(pi))
        if safe:
            valids = game.getValidMoves(board, 1)
            if valids[a] == 0:
                idxs = np.where(valids == 1)[0]
                return int(np.random.choice(idxs))
        return a
    return _p
# --------------------------------------------------------------------- #


def ensure_ckpt(path_dir: str, path_file: str):
    f = os.path.join(path_dir, path_file)
    if not os.path.isfile(f):
        raise FileNotFoundError(f"Checkpoint not found: {f}")
    return f


def load_nnet(game, ckpt_dir, ckpt_file):
    ensure_ckpt(ckpt_dir, ckpt_file)
    nnet = NNet(game)
    nnet.load_checkpoint(ckpt_dir, ckpt_file)
    return nnet


def run_arena(game, p1, p2, num_games=50, verbose=False):
    arena = Arena(p1, p2, game, display=None)
    oneWon, twoWon, draws = arena.playGames(num_games, verbose=verbose)
    return oneWon, twoWon, draws


def append_csv(row, csv_path):
    header = ["datetime","board","games","sims","cpuct","temp",
              "ckpt1_dir","ckpt1_file","vs","ckpt2_dir","ckpt2_file",
              "p1_wins","p2_wins","draws","p1_winrate","drawrate"]
    need_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)


def plot_from_csv(csv_path, png_path):
    """
    eval_log.csv를 읽어 시간순으로 P1 승률 변동을 그려 png 저장.
    - vs 타입별로 다른 곡선
    - 같은 ckpt/보드만 섞어서 보는 게 좋다(혼합돼 있어도 동작은 함)
    """
    import matplotlib.pyplot as plt

    if not os.path.isfile(csv_path):
        print(f"[plot] CSV not found: {csv_path}")
        return

    # 읽기
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                dt = datetime.datetime.fromisoformat(d["datetime"])
            except Exception:
                # 구버전 포맷 방지
                try:
                    dt = datetime.datetime.strptime(d["datetime"], "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    continue
            rows.append({
                "dt": dt,
                "vs": d.get("vs",""),
                "p1_winrate": float(d.get("p1_winrate", 0.0)),
                "sims": int(d.get("sims", 0)),
                "board": int(d.get("board", 0)),
            })

    if not rows:
        print("[plot] No rows to plot.")
        return

    rows.sort(key=lambda x: x["dt"])

    # vs 유형별로 곡선
    series = {}
    for row in rows:
        key = row["vs"]
        series.setdefault(key, {"x": [], "y": []})
        # x축: 시간 인덱스(0..N-1)로, 읽는 순서 그대로 그림
        series[key]["x"].append(row["dt"])
        series[key]["y"].append(row["p1_winrate"] * 100.0)

    plt.figure(figsize=(9, 4.5))
    for vs, s in series.items():
        plt.plot(s["x"], s["y"], marker="o", label=vs)

    plt.title("Othello Evaluation – P1 WinRate over time")
    plt.xlabel("Run timestamp")
    plt.ylabel("WinRate (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"[plot] Saved: {png_path}")


def main():
    ap = argparse.ArgumentParser("Evaluate Othello checkpoints (with CSV + PNG plotting).")
    ap.add_argument("--board", type=int, default=6, help="Othello board size (default 6)")
    ap.add_argument("--games", type=int, default=50, help="Arena games")
    ap.add_argument("--sims", type=int, default=200, help="MCTS sims per move")
    ap.add_argument("--cpuct", type=float, default=1.0, help="MCTS cpuct")
    ap.add_argument("--temp", type=float, default=0.0, help="MCTS temperature")
    ap.add_argument("--ckpt1_dir", type=str, required=True, help="Checkpoint #1 dir")
    ap.add_argument("--ckpt1_file", type=str, required=True, help="Checkpoint #1 file")
    ap.add_argument("--vs", type=str, default="random",
                    choices=["random", "greedy", "self", "ckpt2"],
                    help="Opponent type")
    ap.add_argument("--ckpt2_dir", type=str, help="(vs=ckpt2) dir")
    ap.add_argument("--ckpt2_file", type=str, help="(vs=ckpt2) file")
    ap.add_argument("--log_csv", type=str, default="eval_log.csv", help="CSV log path")
    ap.add_argument("--plot_png", type=str, default="eval_winrate.png", help="PNG output path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Game
    game = Game(n=args.board)

    # Agent 1
    nnet1 = load_nnet(game, args.ckpt1_dir, args.ckpt1_file)
    p1 = mcts_player_fn(game, nnet1, sims=args.sims, cpuct=args.cpuct, temp=args.temp)

    # Opponent 선택
    if args.vs == "random":
        p2 = random_player_fn(game);      opp_name = "Random"
    elif args.vs == "greedy":
        p2 = greedy_player_fn(game);      opp_name = "Greedy"
    elif args.vs == "self":
        p2 = mcts_player_fn(game, nnet1, sims=args.sims, cpuct=args.cpuct, temp=args.temp)
        opp_name = "Self(Mirror)"
    else:
        if not args.ckpt2_dir or not args.ckpt2_file:
            raise ValueError("--vs ckpt2 사용 시 --ckpt2_dir, --ckpt2_file 필요")
        nnet2 = load_nnet(game, args.ckpt2_dir, args.ckpt2_file)
        p2 = mcts_player_fn(game, nnet2, sims=args.sims, cpuct=args.cpuct, temp=args.temp)
        opp_name = f"CKPT2({os.path.join(args.ckpt2_dir,args.ckpt2_file)})"

    # Arena 실행
    oneWon, twoWon, draws = run_arena(game, p1, p2, num_games=args.games, verbose=args.verbose)
    total = oneWon + twoWon + draws
    wr = (oneWon / total) if total else 0.0
    dr = (draws / total) if total else 0.0

    print("=" * 70)
    print(f"[Eval Othello {args.board}x{args.board}]")
    print(f"Agent1 : {os.path.join(args.ckpt1_dir, args.ckpt1_file)}")
    print(f"Agent2 : {opp_name}")
    print(f"Games  : {total}  (sims={args.sims}, cpuct={args.cpuct}, temp={args.temp})")
    print(f"Result : P1={oneWon}, P2={twoWon}, Draws={draws}")
    print(f"P1 WinRate = {wr*100:.2f}%   DrawRate = {dr*100:.2f}%")
    print("=" * 70)

    # CSV 로깅
    if args.log_csv:
        append_csv([
            datetime.datetime.now().isoformat(timespec="seconds"),
            args.board, args.games, args.sims, args.cpuct, args.temp,
            args.ckpt1_dir, args.ckpt1_file, args.vs,
            args.ckpt2_dir or "", args.ckpt2_file or "",
            oneWon, twoWon, draws, f"{wr:.4f}", f"{dr:.4f}"
        ], args.log_csv)
        print(f"[csv] Appended: {args.log_csv}")

    # PNG 플롯 저장
    if args.plot_png and args.log_csv:
        plot_from_csv(args.log_csv, args.plot_png)


if __name__ == "__main__":
    main()
