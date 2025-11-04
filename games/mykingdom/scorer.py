# scorer.py
# Territory scoring for a 9x9 (or NxN) Go-like board.
# Rule:
# - Count only EMPTY regions that are fully surrounded by a single color.
# - If an empty region touches the board edge OR touches both colors -> neutral (not counted).
# - Winner rule: if (black_territory - white_territory) >= 3 => BLACK wins, else WHITE wins.

from collections import deque
import numpy as np

# Constants
EMPTY, BLACK, WHITE = 0, 1, -1

__all__ = [
    "EMPTY", "BLACK", "WHITE",
    "score_territory",
    "winner_by_plus3_rule",
]

def score_territory(board: np.ndarray):
    """
    Compute territory by flood-filling empty regions.

    Parameters
    ----------
    board : np.ndarray of shape (N, N)
        Values must be in {BLACK=1, WHITE=-1, EMPTY=0}.

    Returns
    -------
    (black_terr, white_terr) : tuple[int, int]
        Number of empty points that are territory of black / white.

    Notes
    -----
    - An empty region that touches the board edge is neutral.
    - An empty region bordered by both colors is neutral.
    """
    N = board.shape[0]
    visited = np.zeros_like(board, dtype=bool)

    def nbrs(r, c):
        if r > 0:   yield r-1, c
        if r < N-1: yield r+1, c
        if c > 0:   yield r, c-1
        if c < N-1: yield r, c+1

    black_terr = 0
    white_terr = 0

    for r in range(N):
        for c in range(N):
            if board[r, c] != EMPTY or visited[r, c]:
                continue

            # BFS over one empty region
            q = deque([(r, c)])
            visited[r, c] = True
            region = [(r, c)]
            border_colors = set()
            touches_edge = (r == 0 or r == N-1 or c == 0 or c == N-1)

            while q:
                cr, cc = q.popleft()
                for nr, nc in nbrs(cr, cc):
                    v = board[nr, nc]
                    if v == EMPTY and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        region.append((nr, nc))
                        if nr == 0 or nr == N-1 or nc == 0 or nc == N-1:
                            touches_edge = True
                    elif v == BLACK:
                        border_colors.add(BLACK)
                    elif v == WHITE:
                        border_colors.add(WHITE)

            # Decide ownership
            if not touches_edge and border_colors == {BLACK}:
                black_terr += len(region)
            elif not touches_edge and border_colors == {WHITE}:
                white_terr += len(region)
            # else: neutral -> ignore

    return black_terr, white_terr


def winner_by_plus3_rule(board: np.ndarray):
    """
    Custom winner rule:
    If black_territory - white_territory >= 3 -> BLACK wins, else WHITE wins.
    """
    b, w = score_territory(board)
    return BLACK if (b - w) >= 3 else WHITE


def territory_map(board: np.ndarray) -> np.ndarray:
    """
    반환: same shape, 값은 {BLACK(1)=흑영토, WHITE(-1)=백영토, 0=중립/영토아님/점유칸}
    """
    N = board.shape[0]
    tmap = np.zeros_like(board, dtype=int)
    visited = np.zeros_like(board, dtype=bool)

    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    for r in range(N):
        for c in range(N):
            if board[r,c] != EMPTY or visited[r,c]:
                continue
            # 빈칸 flood-fill
            comp = []
            q = [(r,c)]
            visited[r,c] = True
            adj_colors = set()
            while q:
                x,y = q.pop()
                comp.append((x,y))
                for dx,dy in dirs:
                    nx,ny = x+dx, y+dy
                    if 0<=nx<N and 0<=ny<N:
                        v = board[nx,ny]
                        if v == EMPTY and not visited[nx,ny]:
                            visited[nx,ny] = True
                            q.append((nx,ny))
                        elif v != EMPTY:
                            adj_colors.add(v)
                    else:
                        # 경계는 벽처럼 취급하되 색은 추가하지 않음
                        pass
            owner = 0
            if adj_colors == {BLACK}:
                owner = BLACK
            elif adj_colors == {WHITE}:
                owner = WHITE
            # 그 외(양색 접촉/무색) -> 0
            for (x,y) in comp:
                tmap[x,y] = owner

    # 돌이 놓인 칸은 0
    tmap[board != EMPTY] = 0
    return tmap