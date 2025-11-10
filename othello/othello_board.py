class Board:
    _dirs = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, n):
        self.n = n
        self.pieces = [[0] * n for _ in range(n)]
        self.pieces[n // 2 - 1][n // 2] = 1
        self.pieces[n // 2][n // 2 - 1] = 1
        self.pieces[n // 2 - 1][n // 2 - 1] = -1
        self.pieces[n // 2][n // 2] = -1

    def __getitem__(self, idx):
        return self.pieces[idx]

    def countDiff(self, color):
        s = 0
        for r in range(self.n):
            for c in range(self.n):
                v = self[c][r]
                if v == color:
                    s += 1
                elif v == -color:
                    s -= 1
        return s

    def get_legal_moves(self, color):
        acc = set()
        for r in range(self.n):
            for c in range(self.n):
                if self[c][r] == color:
                    mv = self.get_moves_for_square((c, r))
                    acc.update(mv)
        return list(acc)

    def has_legal_moves(self, color):
        for r in range(self.n):
            for c in range(self.n):
                if self[c][r] == color:
                    mv = self.get_moves_for_square((c, r))
                    if len(mv) > 0:
                        return True
        return False

    def get_moves_for_square(self, sq):
        x, y = sq
        color = self[x][y]
        if color == 0:
            return None
        out = []
        for d in self._dirs:
            m = self._discover_move(sq, d)
            if m:
                out.append(m)
        return out

    def execute_move(self, move, color):
        flips = [f for d in self._dirs for f in self._get_flips(move, d, color)]
        assert len(flips) > 0
        for x, y in flips:
            self[x][y] = color

    def _discover_move(self, origin, d):
        x, y = origin
        color = self[x][y]
        seen = []
        for x, y in Board._increment_move(origin, d, self.n):
            if self[x][y] == 0:
                return (x, y) if seen else None
            if self[x][y] == color:
                return None
            if self[x][y] == -color:
                seen.append((x, y))

    def _get_flips(self, origin, d, color):
        path = [origin]
        for x, y in Board._increment_move(origin, d, self.n):
            val = self[x][y]
            if val == 0:
                return []
            if val == -color:
                path.append((x, y))
            elif val == color and len(path) > 0:
                return path
        return []

    @staticmethod
    def _increment_move(mv, d, n):
        cur = [mv[0] + d[0], mv[1] + d[1]]
        while 0 <= cur[0] < n and 0 <= cur[1] < n:
            yield cur
            cur = [cur[0] + d[0], cur[1] + d[1]]
