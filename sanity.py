# sanity_mykingdom.py
from games.mykingdom.MyKingdomGame import MyKingdomGame as Game

g = Game(9)
board = g.getInitBoard()
action_size = g.getActionSize()
valid = g.getValidMoves(board, 1)  # 첫 턴: 플레이어 1

print("Action size:", action_size)           # 기대: 82 (81 + pass)
print("Center value (4,4):", board[4][4])    # 기대: 중립 성 값(예: 3)
print("Valid moves sum:", sum(valid))        # 기대: 81칸 중 중앙 제외 80 + pass = 81? (구현에 따라 pass가 항상 1일 수도)
print("Pass index assumed last?:", valid[-1])# 기대: 0 또는 1 (설계에 따라 다름; 보통 마지막이 pass)
