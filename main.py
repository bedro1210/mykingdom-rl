import os
import re
import logging
import coloredlogs

from Coach import Coach
from utils import *

# ====== [여기서 게임/네트워크 선택] ==========================================
# 기본은 MyKingdom으로 설정. 없다면 자동으로 Othello로 폴백합니다.
try:
    from temp_mykingdom.MyKingdomGame import MyKingdomGame as Game
    from temp_mykingdom.pytorch.NNet import NNetWrapper as nn
    DEFAULT_GAME = "MyKingdom"
    DEFAULT_BOARD_SIZE = 9
except Exception:
    from othello.OthelloGame import OthelloGame as Game
    from othello.pytorch.NNet import NNetWrapper as nn
    DEFAULT_GAME = "Othello"
    DEFAULT_BOARD_SIZE = 6
# ============================================================================

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # DEBUG로 바꾸면 더 자세히 나옵니다.


# ------- 재시작(Resume) 편의 함수들 ------------------------------------------
def _find_latest_checkpoint(ckpt_dir: str):
    """
    checkpoint 디렉터리에서 가장 최근 checkpoint 파일명을 반환.
    우선순위: best.pth.tar > checkpoint_XX.pth.tar (숫자 큰 것)
    """
    if not os.path.isdir(ckpt_dir):
        return None
    best = os.path.join(ckpt_dir, "best.pth.tar")
    if os.path.isfile(best):
        return ("best.pth.tar")
    # checkpoint_XX.pth.tar 중 가장 큰 XX
    patt = re.compile(r"checkpoint_(\d+)\.pth\.tar$")
    cand = []
    for f in os.listdir(ckpt_dir):
        m = patt.match(f)
        if m:
            cand.append((int(m.group(1)), f))
    if cand:
        cand.sort(reverse=True)
        return cand[0][1]
    return None


def _find_latest_examples(ckpt_dir: str):
    """
    AGZ 저장 포맷에서 가장 최신 trainExamples 파일을 추정해서 반환.
    (프로젝트별로 이름이 조금 다를 수 있으니, 없으면 Coach.loadTrainExamples() 기본 로직 사용)
    """
    if not os.path.isdir(ckpt_dir):
        return None
    # 예: trainExamples_iter_XX.pkl / .examples 등
    patt = re.compile(r"trainExamples.*?(\d+).*")
    cand = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("trainExamples"):
            m = patt.match(f)
            # 숫자가 없으면 0으로 간주
            idx = int(m.group(1)) if m else 0
            cand.append((idx, f))
    if cand:
        cand.sort(reverse=True)
        return cand[0][1]
    return None
# ---------------------------------------------------------------------------


# ===== 학습 파라미터 (32시간 예산용 권장 설정) ================================
args = dotdict({
    # ---- 반복/수집/탐색 ----
    'numIters': 24,                 # 전체 반복 횟수 (32h 안쪽 목표)
    'numEps': 80,                   # 각 iter에서 self-play 판수
    'tempThreshold': 15,
    'numMCTSSims': 25,              # 학습용 MCTS 시뮬레이션 수(속도/성능 균형)

    # ---- Arena(평가) ----
    'arenaCompare': 30,             # 새/구 모델 비교 대국 수
    'updateThreshold': 0.55,        # 승격 기준 승률

    # ---- 버퍼/탐색 상수 ----
    'maxlenOfQueue': 200000,        # 학습 데이터 큐 최대 길이
    'cpuct': 1,

    # ---- 체크포인트/로딩 ----
    'checkpoint': './pretrained_models/mykingdom/',   # 저장 폴더
    'load_model': False,            # 강제 로드 여부(아래 autoresume가 True면 자동 결정)
    'load_folder_file': (None, None), # (폴더, 파일명). autoresume가 채워줌
    'numItersForTrainExamplesHistory': 20,

    # ---- 편의 옵션 ----
    'autoresume': True,             # ✅ 켜두면 중간 재시작 자동 처리
    'board_size': DEFAULT_BOARD_SIZE,
})
# ============================================================================


def main():
    # ----- 체크포인트 폴더 준비 -----
    os.makedirs(args.checkpoint, exist_ok=True)

    # ----- 자동 재시작 처리 -----
    if args.autoresume:
        latest_ckpt = _find_latest_checkpoint(args.checkpoint)
        if latest_ckpt:
            args.load_model = True
            args.load_folder_file = (args.checkpoint, latest_ckpt)
            log.info(f"[AutoResume] Found checkpoint: {latest_ckpt}")
        else:
            log.info("[AutoResume] No checkpoint found. Starting fresh.")

    # ----- 게임/보드 초기화 -----
    log.info('Loading %s...', Game.__name__)
    try:
        g = Game(args.board_size)
    except TypeError:
        # 어떤 게임 클래스는 크기 인자를 안 받기도 함
        g = Game()

    # ----- 네트워크 초기화 -----
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    # ----- 체크포인트 로드 -----
    if args.load_model and args.load_folder_file[0] and args.load_folder_file[1]:
        folder, filename = args.load_folder_file
        log.info('Loading checkpoint "%s/%s"...', folder, filename)
        nnet.load_checkpoint(folder, filename)
    else:
        log.warning('Not loading a checkpoint!')

    # ----- Coach 구성 -----
    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    # ----- 학습 예제(트레이스) 로드 -----
    # Coach.loadTrainExamples() 는 기본적으로 checkpoint 폴더의 저장 포맷을 읽음.
    # 일부 포맷의 경우 직접 파일명을 지정해야 할 수 있어, 가능한 경우 최신 파일을 지정해서 보조.
    if args.load_model:
        try:
            latest_examples = _find_latest_examples(args.checkpoint)
            if latest_examples and hasattr(c, 'loadTrainExamplesFromFile'):
                log.info(f'Loading train examples "{latest_examples}"...')
                c.loadTrainExamplesFromFile(os.path.join(args.checkpoint, latest_examples))
            else:
                log.info("Loading 'trainExamples' via default loader...")
                c.loadTrainExamples()
        except Exception as e:
            log.warning(f"Failed loading train examples: {e}. Starting without history.")

    # ----- 학습 시작 -----
    log.info('Starting the learning process 🎉  [Game=%s, Board=%s]', DEFAULT_GAME, args.board_size)
    c.learn()


if __name__ == "__main__":
    main()
