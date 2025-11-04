\# Othello Reinforcement Learning Project



팀명: Othello-RL  

팀원: 이준희(20201627)  





---



**프로젝트 개요**

본 프로젝트는 강화학습(Reinforcement Learning) 기법을 이용하여  

보드게임 “오셀로” 환경에서 에이전트를 학습시키는 것을 목표로 합니다.  



이 환경은 기존 오셀로 게임의 규칙을 기반으로 

더 높은 전략적 판단과 탐색 능력을 요구하도록 설계되었습니다.  



---



**프로젝트 목표**

\- 강화학습을 통해 게임 환경에서의 최적 행동을 학습하는 에이전트 구현  

\- MCTS(Monte Carlo Tree Search) 기반 정책 탐색 적용  

\- 학습되지 않은 랜덤 에이전트 및 사람과의 대국을 통해 성능 정량·정성 평가  



---



**실행 방법**



(1) 환경 설정

\# 패키지 설치

pip install -r requirements.txt



(2) 학습 실행

python main.py



* 학습된 모델은 pretrained\_models/mykingdom/best.pth 로 저장됩니다.
* 로그 및 self-play 데이터는 temp\_mykingdom/ 폴더에 기록됩니다.



(3) 평가 실행

\# AI 간 대전 (학습 모델 vs 랜덤)

python eval\_othello.py



\# 사람과 대전

python play\_othello.py


평가 옵션을 활용한 다양한 평가 실행 설명은 아래에 있습니다.


**모델 다운로드**



학습된 최종 모델(best.pth)은 아래 경로에서 확인 가능합니다:

pretrained\_models/mykingdom/best.pth



**실험 결과**



학습 에피소드 진행 시 승률 상승 그래프를 통해 성능 향상 확인

학습된 에이전트는 랜덤 플레이어 대비 약 85% 이상의 승률을 달성

정량적 평가 외에도 사람과의 대국 테스트를 통해 전략적 행동 검증 완료



**사용 기술**



Algorithm: MCTS + Policy/Value Neural Network

Framework: PyTorch

Language: Python 3.9



실행 옵션 

필수

--ckpt1_dir : 모델 폴더

--ckpt1_file: 체크포인트 파일명

자주 쓰는 선택

--board (기본 6) : 보드 크기

--games (기본 50) : 대전 판수

--sims (기본 200) : MCTS 시뮬레이션 수(클수록 강하지만 느림)

--cpuct (기본 1.0) : MCTS 탐색/활용 균형

--temp (기본 0.0) : 행동 선택 온도(0이면 argmax)

--vs : 상대 종류 (random/greedy/self/ckpt2)

--log_csv : 결과를 CSV로 축적 (기본 eval_log.csv)

--plot_png : CSV 바탕으로 승률변동 그래프 저장 (기본 eval_winrate.png)

--verbose : 수순 로그 출력

--vs ckpt2 를 쓰면 상대 모델도 --ckpt2_dir, --ckpt2_file 로 지정해야 해요.


실행 예시 

1. 모델 vs 랜덤 (기본 평가)
python eval_othello.py --board 6 --games 100 --sims 200 ^
  --ckpt1_dir "C:\mykingdom\alpha-zero-general\pretrained_models\mykingdom" ^
  --ckpt1_file "best.pth.tar"

2️. 모델 vs Greedy (1수 앞을 보는 탐욕적 상대)
python eval_othello.py --board 6 --games 100 --sims 200 ^
  --ckpt1_dir "C:\mykingdom\alpha-zero-general\pretrained_models\mykingdom" ^
  --ckpt1_file "best.pth.tar" ^
  --vs greedy

3️. 모델 vs 자기 자신 (Self-play 평가)
python eval_othello.py --board 6 --games 100 --sims 200 ^
  --ckpt1_dir "C:\mykingdom\alpha-zero-general\pretrained_models\mykingdom" ^
  --ckpt1_file "best.pth.tar" ^
  --vs self

4️. 체크포인트 간 비교 (이전 vs 최신 모델)
python eval_othello.py --board 6 --games 100 --sims 200 ^
  --ckpt1_dir "C:\mykingdom\alpha-zero-general\pretrained_models\mykingdom" ^
  --ckpt1_file "checkpoint_10.pth.tar" ^
  --vs ckpt2 ^
  --ckpt2_dir "C:\mykingdom\alpha-zero-general\pretrained_models\mykingdom" ^
  --ckpt2_file "best.pth.tar"


