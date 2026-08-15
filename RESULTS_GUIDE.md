# Diffusion Policy 학습 결과

## 2. 한 번의 실험 폴더 구조

```text
09.44.11_test_pusht_lowdim/
├── .hydra/                 # 이 실험에 사용한 설정
│   ├── config.yaml          # 기본값과 변경값을 합친 최종 설정
│   ├── overrides.yaml       # 명령어에서 직접 변경한 값
│   └── hydra.yaml           # Hydra 실행 관련 내부 설정
├── checkpoints/            # 저장된 모델
├── media/                  # PushT 시뮬레이션 평가 영상
├── wandb/                  # Weights & Biases 로그
├── logs.json.txt           # loss, score 등 실제 학습 수치
└── train.log               # 모델 시작·오류 메시지
```

## 3. 가장 먼저 볼 파일

### `logs.json.txt`: 학습이 잘됐는지 확인

각 줄은 한 학습 step 또는 epoch의 결과

| 항목 | 의미 | 보는 법 |
|---|---|---|
| `epoch` | 전체 데이터셋 반복 횟수 | 숫자가 증가하면 학습 진행 중 |
| `global_step` | 지금까지 학습한 미니배치 수 | 세부 학습 진행도 |
| `train_loss` | 학습 데이터에서의 diffusion 오차 | 대체로 낮아지는 것이 좋음 |
| `val_loss` | 학습에 사용하지 않은 데이터의 오차 | 낮아지면 일반화 성능이 개선되는 중 |
| `lr` | 현재 learning rate | scheduler가 조절하는 값 |
| `test/mean_score` | test 시뮬레이션 평균 점수 | `1`에 가까울수록 좋음 |
| `test/sim_max_reward_*` | 특정 test seed의 최고 점수 | 각 평가 환경의 개별 결과 |
| `train/mean_score` | train 초기 조건의 평균 점수 | test score와 함께 참고 |
| `train_action_mse_error` | 생성한 action과 정답 action 사이 MSE | 낮아지는지 확인 |

PushT score는 T 블록이 목표 영역을 얼마나 잘 덮었는지를 나타낸다.

- `0`: 목표에 거의 도달하지 못함
- `1`: 성공 기준에 도달함


`test/mean_score`는 매 epoch 기록되지 않을 수 있다. `rollout_every=50`이면 50 epoch마다 평가하며, epoch 0에서도 한 번 평가한다.

### `.hydra/overrides.yaml`: 실행 조건 확인

실험에서 직접 변경한 설정만 보고 싶을 때 사용한다.

```yaml
- training.seed=42
- training.device=cuda:0
- training.num_epochs=20
- dataloader.batch_size=64
- task.env_runner.n_test=2
- logging.mode=offline
```

`config.yaml`은 모든 기본값까지 포함하므로, 실험을 완전히 재현할 때 본다.

### `checkpoints/`: 학습된 모델 확인

```text
checkpoints/
├── epoch=0000-test_mean_score=0.130.ckpt
└── latest.ckpt
```

- `epoch=...ckpt`: 평가 점수가 좋아서 보존된 모델
- `latest.ckpt`: 가장 최근에 checkpoint를 저장한 시점의 모델
- 학습 재개, 별도 평가, 추론에 사용됨.

### `train.log`: 시작 실패 원인 확인

학습 수치보다는 import 오류, 모델 초기화, 실행 오류를 확인할 때 본다. 학습 폴더에 `logs.json.txt`가 없다면 `train.log`와 `wandb/*/logs/debug.log`를 먼저 확인한다.

### `wandb/`: W&B 로그 확인

`logging.mode=offline`으로 실행하면 W&B 결과가 서버 대신 이 폴더에 저장된다.

- `offline-run-*`: 로컬에 저장된 W&B 실행
- `run-*.wandb`: metric과 그래프 데이터
- `wandb-summary.json`: 최종 요약
- `debug.log`: W&B 시작 실패 등을 확인하는 로그
- `requirements.txt`, `conda-environment.yaml`: 실행 당시 패키지 환경
