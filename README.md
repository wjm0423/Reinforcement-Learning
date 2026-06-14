# Quanser Qube-Servo 3 회전형 역진자 — SAC 강화학습

실제 하드웨어(Quanser Qube-Servo 3)에서 회전형 역진자를 **swing-up + balance**(아래에 매달린 펜듈럼을 흔들어 올려 꼭대기에서 세워 유지) 시키는 강화학습 프로젝트입니다.
**외부 강화학습 패키지를 사용하지 않고**, PyTorch 만으로 SAC(Soft Actor-Critic)를 직접 구현했습니다.

- **알고리즘**: SAC (off-policy, 연속 행동) — 직접 구현
- **결과**: swing-up 후 꼭대기에서 균형 유지 성공. 최고 에피소드 보상 약 **7,600 / 8,000(이론 최댓값)** 수준(2,000스텝 거의 전 구간 직립 유지).
- **학습된 정책**: `models/sac_quanser_best.pth`

---

## 1. 요구 사항

이 코드는 **실제 Quanser Qube-Servo 3 하드웨어에 직접 접속**합니다. 장비 없이는 학습/시연을 실행할 수 없습니다.

- Python 3.10+
- 파이썬 패키지: `numpy`, `torch`(CPU 가능), `gymnasium`
  ```
  pip install numpy torch gymnasium
  ```
- **Quanser SDK + 파이썬 `quanser` 패키지** (PyPI 가 아니라 Quanser SDK 설치 시 제공)
  - SDK: https://github.com/quanser/quanser_sdk_win64/releases 에서 `install_quanser_sdk.exe` 설치
  - 패키지 설치:
    ```
    python -m pip install --upgrade --find-links "%QSDK_DIR%\python" quanser_api quanser_common quanser_communications quanser_devices quanser_hardware
    ```
- Quanser Qube-Servo 3 장비를 전원 연결 + USB 로 PC 에 연결

---

## 2. 파일 구성

| 파일 | 설명 |
|------|------|
| `quanser_env.py` | 하드웨어 환경. 상태/행동/**보상 설계**, 리셋(모터 초기위치 복귀)·보정·안전 가드 포함 |
| `sac_models.py` | SAC 신경망(GaussianPolicy, Twin Q-Network)과 ReplayBuffer |
| `sac_train_quanser.py` | **학습 스크립트** (체크포인트 이어가기·예외 안전 종료 포함) |
| `sac_test_quanser.py` | **학습된 정책 실행/시연 스크립트** (발표 영상 촬영용) |
| `models/sac_quanser_best.pth` | 학습된 최고 성능 정책 |
| `quanser_control_pwm.py`, `quanser_read_motor.py`, `quanser_read_pendulum.py` | 하드웨어 셋업·동작 확인용 보조 스크립트 |

---

## 3. 실행 방법

> 폴더 안에서(`cd rotary_interted_pendulum`) 실행하세요. 파일들이 서로를 같은 폴더 기준으로 import 합니다.

### 학습
```
python sac_train_quanser.py
```
- 처음 2,000스텝은 무작위 탐색(warmup), 이후 학습 시작.
- 5 에피소드마다 `models/` 에 모델과 체크포인트가 자동 저장됩니다.
- 하드웨어 fault·중단 시 멈추더라도, **전원을 껐다 켜고 같은 명령을 다시 실행하면 체크포인트에서 이어서 학습**합니다.

### 시연(학습된 정책 실행)
```
python sac_test_quanser.py --model best --episodes 3
```
- 탐색 없이 결정론적 정책으로 실행하며, 펜듈럼/모터 각도·행동·보상을 csv 로 기록합니다.

### 실행 전 하드웨어 준비
1. 펜듈럼을 아래(6시 방향)로 자유롭게 매달린 **정지 상태**로 둔다 (리셋 기준점 보정용).
2. 회전 팔(로터)을 본체 허브에 **단단히 재부착**(자석식 퀵커넥트). 팔이 휘두를 공간 확보.
3. 장비 전원 ON + USB 연결 확인.

---

## 4. 보상 설계 핵심 (`quanser_env.py`)

목표(직립 균형)를 학습시키기 위해 보상을 여러 차례 개선했습니다. 최종(v5) 설계:

```
height   = (1 - cos(pend_angle)) / 2          # 아래 0, 위 1
slowness = 1 / (1 + 0.05 * pend_vel^2)         # 느릴수록 1, 빠를수록 0
reward   = height * slowness - 0.01 * action^2
if |pend_angle| > 170° and |pend_vel| < 2:     # 꼭대기서 정지
    reward += 3.0
종료(한계 초과/회전 과다) 시 reward = -50
```

- **height × slowness**: "위에서 **정지**해야만" 큰 보상 → 빙글빙글 돌리기(스핀)는 보상이 거의 0.
- 아래에선 height≈0 이라 속도가 보상에 영향이 적음 → swing-up(아래에서 빠르게 휘두르기)은 자유.
- 매 스텝 비음수 → 에피소드를 빨리 끝내려는 비정상 행동 방지.

자세한 설계 변화(자살 행동 → 스핀 → 직립 균형으로 이어진 보상 개선 과정)는 발표자료를 참고하세요.

---

## 5. GitHub

- 저장소(public): `<여기에 본인 GitHub 저장소 URL 입력>`

## 작성자
- 이름: 우정모
- 학번: 2020136084
