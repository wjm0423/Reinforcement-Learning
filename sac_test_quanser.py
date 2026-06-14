"""
Quanser Qube-Servo 3 회전형 역진자 - 학습된 SAC 정책 실행/시연 스크립트

- 학습으로 저장된 정책(models/sac_quanser_best.pth 등)을 불러와 하드웨어에서 실행한다.
- 발표 영상 촬영용. 탐색 없이 결정론적 행동(mean action)을 사용한다.
- 실행 중 (시간, 펜듈럼 각도, 모터 각도, 행동, 보상)을 csv 로 기록하여
  발표자료의 그래프(예: 펜듈럼 각도 수렴 곡선)로 활용할 수 있다.
"""
import os
import csv
import math
import time
import argparse
from datetime import datetime

import numpy as np
import torch

from quanser.hardware import HIL

from sac_models import MODEL_DIR, GaussianPolicy
from quanser_env import QuanserEnv


def to_action_tensor(action) -> torch.Tensor:
    return torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(-1))


def run(env: QuanserEnv, policy: GaussianPolicy, num_episodes: int, control_period: float):
    log_dir = MODEL_DIR
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, "test_log_{0}.csv".format(stamp))
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "step", "pend_angle_deg", "motor_angle_deg", "action", "reward"])

    for i in range(num_episodes):
        observation = np.asarray(env.reset(), dtype=np.float32)
        episode_reward = 0.0
        step = 0
        done\
            = False

        while not done:
            loop_start = time.perf_counter()
            step += 1

            # 결정론적 행동 (탐색 X)
            action = policy.get_action(observation, exploration=False)
            action_t = to_action_tensor(action)

            env.apply_action(action_t)
            elapsed = time.perf_counter() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

            next_observation, reward, terminated, truncated, _ = env.step(action_t)
            next_observation = np.asarray(next_observation, dtype=np.float32)

            # 기록용: 정규화 해제. obs = [motor/1.8, sin, cos, mvel/20, pvel/40]
            sin_p, cos_p = float(next_observation[1]), float(next_observation[2])
            pend_angle_deg = math.degrees(math.atan2(sin_p, cos_p))
            motor_angle_deg = math.degrees(float(next_observation[0]) * 1.8)

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [i, step, round(pend_angle_deg, 2), round(motor_angle_deg, 2),
                     round(float(action[0]), 4), round(float(reward), 4)]
                )

            observation = next_observation
            episode_reward += float(reward)
            done = terminated or truncated

        print("[EPISODE {0}] STEPS: {1}, REWARD: {2:.2f}".format(i, step, episode_reward))

    print("Test log saved to:", log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best",
                        help="불러올 모델 태그 (best / latest / final) 또는 파일 경로")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--control_period", type=float, default=0.006)
    args = parser.parse_args()

    if os.path.isfile(args.model):
        model_path = args.model
    else:
        model_path = os.path.join(MODEL_DIR, "sac_quanser_{0}.pth".format(args.model))

    card = HIL("qube_servo3_usb", "0")
    try:
        env = QuanserEnv(card)

        n_features = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
        policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
        policy.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        policy.eval()
        print("Loaded model:", model_path)

        run(env, policy, num_episodes=args.episodes, control_period=args.control_period)
    finally:
        try:
            from array import array
            card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
            card.write_digital(array('I', [0]), 1, array('I', [0]))
        except Exception:
            pass
        card.close()
        print("HIL card closed safely.")


if __name__ == "__main__":
    main()
