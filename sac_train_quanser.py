"""
Quanser Qube-Servo 3 회전형 역진자 - SAC 학습 스크립트 (실제 하드웨어용)

설계 핵심
---------
1) quanser_env.QuanserEnv 는 일반 gym 환경과 인터페이스가 다르다.
   - reset() 은 (obs, info) 가 아니라 obs(torch.Tensor) 하나만 반환한다.
   - step() 은 모터에 PWM 을 쓰지 않고 '상태 읽기 + 보상 계산' 만 한다.
   - 실제 행동(PWM) 적용은 env.apply_action(action) 이 담당한다.
   => 따라서 한 제어 스텝은  apply_action -> Ts 만큼 유지 -> step(상태 읽기)  순서다.

2) 실제 하드웨어는 6ms(약 167Hz) 주기로 동작한다. 매 스텝마다 신경망을
   업데이트하면 제어 루프의 실시간성이 깨진다. 그래서 본 스크립트는
   '에피소드 진행 중에는 경험만 수집'하고, 모터가 초기 위치로 돌아가는
   '리셋 구간(수 초 소요)에서 그래디언트 업데이트를 몰아서' 수행한다.
   (수집 스텝 수 : 업데이트 수 = 1 : 1, UTD≈1)

3) 외부 강화학습 패키지는 사용하지 않는다. PyTorch 로 SAC 를 직접 구현했다.
"""
import os
import csv
import time
import collections
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from quanser.hardware import HIL, HILError

from sac_models import MODEL_DIR, GaussianPolicy, SoftQNetwork, ReplayBuffer, Transition, DEVICE
from quanser_env import QuanserEnv


def to_action_tensor(action) -> torch.Tensor:
    """numpy 행동을 env.apply_action / env.step 이 기대하는 torch.Tensor(shape:(1,))로 변환."""
    return torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(-1))


class SAC:
    def __init__(self, env: QuanserEnv, config: dict, use_wandb: bool = False):
        self.env = env
        self.config = config
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]
        self.automatic_entropy_tuning = config["automatic_entropy_tuning"]
        self.print_episode_interval = config["print_episode_interval"]
        self.save_episode_interval = config["save_episode_interval"]
        self.max_grad_updates_per_episode = config["max_grad_updates_per_episode"]
        self.control_period = config["control_period"]  # 제어 주기(초)

        n_features = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        # ----- 네트워크 -----
        self.policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.q_network = SoftQNetwork(n_features=n_features, n_actions=n_actions)
        self.target_q_network = SoftQNetwork(n_features=n_features, n_actions=n_actions)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        # ----- 엔트로피 자동 조절(alpha) -----
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(DEVICE)).item()
            self.log_alpha = torch.tensor(float(np.log(0.2)), dtype=torch.float32, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.get("fixed_alpha", 0.2)
        self.max_alpha = 5.0
        self.min_alpha = config["min_alpha"]   # 탐색이 0으로 붕괴하지 않도록 하한

        self.time_steps = 0
        self.training_time_steps = 0
        self.best_episode_reward = -np.inf

        # ----- 로그 저장 -----
        self.log_path = os.path.join(MODEL_DIR, "train_log_{0}.csv".format(self.current_time))
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "time_steps", "episode_steps", "episode_reward",
                 "policy_loss", "q1_loss", "q2_loss", "alpha", "entropy"]
            )

        if self.use_wandb:
            import wandb
            self.wandb = wandb.init(project="sac_quanser", name=self.current_time, config=config)
        else:
            self.wandb = None

        # ----- 체크포인트(이어서 학습) -----
        # 하드웨어 fault/로터 분리/중단 후 전원만 재시작하면 '처음부터'가 아니라
        # 쌓인 학습(신경망·버퍼·스텝)을 이어받아 균형까지 누적 전진한다.
        self.ckpt_path = os.path.join(MODEL_DIR, "sac_quanser_checkpoint.pt")
        if config.get("resume", True):
            self.load_checkpoint()

    # ---------------------------------------------------------------- #
    #  학습 루프
    # ---------------------------------------------------------------- #
    def train_loop(self) -> None:
        total_start = time.time()
        policy_loss = q1_loss = q2_loss = entropy = 0.0

        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward, episode_steps = self.run_episode()

            # 에피소드 사이(리셋 구간)에서 그래디언트 업데이트 수행
            if self.time_steps >= self.learning_starts and self.replay_buffer.size() >= self.batch_size:
                grad_updates = min(episode_steps, self.max_grad_updates_per_episode)
                for _ in range(grad_updates):
                    policy_loss, q1_loss, q2_loss, _, _, entropy = self.train_step()

            # 로그 기록
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [n_episode, self.time_steps, episode_steps, round(episode_reward, 3),
                     round(policy_loss, 4), round(q1_loss, 4), round(q2_loss, 4),
                     round(self.alpha, 4), round(entropy, 4)]
                )

            if self.wandb is not None:
                self.wandb.log({
                    "episode_reward": episode_reward, "policy_loss": policy_loss,
                    "q1_loss": q1_loss, "q2_loss": q2_loss, "alpha": self.alpha,
                    "entropy": entropy, "buffer": self.replay_buffer.size(),
                    "episode": n_episode, "time_steps": self.time_steps,
                })

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Epi.{:4,} | Steps {:7,}] R: {:8.2f} | P_L: {:7.3f} | "
                    "Q_L: {:7.3f}/{:7.3f} | alpha: {:5.3f} | ent: {:6.3f} | buf: {:6,}".format(
                        n_episode, self.time_steps, episode_reward, policy_loss,
                        q1_loss, q2_loss, self.alpha, entropy, self.replay_buffer.size()
                    )
                )

            # 모델 저장 (최고 성능 + 주기적)
            if episode_reward > self.best_episode_reward:
                self.best_episode_reward = episode_reward
                self.model_save("best")
            if n_episode % self.save_episode_interval == 0:
                self.model_save("latest")
                self.save_checkpoint()   # 이어가기용 전체 상태 저장

        self.model_save("final")
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - total_start))
        print("Total Training Time:", elapsed)
        if self.wandb is not None:
            self.wandb.finish()

    def run_episode(self) -> tuple[float, int]:
        """한 에피소드를 실시간으로 수집한다 (학습은 하지 않음)."""
        observation = self.env.reset()              # QuanserEnv.reset() 은 obs 만 반환
        observation = np.asarray(observation, dtype=np.float32)

        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            loop_start = time.perf_counter()
            self.time_steps += 1
            episode_steps += 1

            # 행동 선택: 초기에는 무작위 탐색 → 충분한 경험 확보
            if self.time_steps < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.policy.get_action(observation, exploration=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            action_t = to_action_tensor(action)

            # ① 실제 모터에 PWM 적용
            self.env.apply_action(action_t)

            # ② 제어 주기(Ts)만큼 유지 (실시간성 확보)
            elapsed = time.perf_counter() - loop_start
            if elapsed < self.control_period:
                time.sleep(self.control_period - elapsed)

            # ③ 결과 상태 읽기 + 보상 계산
            next_observation, reward, terminated, truncated, _ = self.env.step(action_t)
            next_observation = np.asarray(next_observation, dtype=np.float32)

            self.replay_buffer.append(
                Transition(observation, action, next_observation, float(reward), bool(terminated))
            )

            observation = next_observation
            episode_reward += float(reward)
            done = terminated or truncated

        return episode_reward, episode_steps

    # ---------------------------------------------------------------- #
    #  SAC 업데이트 (1 스텝)
    # ---------------------------------------------------------------- #
    def train_step(self):
        self.training_time_steps += 1
        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        # ----- Q network 업데이트 -----
        with torch.no_grad():
            next_action, next_log_pi, _, _ = self.policy.sample(next_observations)
            q1_next, q2_next = self.target_q_network(next_observations, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            min_q_next[dones] = 0.0
            target_q = rewards + self.gamma * min_q_next

        q1, q2 = self.q_network(observations, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_network_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 3.0)
        self.q_network_optimizer.step()

        # ----- Policy 업데이트 -----
        sample_actions, log_pi, mu, entropy = self.policy.sample(observations, reparameterization_trick=True)
        q1_pi, q2_pi = self.q_network(observations, sample_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 3.0)
        self.policy_optimizer.step()

        # ----- alpha(엔트로피 계수) 업데이트 -----
        if self.automatic_entropy_tuning:
            with torch.no_grad():
                _, log_pi_detached, _, _ = self.policy.sample(observations)
            alpha_loss = (-self.log_alpha.exp() * (log_pi_detached + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha = self.log_alpha.exp().item()
                if self.alpha > self.max_alpha:
                    self.log_alpha.data = torch.log(torch.tensor(self.max_alpha, device=DEVICE))
                    self.alpha = self.max_alpha
                if self.alpha < self.min_alpha:   # 탐색 붕괴 방지(하한 고정)
                    self.log_alpha.data = torch.log(torch.tensor(self.min_alpha, device=DEVICE))
                    self.alpha = self.min_alpha
        else:
            alpha_loss = torch.tensor(0.0)

        # ----- target network soft update -----
        self.soft_update(self.q_network, self.target_q_network, self.soft_update_tau)

        return (policy_loss.item(), q1_loss.item(), q2_loss.item(),
                alpha_loss.item(), mu.mean().item(), entropy.item())

    def soft_update(self, source, target, tau):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(tau * t_param.data + (1.0 - tau) * s_param.data)

    def model_save(self, tag: str) -> None:
        filename = "sac_quanser_{0}.pth".format(tag)
        torch.save(self.policy.state_dict(), os.path.join(MODEL_DIR, filename))

    # ---------------------------------------------------------------- #
    #  체크포인트 저장/복원 (이어서 학습)
    # ---------------------------------------------------------------- #
    def save_checkpoint(self) -> None:
        ckpt = {
            "policy": self.policy.state_dict(),
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_network_optimizer": self.q_network_optimizer.state_dict(),
            "alpha": self.alpha,
            "time_steps": self.time_steps,
            "training_time_steps": self.training_time_steps,
            "best_episode_reward": self.best_episode_reward,
            "replay_buffer": list(self.replay_buffer.buffer),
        }
        if self.automatic_entropy_tuning:
            ckpt["log_alpha"] = self.log_alpha.detach().cpu()
            ckpt["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        # 원자적 저장: 임시파일에 쓴 뒤 교체 → 저장 중 fault가 나도 기존 체크포인트 보존
        tmp_path = self.ckpt_path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, self.ckpt_path)
        print("[checkpoint saved] steps={0:,}, buffer={1:,}, best={2:.1f}".format(
            self.time_steps, self.replay_buffer.size(), self.best_episode_reward))

    def load_checkpoint(self) -> bool:
        if not os.path.exists(self.ckpt_path):
            print("[checkpoint] 없음 → 처음부터 학습 시작")
            return False
        ckpt = torch.load(self.ckpt_path, map_location=DEVICE, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_q_network.load_state_dict(ckpt["target_q_network"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.q_network_optimizer.load_state_dict(ckpt["q_network_optimizer"])
        self.alpha = ckpt["alpha"]
        self.time_steps = ckpt["time_steps"]
        self.training_time_steps = ckpt["training_time_steps"]
        self.best_episode_reward = ckpt["best_episode_reward"]
        self.replay_buffer.buffer = collections.deque(
            ckpt["replay_buffer"], maxlen=self.replay_buffer_size)
        if self.automatic_entropy_tuning and "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(DEVICE))
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        print("[checkpoint loaded] 이어서 학습: steps={0:,}, buffer={1:,}, best={2:.1f}".format(
            self.time_steps, self.replay_buffer.size(), self.best_episode_reward))
        return True


def main() -> None:
    print("TORCH VERSION:", torch.__version__, "| DEVICE:", DEVICE)

    config = {
        "env_name": "QuanserQube",
        "max_num_episodes": 300,            # 하드웨어 학습은 에피소드 비용이 크다
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "soft_update_tau": 0.995,           # target = tau*target + (1-tau)*source
        "replay_buffer_size": 200_000,
        "learning_starts": 2_000,           # 이 스텝 전까지는 무작위 행동으로 탐색
        "automatic_entropy_tuning": True,
        "min_alpha": 0.1,                   # 탐색량(alpha) 하한 → 조기 수렴(가만히 있기) 방지
        "max_grad_updates_per_episode": 1_000,  # 리셋 구간에서 몰아서 학습할 최대 업데이트 수
        "control_period": 0.006,            # 제어 주기(=env.Ts, 약 167Hz)
        "print_episode_interval": 1,
        "save_episode_interval": 5,
        "resume": True,                     # 체크포인트 있으면 이어서 학습(처음부터 하려면 False 또는 checkpoint 파일 삭제)
    }

    card = HIL("qube_servo3_usb", "0")
    sac = None
    try:
        env = QuanserEnv(card)
        sac = SAC(env=env, config=config, use_wandb=False)
        sac.train_loop()
    except KeyboardInterrupt:
        print("\n[중단] 사용자 종료 — 체크포인트 저장 후 종료합니다.")
        if sac is not None:
            try:
                sac.save_checkpoint()
            except Exception as se:
                print("checkpoint save failed:", se)
    except HILError as e:
        print("\n[하드웨어 FAULT] HILError:", e)
        print(">>> Qube 전원을 껐다(약 30초) 켜고 케이블/전원을 확인한 뒤, "
              "다시 'python sac_train_quanser.py'를 실행하면 체크포인트에서 이어서 학습합니다.")
        if sac is not None:
            try:
                sac.save_checkpoint()
            except Exception as se:
                print("checkpoint save failed:", se)
    except RuntimeError as e:
        # 로터 분리 감지 등 환경에서 올라온 중단
        print("\n[중단]", e)
        if sac is not None:
            try:
                sac.save_checkpoint()
            except Exception as se:
                print("checkpoint save failed:", se)
    finally:
        # 안전 종료: 모터 정지 + 앰프 off + 카드 닫기
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
