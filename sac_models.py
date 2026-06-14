"""
SAC 신경망 모델 및 리플레이 버퍼
- _01_code/_15_SAC/a_sac_models.py 를 기반으로 Quanser 회전형 역진자(실제 하드웨어)에
  맞게 정리한 버전.
- 외부 강화학습 패키지는 사용하지 않으며, PyTorch 만으로 SAC 를 직접 구현한다.
- [수정 포인트] 원본 ReplayBuffer 는 action 을 int64 로 캐스팅하여 연속 행동(PWM)이
  정수로 잘리는 버그가 있었다. 본 프로젝트는 연속 제어이므로 float32 로 저장한다.
"""
import collections
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
EPSILON = 1e-6


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian 정책 (SAC actor)."""

    def __init__(self, n_features, n_actions, hidden_dim=256, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(n_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, n_actions)
        self.log_std_linear = nn.Linear(hidden_dim, n_actions)

        # action rescaling: tanh 출력([-1,1])을 실제 행동 범위로 변환
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

        self.to(DEVICE)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        elif isinstance(state, torch.Tensor):
            state = state.to(dtype=torch.float32, device=DEVICE)

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def get_action(self, state, exploration: bool = True):
        """학습/검증 시 호출. numpy 행동(shape: (n_actions,))을 반환."""
        if exploration:
            action, _, _, _ = self.sample(state)
        else:
            _, _, action, _ = self.sample(state)
        return action.detach().cpu().numpy().reshape(-1)

    def sample(self, state, reparameterization_trick=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        if reparameterization_trick:
            x_t = dist.rsample()  # 재매개변수화: mean + std * N(0,1)
        else:
            x_t = dist.sample()

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = dist.log_prob(x_t)
        # tanh 변환에 대한 보정 (Enforcing Action Bound)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        entropy = dist.entropy().mean()

        return action, log_prob, mean_action, entropy

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class SoftQNetwork(nn.Module):
    """Twin Q-network (clipped double-Q)."""

    def __init__(self, n_features: int = 5, n_actions: int = 1, hidden_dim=256):
        super().__init__()
        self.fc1_1 = nn.Linear(n_features + n_actions, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_3 = nn.Linear(hidden_dim, 1)

        self.fc2_1 = nn.Linear(n_features + n_actions, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_3 = nn.Linear(hidden_dim, 1)

        self.to(DEVICE)

    def forward(self, x, action) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = torch.cat(tensors=[x, action], dim=-1)

        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)

        x2 = F.relu(self.fc2_1(x))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)
        return x1, x2


Transition = collections.namedtuple(
    typename="Transition",
    field_names=["observation", "action", "next_observation", "reward", "done"],
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def clear(self) -> None:
        self.buffer.clear()

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        observations, actions, next_observations, rewards, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )

        observations = np.array(observations, dtype=np.float32)
        next_observations = np.array(next_observations, dtype=np.float32)

        actions = np.array(actions, dtype=np.float32)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards, dtype=np.float32)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)

        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        # [수정] 연속 행동이므로 float32 (원본은 int64 버그)
        actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
