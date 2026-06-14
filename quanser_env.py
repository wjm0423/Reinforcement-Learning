from quanser.hardware import MAX_STRING_LENGTH, HILError
from array import array
import copy
import math
import time

import gymnasium as gym
import numpy as np
import torch
from collections import deque

np.set_printoptions(precision=5, suppress=True)

ENCODER_CPR = 2048.0
RAD_PER_COUNT = 2.0 * math.pi / ENCODER_CPR


class QuanserEnv(gym.Env):
    def __init__(self, card):
        self.card = card

        # PWM mode setup
        self.card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)
        input_channels = array('I', [1])
        output_channels = array('I', [0])
        self.card.set_digital_directions(
            input_channels, len(input_channels),
            output_channels, len(output_channels),
        )
        self.card.write_digital(array('I', [0]), 1, array('I', [1]))

        # channels
        self.pwm_ch = array('I', [0])
        self.motor_enc_ch = array('I', [0])
        self.pend_enc_ch = array('I', [1])
        self.tach_ch = array('I', [14001])
        self.tach_motor_ch = array('I', [14000])

        # LED channels
        self.led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

        # encoder value buffers
        self.motor_enc_val = array('i', [0])
        self.pend_enc_val = array('i', [0])
        self.tach_vel_val = array('d', [0.0])
        self.tach_motor_vel_val = array('d', [0.0])

        # control params
        self.Ts = 0.006
        self.max_steps = 2000
        self.action_scale = 0.27   # 분리/앰프 fault 위험 완화. swing-up은 유지되는 선에서 mild하게(0.35→0.30→0.27)

        # counters
        self.time_steps = 0
        self.step_count = 0
        self.reset_count = 0

        # PD gains
        self.Kp = 0.8
        self.Kd = 0.02

        self.last_action = 0.0
        self.std_error = deque(maxlen=2000)
        self.pen_init_count_list = []

        low_obs = np.array([
            -2.5, -1.0, -1.0, -np.inf, -np.inf,
        ], dtype=np.float32)
        high_obs = np.array([
            2.5, 1.0, 1.0, np.inf, np.inf,
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.init_count = 0
        self.pend_init_count = 0
        self._init_count_valid = False   # 첫 보정 전에는 점프 가드를 적용하지 않음
        self._calib_reject_count = 0     # 연속 보정 거부 횟수(데드락 방지용)
        self._consec_abnormal = 0        # 연속 span 이상 횟수(로터 분리 감지용)
        self._reset_init_count()

        self.step_time = None

    def _get_motor_angle(self):
        self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
        count = self.motor_enc_val[0] - self.init_count
        return count * RAD_PER_COUNT

    def _get_pendulum_angle(self):
        """[-π, π] 범위의 진자 각도를 반환한다. 6시 방향이 0."""
        self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
        raw_angle = (self.pend_enc_val[0] - self.pend_init_count) * RAD_PER_COUNT
        return ((raw_angle + math.pi) % (2 * math.pi)) - math.pi

    def _get_motor_velocity(self):
        self.card.read_other(self.tach_motor_ch, 1, self.tach_motor_vel_val)
        return self.tach_motor_vel_val[0] * RAD_PER_COUNT

    def _get_pendulum_velocity(self):
        self.card.read_other(self.tach_ch, 1, self.tach_vel_val)
        return self.tach_vel_val[0] * RAD_PER_COUNT

    def _reset_init_count(self):
        print("calibrate motor init count...")

        # 하드 스톱이 있으므로, 끝단에 '확실히' 밀착시킨 뒤 안정화하여 읽는다.
        # duty를 0.10으로 약간 높여 스톱에 밀착, 밀착 후 여러 번 읽어 중앙값(글리치 완화).
        for duty_val in (-0.10, 0.10):
            for _ in range(301):
                self.card.write_pwm(self.pwm_ch, 1, array('d', [duty_val]))
                time.sleep(0.01)
            # 스톱에 밀착 유지하며 안정화 + 다중 샘플
            samples = []
            for _ in range(15):
                self.card.write_pwm(self.pwm_ch, 1, array('d', [duty_val]))
                self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
                samples.append(self.motor_enc_val[0])
                time.sleep(0.005)
            reading = int(np.median(samples))
            if duty_val < 0:
                push_max_count = reading
            else:
                push_min_count = reading

        candidate = (push_max_count + push_min_count) // 2
        span = abs(push_max_count - push_min_count)

        # 가드1: 끝점 간격이 물리 가동범위(총≈1600count)보다 비정상적으로 크면
        # (단일 reading 글리치) 보정 거부.
        if span > 2000:
            self._calib_reject_count += 1
            self._consec_abnormal += 1
            print(f"[WARN] calibration span abnormal ({span} counts). "
                  f"keep init_count={self.init_count}")
            # span 이상이 연속으로 누적되면 = 로터가 본체에서 분리됐을 가능성 높음.
            # 무인 실행이 몇 시간씩 헛도는 것을 막기 위해 명확한 에러로 중단한다.
            if self._consec_abnormal >= 10:
                raise RuntimeError(
                    "HARDWARE ERROR: 모터 엔코더 span 이상이 10회 연속 발생했습니다. "
                    "로터(회전 팔)가 본체 허브에서 분리됐을 가능성이 큽니다. "
                    "전원을 끄고 로터를 올바른 방향으로 단단히 재부착한 뒤 다시 실행하세요."
                )
            return
        # 가드2: 직전값 대비 큰 점프는 글리치 의심 → 거부.
        # 단, 연속 2회 이상 거부되면 '실제 기준 이동'으로 보고 수용(무한 재보정 데드락 방지).
        if (self._init_count_valid and abs(candidate - self.init_count) > 1500
                and self._calib_reject_count < 2):
            self._calib_reject_count += 1
            print(f"[WARN] init_count jump suspicious ({candidate - self.init_count}). "
                  f"keep init_count={self.init_count}")
            return

        self.init_count = candidate
        self._init_count_valid = True
        self._calib_reject_count = 0
        self._consec_abnormal = 0
        print("set init count:", self.init_count, "| span:", span)

    def _reset_pendulum_init_count(self) -> None:
        self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
        self.pen_init_count_list.append(copy.deepcopy(self.pend_enc_val[0]))
        if len(self.pen_init_count_list) > 100:
            pend_init_count_mean = int(np.mean(self.pen_init_count_list))
            print("pendulum init count diff:",
                  (self.pend_init_count - pend_init_count_mean) % int(ENCODER_CPR))
            self.pend_init_count = pend_init_count_mean

    def _get_pend_spin_num(self):
        return (self.pend_init_count - self.pend_enc_val[0]) / ENCODER_CPR

    def get_init_observations(self):
        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        motor_vel = self._get_motor_velocity()
        pend_vel = self._get_pendulum_velocity()

        return torch.tensor([
            motor_angle, math.sin(pend_angle), math.cos(pend_angle),
            motor_vel, pend_vel,
        ], dtype=torch.float32)

    def normalize_observation(self, observation):
        return torch.tensor([
            observation[0] / 1.8,
            observation[1],
            observation[2],
            observation[3] / 20.0,
            observation[4] / 40.0,
        ], dtype=torch.float32)

    def _set_led(self, r: float, g: float, b: float):
        values = np.array([r, g, b], dtype=np.float64)
        self.card.write_other(self.led_channels, len(self.led_channels), values)

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        self.step_time = None
        self.last_action = 0.0
        self.std_error = deque(maxlen=2000)

        print("\n======RESET START======")
        self._set_led(0.0, 0.0, 1.0)

        start = time.time()
        reset_counter = 0
        reset_success_num = 0

        if self.reset_count % 5 == 0:
            self._reset_init_count()

        thresh_pend_vel = 0.3     # 리셋 가속(펜듈럼 미세 흔들림 허용) → 재시도 폭주 감소
        thresh_error_rad = 0.1

        while True:
            reset_counter += 1
            cur_rad = self._get_motor_angle()
            error_rad = -cur_rad
            pend_vel = self._get_pendulum_velocity()

            if abs(error_rad) < thresh_error_rad and abs(pend_vel) < thresh_pend_vel:
                reset_success_num += 1
            else:
                reset_success_num = 0
                self.pen_init_count_list = []

            if reset_success_num > 50:
                self.card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
                break

            omega = self._get_motor_velocity()
            duty = np.clip(self.Kp * 2 * error_rad - self.Kd * omega, -0.04, 0.04)
            self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
            time.sleep(0.005)

            if reset_counter > 1000 and reset_success_num == 0:
                thresh_pend_vel += 0.05
                thresh_error_rad += 0.05
                reset_counter = 0
                if abs(error_rad) > 0.3:
                    self._reset_init_count()
                    print("RESET FAILED, RE-CALIBRATE MOTOR INIT COUNT")
                else:
                    print(f"FAILED TO RESET, RETRYING... "
                          f"error_rad: {abs(error_rad):.3f}, "
                          f"pend_vel: {abs(pend_vel):.3f}")

        self.pen_init_count_list = []
        for _ in range(101):
            self._reset_pendulum_init_count()
            time.sleep(0.003)

        print("\n======RESET END======")
        print(f"Reset time: {time.time() - start:.2f} sec")

        obs = self.normalize_observation(self.get_init_observations())
        self.step_time = time.perf_counter()
        self._set_led(1.0, 0.0, 0.0)
        return obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        self.time_steps += 1
        self.step_count += 1
        self.last_action = actions.item()

        # read observations
        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        motor_vel = self._get_motor_velocity()
        pend_vel = self._get_pendulum_velocity()

        next_obs = torch.tensor([
            motor_angle, math.sin(pend_angle), math.cos(pend_angle),
            motor_vel, pend_vel,
        ], dtype=torch.float32)
        next_obs = self.normalize_observation(next_obs)

        # ----- reward (직접 설계 v5: 스핀 차단 + balance 강제) -----
        # 핵심: height 보상에 'slowness(느림)'을 곱한다.
        #   - 위에서 '정지'해야만 큰 보상 → 돌리기(스핀)는 보상이 거의 0.
        #   - 아래(height≈0)선 속도가 보상에 거의 영향 없음 → swing-up은 그대로 가능.
        #   - 매 스텝 비음수 → '빨리 끝내기(자살)' 유인 없음.
        height = (1.0 - math.cos(pend_angle)) / 2.0       # 아래 0, 위 1
        slowness = 1.0 / (1.0 + 0.05 * (pend_vel ** 2))   # 느릴수록 1, 빠를수록 0
        reward = height * slowness
        reward -= 0.01 * (self.last_action ** 2)          # 부드러운 제어(미세)
        if abs(pend_angle) > 2.96706 and abs(pend_vel) < 2.0:  # 꼭대기서 정지 = 큰 보너스(강한 목표점)
            reward += 3.0

        # termination
        terminated = False
        pend_spin_num = self._get_pend_spin_num()
        if abs(pend_spin_num) > 3.0:   # 지속 회전 조기 차단(swing-up 오버슈트는 허용)
            print("PENDULUM SPIN OVER:", pend_spin_num)
            terminated = True
        if abs(math.degrees(motor_angle)) > 100.0:   # 팔이 멀리 나가 떨어지는 것 방지(여유는 유지)
            print("MOTOR ANGLE OVER:", math.degrees(motor_angle))
            terminated = True
        if abs(pend_vel) > 40.0:
            print("PENDULUM VELOCITY OVER:", pend_vel)
            terminated = True

        if terminated:
            reward = -50.0      # 한계 초과/스핀 = 큰 패널티 → 조기 종료가 '가만히 있기'보다 훨씬 나쁘게

        truncated = self.step_count >= self.max_steps
        if truncated:
            print("TRUNCATED")

        if terminated or truncated:
            self._set_led(0.0, 0.0, 1.0)
            self.reset_helper()

        return next_obs, reward, terminated, truncated, {}
    
    def reset_helper(self):
        reset_counter = 0
        reset_success_num = 0

        while reset_counter < 3000:
            reset_counter += 1
            self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
            cur_rad = (self.motor_enc_val[0] - self.init_count) * RAD_PER_COUNT
            error_rad = -cur_rad

            if abs(error_rad) < 0.2:
                reset_success_num += 1
            else:
                reset_success_num = 0

            if reset_success_num > 150:
                self.card.write_pwm(self.pwm_ch, 1, array('d', [0.0]))
                break

            omega = self._get_motor_velocity()

            # 자석식 모듈이 떨어지지 않도록 복귀 슬램을 부드럽게(0.4→0.22, clip 0.2→0.15)
            if cur_rad > 1.57:
                duty = -0.22
            elif cur_rad < -1.57:
                duty = 0.22
            else:
                duty = np.clip(self.Kp * error_rad - self.Kd * omega, -0.15, 0.15)

            self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
            time.sleep(0.005)

    def apply_action(self, actions):
        pwm = float(actions.item()) * self.action_scale
        self.card.write_pwm(self.pwm_ch, 1, array('d', [pwm]))
