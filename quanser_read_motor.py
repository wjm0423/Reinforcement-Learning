from quanser.hardware import HIL
from array   import array
import time
import math

card = HIL("qube_servo3_usb", "0")
try:
    enc_ch  = array('I', [0])             # 모터 = 채널 0
    enc_val = array('l', [0])

    # ── (1) 첫 번째 읽기 → bias 저장 ───────────────────────────────
    card.read_encoder(enc_ch, 1, enc_val)
    bias_count = enc_val[0]               # 현재 카운트를 0° 기준으로 사용

    # ── (2) 계속 각도 출력 ────────────────────────────────────────
    while True:
        card.read_encoder(enc_ch, 1, enc_val)
        count = enc_val[0] - bias_count   # 오프셋 보정

        alpha = count * 2*math.pi / 2048.0 
        alpha_deg = alpha * 180/math.pi

        print(f"motor α = {alpha_deg:+.1f} °")
        time.sleep(0.5)

finally:
    card.close()
