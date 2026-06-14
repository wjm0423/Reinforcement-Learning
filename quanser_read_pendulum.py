from quanser.hardware import HIL
from array import array
import math
import time


def read_pendulum_angle(card):
    enc_ch = array('I', [1])
    enc_val = array('l', [0])
    pend_init_count = 0

    while True:
        card.read_encoder(enc_ch, 1, enc_val)

        alpha = (enc_val[0] - pend_init_count) * 2 * math.pi / 2048.0
        alpha = ((alpha + math.pi) % (2 * math.pi)) - math.pi
        alpha_deg = alpha * 180 / math.pi

        print(f"pendulum α = {alpha_deg} °")
        time.sleep(0.5)


def main():
    card = HIL("qube_servo3_usb", "0")
    try:
        read_pendulum_angle(card)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        card.close()


if __name__ == "__main__":
    main()