from quanser.hardware import HIL, MAX_STRING_LENGTH
from array import array
import numpy as np, time, math

card = HIL("qube_servo3_usb", "0")
try:
    # ① PWM 모드 켜기
    card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)        # 필수! :contentReference[oaicite:8]{index=8}
    input_channels = array('I', [1])
    output_channels = array('I', [0])
    num_input_channels = len(input_channels)
    num_output_channels = len(output_channels)
    card.set_digital_directions(input_channels, num_input_channels, output_channels, num_output_channels)

    # ② 앰프 Enable
    card.write_digital(array('I',[0]),1,array('I',[1]))

    pwm_ch  = array('I',[0])
    samples = 5000
    Ts      = 0.1        # 1 kHz 루프

    for k in range(samples):
        if k % 2 == 0:
            duty = 0.1
        else:
            duty = -0.1
        card.write_pwm(pwm_ch, 1, array('d',[duty]))
        time.sleep(Ts)

finally:
    card.write_pwm(array('I',[0]),1,array('d',[0.0]))
    card.write_digital(array('I',[0]),1,array('I',[0]))
    card.close()