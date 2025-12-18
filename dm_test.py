import pytest

pytest.skip("Legacy DM_CAN test skipped in FastAPI refactor branch", allow_module_level=True)

#%%
from DM_CAN import *
import serial

from zumi_config import MOTOR_CONF

Motor1=Motor(DM_Motor_Type.DMH3510, MOTOR_CONF.SLAVE_ID, MOTOR_CONF.MASTER_ID)
serial_device = serial.Serial(MOTOR_CONF.SERIAL_PORT, 921600, timeout=0.5)
MotorControl1=MotorControl(serial_device)
MotorControl1.addMotor(Motor1)
if MotorControl1.switchControlMode(Motor1,Control_Type.MIT): # MIT模式可以更好地控制位置和感受力反馈
    print("switch MIT success for Motor1")
else:
    print("switch MIT failed for Motor1")
MotorControl1.enable(Motor1)
MotorControl1.set_zero_position(Motor1)

#%%
import time
rate = 200 # 200Hz
dt = 1.0 / rate
t_start = time.monotonic()

while True:
    t_now = time.monotonic()
    if t_now - t_start >= dt:
        MotorControl1.controlMIT(Motor1, 0.0, 0.0, 0.0, 0.0, 0.0)
        pos = Motor1.getPosition()
        vel = Motor1.getVelocity()
        tau = Motor1.getTorque()
        print(f"pos: {pos:.4f}, vel: {vel:.4f}, tau: {tau:.4f}")
        t_start = t_now
    time.sleep(dt)
