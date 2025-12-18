#%%
from DM_CAN import *
import serial
from dotenv import load_dotenv

import os

MOTOR_SlaveID = int(os.getenv("MOTOR_SLAVE_ID", "0x07"), 16)
MOTOR_MasterID = int(os.getenv("MOTOR_MASTER_ID", "0x17"), 16)
MOTOR_SERIAL_PORT = os.getenv("MOTOR_SERIAL_PORT", "/dev/ttyACM0")

Motor1=Motor(DM_Motor_Type.DMH3510,MOTOR_SlaveID, MOTOR_MasterID)
serial_device = serial.Serial(MOTOR_SERIAL_PORT, 921600, timeout=0.5)
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
