import time
from random import gauss

from motor_interface import MotorDriver, MotorState


class MockMotorDriver(MotorDriver):
    """
    Lightweight physics mock so node_motor can run without hardware.
    """

    def __init__(self):
        self.enabled = False
        self.state = MotorState(0.0, 0.0, 0.0)
        self.last_ts = time.monotonic()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_zero(self):
        self.state = MotorState(0.0, 0.0, 0.0)

    def command(self, torque: float, position: float = 0.0, velocity: float = 0.0, kp: float = 0.0, kd: float = 0.0):
        now = time.monotonic()
        dt = now - self.last_ts
        self.last_ts = now

        if not self.enabled:
            return

        # Very simple integrator model: torque -> accel -> velocity/position
        accel = torque
        vel = self.state.velocity + accel * dt
        pos = self.state.position + vel * dt

        # Add tiny noise to avoid perfectly flat data
        vel += gauss(0, 1e-4)
        pos += gauss(0, 1e-4)

        self.state = MotorState(pos, vel, torque)

    def get_state(self) -> MotorState:
        return self.state

    def shutdown(self):
        self.disable()
