from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MotorState:
    position: float
    velocity: float
    torque: float


class MotorDriver(ABC):
    @abstractmethod
    def enable(self):
        ...

    @abstractmethod
    def disable(self):
        ...

    @abstractmethod
    def set_zero(self):
        ...

    @abstractmethod
    def command(self, torque: float, position: float = 0.0, velocity: float = 0.0, kp: float = 0.0, kd: float = 0.0):
        ...

    @abstractmethod
    def get_state(self) -> MotorState:
        ...

    @abstractmethod
    def shutdown(self):
        ...
