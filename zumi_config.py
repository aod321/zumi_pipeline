import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass


class NodeStatus(str, Enum):
    INIT = "INIT"
    IDLE = "IDLE"
    READY = "READY"
    RECORDING = "RECORDING"
    SAVING = "SAVING"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"
    RECOVERING = "RECOVERING"


@dataclass
class StorageConfig:
    DATA_DIR: Path = Path("data")


@dataclass
class ZMQConfig:
    ORCHESTRATOR_IP: str = "127.0.0.1"
    STATUS_PORT: int = 5556


@dataclass
class HttpNodeConfig:
    GOPRO_URL: str = "http://127.0.0.1:8001"
    MOTOR_URL: str = "http://127.0.0.1:8002"
    GOPRO_HOST: str = "0.0.0.0"
    GOPRO_PORT: int = 8001
    MOTOR_HOST: str = "0.0.0.0"
    MOTOR_PORT: int = 8002


@dataclass
class MotorConfig:
    DRIVER: str = "dm"  # "dm" or "mock"
    SLAVE_ID: int = 0x07
    MASTER_ID: int = 0x17
    SERIAL_PORT: str = "/dev/dm_can0"


@dataclass
class GoProConfig:
    SN: str = None  # Serial number (optional, for IP derivation)
    IP: str = None  # Direct IP (optional, auto-discover if None)


STORAGE_CONF = StorageConfig()
HTTP_CONF = HttpNodeConfig()
ZMQ_CONF = ZMQConfig()
MOTOR_CONF = MotorConfig()
GOPRO_CONF = GoProConfig()

STORAGE_CONF.DATA_DIR.mkdir(exist_ok=True, parents=True)
