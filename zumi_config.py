import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass


class Commands(str, Enum):
    PREPARE = "PREPARE"
    START_SYNC = "START_SYNC"
    STOP_SYNC = "STOP_SYNC"
    START_DOWNLOAD = "START_DOWNLOAD"
    DISCARD_RUN = "DISCARD_RUN"
    EXIT = "EXIT"


class NodeStatus(str, Enum):
    INIT = "INIT"
    IDLE = "IDLE"
    READY = "READY"
    RECORDING = "RECORDING"
    SAVING = "SAVING"
    ERROR = "ERROR"


@dataclass
class NetworkConfig:
    ORCHESTRATOR_IP: str = os.getenv("ZUMI_ORCHESTRATOR_IP", "127.0.0.1")
    PORT_BASE: int = int(os.getenv("ZUMI_PORT_BASE", 5555))
    CMD_PORT: int = int(os.getenv("ZUMI_CMD_PORT", PORT_BASE))
    STATUS_PORT: int = int(os.getenv("ZUMI_STATUS_PORT", PORT_BASE + 1))


@dataclass
class StorageConfig:
    DATA_DIR: Path = Path("data")
    VIDEO_DIR: Path = Path("data/videos")


NET_CONF = NetworkConfig()
STORAGE_CONF = StorageConfig()

STORAGE_CONF.DATA_DIR.mkdir(exist_ok=True, parents=True)
STORAGE_CONF.VIDEO_DIR.mkdir(exist_ok=True, parents=True)
