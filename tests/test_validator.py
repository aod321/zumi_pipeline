import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import validator


def build_motor_npz(path: Path):
    # Minimal motor dataset: timestamps and dummy values
    data = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0],
            [1.01, 0.1, 0.1, 0.0, 1],
            [1.02, 0.2, 0.2, 0.0, 2],
        ],
        dtype=np.float64,
    )
    np.savez_compressed(path, data=data, columns=np.array(["ts", "pos", "vel", "tau", "iter"]))


def test_validator_passes_with_motor_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id = "run_001"
        data_dir = Path(tmpdir)
        video_dir = data_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        motor_path = data_dir / f"{run_id}_motor.npz"
        build_motor_npz(motor_path)

        # Point validator to temp dirs
        validator.DATA_DIR = data_dir
        validator.VIDEO_DIR = video_dir

        assert validator.validate(run_id) is False


def test_validator_fails_when_motor_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id = "run_999"
        data_dir = Path(tmpdir)
        video_dir = data_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        validator.DATA_DIR = data_dir
        validator.VIDEO_DIR = video_dir

        assert validator.validate(run_id) is False
