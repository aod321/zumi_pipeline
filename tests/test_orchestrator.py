import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("click")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import orchestrator  # noqa: E402


def test_generate_and_infer_episode():
    run_id = orchestrator.generate_run_id("test")
    assert run_id.startswith("run_")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create run_id directory with dummy files
        run_dir = data_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"{run_id}_ep001_motor.npz").touch()
        (run_dir / f"{run_id}_ep002_video.MP4").touch()

        # Point STORAGE_CONF to temp dir
        orchestrator.STORAGE_CONF.DATA_DIR = data_dir

        assert orchestrator.infer_next_episode(run_id) == 3


def test_status_helpers():
    expected = ["go_pro_node", "DM3510"]
    now = time.time()
    status_map = {
        "go_pro_node": {"status": "READY", "ts": now},
        "DM3510": {"status": "READY", "ts": now},
    }
    line = orchestrator.format_status_line(status_map, expected)
    assert "go_pro_node" in line and "DM3510" in line
    result = orchestrator.classify(status_map, expected)
    assert result["all_ready"] is True

    status_map["DM3510"]["status"] = "IDLE"
    result = orchestrator.classify(status_map, expected)
    assert result["all_ready"] is False
