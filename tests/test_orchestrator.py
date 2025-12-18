import os
import sys
import tempfile
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
        videos_dir = data_dir / "videos"
        data_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy files to infer episodes
        (data_dir / f"{run_id}_ep001_motor.npz").touch()
        (videos_dir / f"{run_id}_ep002_video.MP4").touch()

        # Point STORAGE_CONF to temp dirs
        orchestrator.STORAGE_CONF.DATA_DIR = data_dir
        orchestrator.STORAGE_CONF.VIDEO_DIR = videos_dir

        assert orchestrator.infer_next_episode(run_id) == 3


def test_status_helpers():
    status_map = {
        "go_pro_node": {"status": "READY"},
        "DM3510": {"status": "READY"},
    }
    line = orchestrator.format_status_line(status_map)
    assert "go_pro_node" in line and "DM3510" in line
    assert orchestrator.all_ready(status_map, ["go_pro_node", "DM3510"]) is True

    status_map["DM3510"]["status"] = "IDLE"
    assert orchestrator.all_ready(status_map, ["go_pro_node", "DM3510"]) is False
