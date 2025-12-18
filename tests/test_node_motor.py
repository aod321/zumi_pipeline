import os
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from zumi_core import NodeHTTPService  # noqa: E402
from zumi_config import NodeStatus  # noqa: E402


class DummyNode(NodeHTTPService):
    def __init__(self):
        self.prepare_called = False
        self.started = False
        self.stopped = False
        self.discarded = None
        self.download_triggered = False
        super().__init__(name="dummy", host="127.0.0.1", port=0)

    def on_init(self):
        self.status = NodeStatus.IDLE

    def main_loop(self):
        while self.is_running:
            time.sleep(0.01)

    def on_prepare(self, run_id, episode=None):
        self.prepare_called = True
        return True

    def on_start_recording(self, run_id, episode=None):
        self.started = True

    def on_stop_recording(self):
        self.stopped = True

    def on_discard_run(self, run_id, episode=None):
        self.discarded = (run_id, episode)

    def on_start_download(self):
        self.download_triggered = True


def test_state_machine_and_endpoints():
    node = DummyNode()
    with TestClient(node.app) as client:
        # start should fail before prepare
        resp = client.post("/start", json={"run_id": "r1", "episode": 1})
        assert resp.status_code == 409

        # prepare -> READY
        resp = client.post("/prepare", json={"run_id": "r1", "episode": 1})
        assert resp.status_code == 200
        assert node.prepare_called is True

        # start -> RECORDING
        resp = client.post("/start", json={"run_id": "r1", "episode": 1})
        assert resp.status_code == 200 or resp.status_code == 202
        time.sleep(0.1)
        status = client.get("/status").json()
        assert status["status"] == NodeStatus.RECORDING.value
        assert status["run_id"] == "r1"
        assert node.started is True

        # stop -> back to IDLE
        resp = client.post("/stop", json={})
        assert resp.status_code == 200 or resp.status_code == 202
        time.sleep(0.1)
        status = client.get("/status").json()
        assert status["status"] == NodeStatus.IDLE.value
        assert status["run_id"] is None
        assert node.stopped is True

        # discard and download
        resp = client.post("/discard", json={"run_id": "r1", "episode": 1})
        assert resp.status_code == 200
        assert node.discarded == ("r1", 1)

        resp = client.post("/download")
        assert resp.status_code == 200
        assert node.download_triggered is True
