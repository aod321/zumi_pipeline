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


class DownloadNode(NodeHTTPService):
    def __init__(self):
        self.download_triggered = False
        super().__init__(name="downloader", host="127.0.0.1", port=0)

    def on_init(self):
        self.status = NodeStatus.IDLE

    def main_loop(self):
        while self.is_running:
            time.sleep(0.01)

    def on_start_download(self):
        self.download_triggered = True

    def on_start_recording(self, run_id, episode=None):
        self.status = NodeStatus.RECORDING

    def on_stop_recording(self):
        self.status = NodeStatus.IDLE


def test_download_endpoint_and_status():
    node = DownloadNode()
    with TestClient(node.app) as client:
        resp = client.post("/download")
        assert resp.status_code == 200
        assert node.download_triggered is True

        # Prepare to READY, start, stop
        client.post("/prepare", json={"run_id": "r1", "episode": 1})
        resp = client.post("/start", json={"run_id": "r1", "episode": 1})
        assert resp.status_code in (200, 202)
        time.sleep(0.05)
        assert client.get("/status").json()["status"] == NodeStatus.RECORDING.value

        resp = client.post("/stop", json={})
        assert resp.status_code in (200, 202)
        time.sleep(0.05)
        assert client.get("/status").json()["status"] == NodeStatus.IDLE.value
