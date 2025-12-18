import importlib
import os
import socket
import sys
import time

import zmq

# Ensure repo root on path for direct execution
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def get_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def recv_json_with_timeout(sock, timeout_ms=500):
    if sock.poll(timeout_ms):
        return sock.recv_json()
    raise TimeoutError("Timeout waiting for message on ZMQ socket")


def test_orchestrator_commands_and_status():
    # Configure unique ports to avoid collisions
    cmd_port = get_free_port()
    status_port = get_free_port()
    os.environ["ZUMI_CMD_PORT"] = str(cmd_port)
    os.environ["ZUMI_STATUS_PORT"] = str(status_port)

    import orchestrator as orch_module

    orch_module = importlib.reload(orch_module)
    orch = orch_module.Orchestrator()

    ctx = zmq.Context.instance()

    # Listener for commands the orchestrator publishes
    cmd_sub = ctx.socket(zmq.SUB)
    cmd_sub.connect(f"tcp://localhost:{cmd_port}")
    cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    # Publisher to feed heartbeats to orchestrator
    status_pub = ctx.socket(zmq.PUB)
    status_pub.connect(f"tcp://localhost:{status_port}")

    try:
        # Allow PUB/SUB handshake to establish
        time.sleep(0.2)

        # Send heartbeat to mark node READY (send twice for reliability)
        status_pub.send_json({"node": "motor", "status": "READY", "ts": time.time()})
        time.sleep(0.05)
        status_pub.send_json({"node": "motor", "status": "READY", "ts": time.time()})
        time.sleep(0.2)
        assert orch.are_nodes_ready() is True

        # Prepare
        orch.send_prepare("run_001")
        msg = recv_json_with_timeout(cmd_sub)
        assert msg["cmd"] == orch_module.Commands.PREPARE
        assert msg["payload"]["run_id"] == "run_001"

        # Start
        start_ts = orch.send_start("run_001", delay=0.05)
        msg = recv_json_with_timeout(cmd_sub)
        assert msg["cmd"] == orch_module.Commands.START_SYNC
        assert msg["payload"]["run_id"] == "run_001"
        assert abs(msg["payload"]["start_time"] - start_ts) < 0.2

        # Stop
        stop_ts = orch.send_stop(delay=0.05)
        msg = recv_json_with_timeout(cmd_sub)
        assert msg["cmd"] == orch_module.Commands.STOP_SYNC
        assert abs(msg["payload"]["stop_time"] - stop_ts) < 0.2

        # Discard
        orch.send_discard("run_001")
        msg = recv_json_with_timeout(cmd_sub)
        assert msg["cmd"] == orch_module.Commands.DISCARD_RUN
        assert msg["payload"]["run_id"] == "run_001"

        # Exit
        orch.send_exit()
        msg = recv_json_with_timeout(cmd_sub)
        assert msg["cmd"] == orch_module.Commands.EXIT
    finally:
        cmd_sub.close(0)
        status_pub.close(0)
        try:
            orch.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Allow running directly for quick sanity checks without pytest
    test_orchestrator_commands_and_status()
    print("test_orchestrator_commands_and_status: PASS")
