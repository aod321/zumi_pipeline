import os
import re
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timezone
from select import select

import click
import requests
import zmq

from zumi_config import HTTP_CONF, NodeStatus, STORAGE_CONF, ZMQ_CONF


def _sanitize_tag(tag: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "", tag or "").strip("_")
    return clean


def generate_run_id(tag=None) -> str:
    base = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    clean_tag = _sanitize_tag(tag)
    return f"{base}_{clean_tag}" if clean_tag else base


def _extract_episode(run_id: str, name: str) -> int:
    ep_match = re.search(rf"^{re.escape(run_id)}_ep(\d+)", name)
    if ep_match:
        try:
            return int(ep_match.group(1))
        except ValueError:
            return None
    legacy_match = re.search(rf"^{re.escape(run_id)}_", name)
    if legacy_match:
        return 1
    return None


def infer_next_episode(run_id: str) -> int:
    max_ep = 0
    for directory in [STORAGE_CONF.DATA_DIR, STORAGE_CONF.VIDEO_DIR]:
        if not directory.exists():
            continue
        for name in os.listdir(directory):
            ep = _extract_episode(run_id, name)
            if ep:
                max_ep = max(max_ep, ep)
    return max_ep + 1 if max_ep else 1


def format_episode(ep: int) -> str:
    return f"{ep:03d}"


class NodeClient:
    def __init__(self, name: str, base_url: str, timeout: float = 2.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload=None):
        try:
            resp = requests.post(f"{self.base_url}{path}", json=payload or {}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"error": str(exc)}

    def prepare(self, run_id, episode):
        return self._post("/prepare", {"run_id": run_id, "episode": episode})

    def start(self, run_id, episode, start_time=None):
        return self._post("/start", {"run_id": run_id, "episode": episode, "start_time": start_time})

    def stop(self, stop_time=None):
        return self._post("/stop", {"stop_time": stop_time})

    def discard(self, run_id, episode=None):
        payload = {"run_id": run_id}
        if episode is not None:
            payload["episode"] = episode
        return self._post("/discard", payload)

    def download(self):
        return self._post("/download", {})


def format_status_line(status_map, expected):
    now = time.time()
    parts = []
    seen = set()
    for name in expected:
        info = status_map.get(name)
        seen.add(name)
        if not info:
            parts.append(click.style(f"○ {name}: MISSING", fg="yellow", dim=True))
            continue
        status = info.get("status")
        # Immediately recognize OFFLINE status from node
        if status == NodeStatus.OFFLINE.value or now - info.get("ts", 0) > 3.0:
            parts.append(click.style(f"× {name}: OFFLINE", fg="red"))
            continue

        if status == NodeStatus.READY.value:
            parts.append(click.style(f"√ {name}: READY", fg="green"))
        elif status == NodeStatus.IDLE.value:
            parts.append(click.style(f"√ {name}: IDLE", fg="green"))
        elif status == NodeStatus.RECORDING.value:
            parts.append(click.style(f"● {name}: REC", fg="red"))
        elif status == NodeStatus.SAVING.value:
            parts.append(click.style(f"⟳ {name}: SAVING", fg="yellow"))
        elif status == NodeStatus.ERROR.value:
            parts.append(click.style(f"! {name}: ERROR", fg="red", bold=True))
        elif status == NodeStatus.RECOVERING.value:
            parts.append(click.style(f"⟳ {name}: RECOVERING", fg="yellow", bold=True))
        else:
            parts.append(click.style(f"√ {name}: {status}", fg="cyan"))

    for name, info in status_map.items():
        if name in seen:
            continue
        status = info.get("status")
        parts.append(click.style(f"{name}: {status}", fg="cyan"))
    return " | ".join(parts)


def classify(status_map, expected, stale_after=3.0):
    now = time.time()
    all_ready = True
    any_offline = False
    any_error = False
    any_recording = False
    any_recovering = False

    for name in expected:
        info = status_map.get(name)
        if not info:
            any_offline = True
            all_ready = False
            continue
        status = info.get("status")
        # Immediately recognize OFFLINE status from node
        if status == NodeStatus.OFFLINE.value or now - info.get("ts", 0) > stale_after:
            any_offline = True
            all_ready = False
            continue
        if status == NodeStatus.ERROR.value:
            any_error = True
            all_ready = False
        elif status == NodeStatus.RECORDING.value:
            any_recording = True
            all_ready = False
        elif status == NodeStatus.RECOVERING.value:
            any_recovering = True
            all_ready = False
        elif status != NodeStatus.READY.value:
            all_ready = False

    return {
        "all_ready": all_ready,
        "any_offline": any_offline,
        "any_error": any_error,
        "any_recording": any_recording,
        "any_recovering": any_recovering,
    }


@click.command()
@click.option("--delay", default=0.1, help="Sync delay latency in seconds.")
@click.option("--run-id", default=None, help="Resume an existing run id.")
@click.option("--tag", default=None, help="Optional tag when creating a new run id.")
@click.option(
    "--validation-mode",
    type=click.Choice(["episode", "run"]),
    default="episode",
    show_default=True,
    help="Validation mode indicator for UI.",
)
def main(delay, run_id, tag, validation_mode):
    clients = [
        NodeClient("go_pro_node", HTTP_CONF.GOPRO_URL),
        NodeClient("DM3510", HTTP_CONF.MOTOR_URL),
    ]
    expected_names = [c.name for c in clients]

    ctx = zmq.Context()
    status_socket = ctx.socket(zmq.SUB)
    status_socket.bind(f"tcp://*:{ZMQ_CONF.STATUS_PORT}")
    status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    nodes = {}
    running = True

    def monitor_loop():
        while running:
            try:
                if not status_socket.poll(100):
                    continue
                msg = status_socket.recv_json()
                name = msg.get("node")
                if not name:
                    continue
                nodes[name] = {
                    "status": msg.get("status"),
                    "ts": time.time(),
                }
            except Exception:
                continue

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    active_run_id = run_id or generate_run_id(tag)
    next_episode = infer_next_episode(active_run_id)

    click.clear()
    click.secho("=== Zumi Orchestrator 2.0 (HTTP) ===", fg="cyan", bold=True)
    click.echo(f"[+] Data Dir: {STORAGE_CONF.DATA_DIR}")
    click.echo(f"[+] Run ID: {active_run_id} | Next episode: ep{format_episode(next_episode)}")
    click.echo(f"[+] Validation mode: {validation_mode}")
    click.echo("Press Enter to Start/Stop, d to discard last, q to quit.")

    is_recording = False
    current_episode = None
    last_record = None
    refresh_interval = 0.2
    state_ready = False

    # Save terminal settings and set cbreak mode
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            status_line = format_status_line(nodes, expected_names)
            click.echo(f"\rNodes: {status_line}\033[K", nl=False)
            sys.stdout.flush()

            key_ready, _, _ = select([sys.stdin], [], [], refresh_interval)
            if not key_ready:
                # auto-advance if we were waiting for READY and现在全READY
                if not is_recording and not state_ready:
                    status_flags = classify(nodes, expected_names)
                    if status_flags["all_ready"]:
                        state_ready = True
                continue

            c = sys.stdin.read(1)
            if c in ("\r", "\n"):
                status_flags = classify(nodes, expected_names)
                if is_recording:
                    stop_ts = time.time() + delay
                    for client in clients:
                        client.stop(stop_time=stop_ts)
                    is_recording = False
                    click.secho(f"STOPPED {active_run_id} ep{format_episode(current_episode)}", fg="red")
                    continue

                # not recording: step-wise control
                if status_flags["any_offline"]:
                    click.secho("\nNodes offline/missing. Prepare not sent.", fg="yellow")
                    continue
                if status_flags["any_error"]:
                    click.secho("\nNodes in ERROR. Resolve before starting.", fg="red")
                    continue

                if status_flags["all_ready"]:
                    start_ts = time.time() + delay
                    for client in clients:
                        client.start(active_run_id, current_episode, start_time=start_ts)
                    is_recording = True
                    last_record = (active_run_id, current_episode)
                    next_episode += 1
                    click.secho(f"STARTED {active_run_id} ep{format_episode(current_episode)}", fg="green")
                    state_ready = False
                else:
                    # send prepare to non-ready nodes
                    current_episode = next_episode
                    click.echo(f"\nPreparing {active_run_id} ep{format_episode(current_episode)}...")
                    for client in clients:
                        client.prepare(active_run_id, current_episode)
                    state_ready = False
                    click.secho("Waiting for all nodes to become READY.", fg="yellow")
            elif c == "d":
                if last_record and not is_recording:
                    lr_run, lr_ep = last_record
                    for client in clients:
                        client.discard(lr_run, lr_ep)
                    click.secho(f"DISCARDED {lr_run} ep{format_episode(lr_ep)}", fg="red")
                    last_record = None
                else:
                    click.echo("\nNo episode to discard or currently recording.")
            elif c == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        running = False
        status_socket.close(0)
        ctx.term()
        click.echo("\nExiting...")


if __name__ == "__main__":
    main()
