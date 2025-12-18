import errno
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from select import select

import click
import zmq

from zumi_config import Commands, NET_CONF, STORAGE_CONF


class Orchestrator:
    EXPECTED_NODES = ["go_pro_node", "DM3510"]

    def __init__(self, cmd_port=None, status_port=None):
        self.context = zmq.Context()

        self.cmd_port = cmd_port or NET_CONF.CMD_PORT
        self.status_port = status_port or NET_CONF.STATUS_PORT

        self.cmd_socket = self.context.socket(zmq.PUB)
        self._bind_socket(self.cmd_socket, self.cmd_port, "command publisher")

        self.status_socket = self.context.socket(zmq.SUB)
        self._bind_socket(self.status_socket, self.status_port, "status subscriber")
        self.status_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.nodes = {}
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _bind_socket(self, socket, port, description):
        address = f"tcp://*:{port}"
        try:
            socket.bind(address)
        except zmq.ZMQError as exc:
            socket.close(0)
            if exc.errno in (getattr(zmq, "EADDRINUSE", None), errno.EADDRINUSE):
                raise RuntimeError(
                    f"Port {port} already in use for {description}. "
                    "Set ZUMI_CMD_PORT/ZUMI_STATUS_PORT (or ZUMI_PORT_BASE) to unused ports, "
                    "or pass --cmd-port/--status-port."
                ) from exc
            raise

    def _monitor_loop(self):
        while self.running:
            try:
                if not self.status_socket.poll(100):
                    continue
                msg = self.status_socket.recv_json()
                name = msg.get("node")
                status = msg.get("status")
                self.nodes[name] = {"status": status, "last_seen": time.time()}
            except Exception:
                continue

    def get_node_status_summary(self):
        now = time.time()
        processed = set()
        parts = []

        def format_part(name, info=None):
            if info:
                status = info["status"]
                is_online = (now - info["last_seen"]) < 3.0

                if not is_online:
                    return click.style(f"× {name}: OFFLINE", fg="red")

                if status in ("READY", "IDLE"):
                    return click.style(f"√ {name}: {status}", fg="green")
                elif status == "RECORDING":
                    return click.style(f"● {name}: REC", fg="red", blink=True)
                elif status == "SAVING":
                    return click.style(f"⟳ {name}: SAVING", fg="yellow")
                elif status == "ERROR":
                    return click.style(f"! {name}: ERROR", fg="red", bold=True)
                else:
                    return click.style(f"√ {name}: {status}", fg="cyan")
            else:
                return click.style(f"○ {name}: MISSING", fg="yellow", dim=True)

        for name in self.EXPECTED_NODES:
            processed.add(name)
            info = self.nodes.get(name)
            parts.append(format_part(name, info))

        for name, info in self.nodes.items():
            if name not in processed:
                parts.append(format_part(name, info))

        return " | ".join(parts)

    def are_nodes_ready(self):
        now = time.time()
        for name in self.EXPECTED_NODES:
            info = self.nodes.get(name)
            if not info:
                return False
            if now - info["last_seen"] > 3.0:
                return False
            if info["status"] != "READY":
                return False
        return True

    def send_prepare(self, run_id, episode):
        self.cmd_socket.send_json({"cmd": Commands.PREPARE, "payload": {"run_id": run_id, "episode": episode}})

    def send_start(self, run_id, episode, delay=0.1):
        start_ts = time.time() + delay
        self.cmd_socket.send_json(
            {
                "cmd": Commands.START_SYNC,
                "payload": {"run_id": run_id, "episode": episode, "start_time": start_ts},
            }
        )
        return start_ts

    def send_stop(self, delay=0.1):
        stop_ts = time.time() + delay
        self.cmd_socket.send_json({"cmd": Commands.STOP_SYNC, "payload": {"stop_time": stop_ts}})
        return stop_ts

    def send_discard(self, run_id, episode=None):
        payload = {"run_id": run_id}
        if episode is not None:
            payload["episode"] = episode
        self.cmd_socket.send_json({"cmd": Commands.DISCARD_RUN, "payload": payload})

    def send_download(self):
        self.cmd_socket.send_json({"cmd": Commands.START_DOWNLOAD, "payload": {}})

    def send_exit(self):
        self.cmd_socket.send_json({"cmd": Commands.EXIT, "payload": {}})

    def close(self):
        self.running = False
        try:
            if hasattr(self, "cmd_socket"):
                self.cmd_socket.close(0)
            if hasattr(self, "status_socket"):
                self.status_socket.close(0)
        finally:
            self.context.term()

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
    # Backward compatibility: treat plain run_id prefix as episode 1
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


@click.command()
@click.option("--delay", default=0.1, help="Sync delay latency in seconds.")
@click.option(
    "--cmd-port",
    type=int,
    default=None,
    help="Command publish port (defaults to ZUMI_CMD_PORT or ZUMI_PORT_BASE).",
)
@click.option(
    "--status-port",
    type=int,
    default=None,
    help="Status subscribe port (defaults to ZUMI_STATUS_PORT or ZUMI_PORT_BASE+1).",
)
@click.option("--run-id", default=None, help="Resume an existing run id.")
@click.option("--tag", default=None, help="Optional tag when creating a new run id.")
@click.option(
    "--validation-mode",
    type=click.Choice(["episode", "run"]),
    default="episode",
    show_default=True,
    help="Validation mode indicator for UI.",
)
def main(delay, cmd_port, status_port, run_id, tag, validation_mode):
    try:
        orch = Orchestrator(cmd_port=cmd_port, status_port=status_port)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))

    active_run_id = run_id or generate_run_id(tag)
    next_episode = infer_next_episode(active_run_id)

    click.clear()
    click.secho("=== Zumi Orchestrator 2.0 ===", fg="cyan", bold=True)
    click.echo(f"[+] CMD_PORT={orch.cmd_port}, STATUS_PORT={orch.status_port}")
    click.echo(f"[+] Data Dir: {STORAGE_CONF.DATA_DIR}")
    click.echo(f"[+] Run ID: {active_run_id} | Next episode: ep{format_episode(next_episode)}")
    click.echo(f"[+] Validation mode: {validation_mode}")
    click.echo("Press Enter to Start/Stop, d to discard last, q to quit.")

    is_recording = False
    current_episode = None
    last_record = None
    download_notice_shown = False

    refresh_interval = 0.2
    prepare_timeout = 5.0

    try:
        while True:
            status_line = orch.get_node_status_summary()
            click.echo(f"\rNodes: {status_line}\033[K", nl=False)
            sys.stdout.flush()

            # Warn if GoPro is still saving/downloading
            go_pro_info = orch.nodes.get("go_pro_node")
            go_pro_state = go_pro_info.get("status") if go_pro_info else None
            if go_pro_state == "SAVING" and not is_recording:
                if not download_notice_shown:
                    click.secho("\n[DL] Download in progress, will validate afterward.", fg="yellow")
                    download_notice_shown = True
            else:
                download_notice_shown = False

            key_ready, _, _ = select([sys.stdin], [], [], refresh_interval)
            if not key_ready:
                continue

            c = click.getchar()
            if c in ("\r", "\n"):
                if not is_recording:
                    current_episode = next_episode
                    click.echo(f"\nPreparing {active_run_id} ep{format_episode(current_episode)}...")
                    orch.send_prepare(active_run_id, current_episode)
                    start_deadline = time.time() + prepare_timeout
                    while time.time() < start_deadline:
                        if orch.are_nodes_ready():
                            break
                        time.sleep(refresh_interval)
                    if not orch.are_nodes_ready():
                        click.secho("Warning: Nodes not READY. Recording not started.", fg="yellow")
                        continue
                    orch.send_start(active_run_id, current_episode, delay=delay)
                    is_recording = True
                    last_record = (active_run_id, current_episode)
                    next_episode += 1
                    click.secho(
                        f"STARTED {active_run_id} ep{format_episode(current_episode)}", fg="green"
                    )
                else:
                    orch.send_stop(delay=delay)
                    is_recording = False
                    click.secho(
                        f"STOPPED {active_run_id} ep{format_episode(current_episode)}", fg="red"
                    )
            elif c == "d":
                if last_record and not is_recording:
                    lr_run, lr_ep = last_record
                    orch.send_discard(lr_run, lr_ep)
                    click.secho(
                        f"DISCARDED {lr_run} ep{format_episode(lr_ep)}", fg="red"
                    )
                    last_record = None
                else:
                    click.echo("\nNo episode to discard or currently recording.")
            elif c == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        click.echo("\nExiting...")
        orch.close()


if __name__ == "__main__":
    main()
