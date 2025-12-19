import os
import re
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from select import select
from typing import Optional

import click
import requests
import zmq

from zumi_config import HTTP_CONF, NodeStatus, STORAGE_CONF, ZMQ_CONF
from validator import validate


# =============================================================================
# Sound Player
# =============================================================================

SOUNDS = {
    "ready": "/usr/share/sounds/freedesktop/stereo/complete.oga",
    "start": "/usr/share/sounds/freedesktop/stereo/camera-shutter.oga",
    "stop": "/usr/share/sounds/freedesktop/stereo/bell.oga",
    "error": "/usr/share/sounds/freedesktop/stereo/suspend-error.oga",
}


class SoundPlayer:
    def __init__(self):
        self._alert_thread = None
        self._alert_stop = threading.Event()

    def play(self, sound_key):
        subprocess.Popen(
            ["paplay", SOUNDS[sound_key]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def start_alert(self):
        if self._alert_thread and self._alert_thread.is_alive():
            return
        self._alert_stop.clear()
        self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self._alert_thread.start()

    def _alert_loop(self):
        while not self._alert_stop.wait(2.0):
            self.play("error")

    def stop_alert(self):
        self._alert_stop.set()


sound = SoundPlayer()


# =============================================================================
# State Machine Design
# =============================================================================

class OrchestratorState(Enum):
    """Orchestrator state derived from node states."""
    IDLE = "idle"           # All nodes IDLE, can prepare
    READY = "ready"         # All nodes READY, can start
    RECORDING = "recording" # Any node RECORDING, can only stop
    SAVING = "saving"       # Any node SAVING, wait
    RECOVERING = "recovering"  # Any node recovering, wait
    ERROR = "error"         # Any node in error
    OFFLINE = "offline"     # Any node offline/missing


# Key -> Action mapping per state
ALLOWED_ACTIONS = {
    OrchestratorState.IDLE: {
        "enter": "prepare",
        "s": "download",
        "x": "discard",
        "v": "validate",
        "q": "quit",
    },
    OrchestratorState.READY: {
        "enter": "start",
        "s": "download",
        "x": "discard",
        "v": "validate",
        "q": "quit",
    },
    OrchestratorState.RECORDING: {
        "enter": "stop",
        # All other keys forbidden
    },
    OrchestratorState.SAVING: {
        # All forbidden, wait for completion
    },
    OrchestratorState.RECOVERING: {
        # All forbidden, wait for recovery
    },
    OrchestratorState.ERROR: {
        "q": "quit",  # Can only quit
    },
    OrchestratorState.OFFLINE: {
        "q": "quit",  # Can only quit
    },
}


# =============================================================================
# Result Type for Interface Contract
# =============================================================================

@dataclass
class Result:
    """Result type for node client calls."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

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
    """Infer next episode number by scanning the run_id directory."""
    run_dir = STORAGE_CONF.DATA_DIR / run_id
    max_ep = 0
    if run_dir.exists():
        for name in os.listdir(run_dir):
            ep = _extract_episode(run_id, name)
            if ep:
                max_ep = max(max_ep, ep)
    return max_ep + 1 if max_ep else 1


def format_episode(ep: int) -> str:
    if ep is None:
        return "???"
    return f"{ep:03d}"


# =============================================================================
# Node Client with Result Type
# =============================================================================

class NodeClient:
    def __init__(self, name: str, base_url: str, timeout: float = 2.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload=None) -> Result:
        try:
            resp = requests.post(
                f"{self.base_url}{path}",
                json=payload or {},
                timeout=self.timeout
            )
            resp.raise_for_status()
            return Result(success=True, data=resp.json())
        except requests.Timeout:
            return Result(success=False, error="timeout")
        except requests.HTTPError as e:
            return Result(success=False, error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return Result(success=False, error=str(e))

    def prepare(self, run_id, episode) -> Result:
        return self._post("/prepare", {"run_id": run_id, "episode": episode})

    def start(self, run_id, episode, start_time=None) -> Result:
        return self._post("/start", {"run_id": run_id, "episode": episode, "start_time": start_time})

    def stop(self, stop_time=None) -> Result:
        return self._post("/stop", {"stop_time": stop_time})

    def discard(self, run_id, episode=None) -> Result:
        payload = {"run_id": run_id}
        if episode is not None:
            payload["episode"] = episode
        return self._post("/discard", payload)

    def download(self) -> Result:
        return self._post("/download", {})


# =============================================================================
# Status Display and Classification
# =============================================================================

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
    """Classify node states into flags."""
    now = time.time()
    all_ready = True
    any_offline = False
    any_error = False
    any_recording = False
    any_recovering = False
    any_saving = False

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
        elif status == NodeStatus.SAVING.value:
            any_saving = True
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
        "any_saving": any_saving,
    }


def get_orchestrator_state(nodes, expected) -> OrchestratorState:
    """Derive orchestrator state from node states."""
    flags = classify(nodes, expected)

    if flags["any_offline"]:
        return OrchestratorState.OFFLINE
    if flags["any_error"]:
        return OrchestratorState.ERROR
    if flags["any_recovering"]:
        return OrchestratorState.RECOVERING
    if flags["any_recording"]:
        return OrchestratorState.RECORDING
    if flags["any_saving"]:
        return OrchestratorState.SAVING
    if flags["all_ready"]:
        return OrchestratorState.READY
    return OrchestratorState.IDLE


def get_pending_count(nodes) -> int:
    """Get pending download count from GoPro node."""
    info = nodes.get("go_pro_node", {})
    return info.get("pending_tasks", 0)


# =============================================================================
# Context for Action Functions
# =============================================================================

@dataclass
class OrchestratorContext:
    """Context passed to action functions."""
    clients: list
    nodes: dict
    expected_names: list
    run_id: str
    delay: float
    current_episode: Optional[int] = None
    next_episode: int = 1
    last_record: Optional[tuple] = None
    quit_confirmed: bool = False
    old_settings: any = None
    prev_state: Optional["OrchestratorState"] = None
    prepare_sent: bool = False


# =============================================================================
# Action Functions
# =============================================================================

def do_prepare(ctx: OrchestratorContext) -> bool:
    """Send prepare to all nodes."""
    ctx.current_episode = ctx.next_episode
    click.echo(f"\nPreparing {ctx.run_id} ep{format_episode(ctx.current_episode)}...")

    results = [client.prepare(ctx.run_id, ctx.current_episode) for client in ctx.clients]
    failed = [(c.name, r.error) for c, r in zip(ctx.clients, results) if not r.success]

    if failed:
        click.secho(f"Prepare failed: {failed}", fg="red")
        return False

    click.secho("Waiting for all nodes to become READY.", fg="yellow")
    return True


def do_start(ctx: OrchestratorContext) -> bool:
    """Start recording on all nodes."""
    # Ensure episode is set
    if ctx.current_episode is None:
        ctx.current_episode = ctx.next_episode

    start_ts = time.time() + ctx.delay
    results = [client.start(ctx.run_id, ctx.current_episode, start_time=start_ts)
               for client in ctx.clients]

    failed = [(c.name, r.error) for c, r in zip(ctx.clients, results) if not r.success]
    if failed:
        click.secho(f"\nStart failed: {failed}", fg="red")
        return False

    ctx.last_record = (ctx.run_id, ctx.current_episode)
    ctx.next_episode += 1
    ctx.prepare_sent = False
    sound.play("start")
    click.secho(f"\nSTARTED {ctx.run_id} ep{format_episode(ctx.current_episode)}", fg="green")
    return True


def do_stop(ctx: OrchestratorContext) -> bool:
    """Stop recording on all nodes."""
    stop_ts = time.time() + ctx.delay
    results = [client.stop(stop_time=stop_ts) for client in ctx.clients]

    failed = [(c.name, r.error) for c, r in zip(ctx.clients, results) if not r.success]
    if failed:
        click.secho(f"\nStop failed: {failed}", fg="red")
        return False

    sound.play("stop")
    click.secho(f"\nSTOPPED {ctx.run_id} ep{format_episode(ctx.current_episode)}", fg="red")
    return True


def do_download(ctx: OrchestratorContext) -> bool:
    """Trigger download on GoPro node."""
    click.echo("\nTriggering download...")
    for client in ctx.clients:
        if client.name == "go_pro_node":
            result = client.download()
            if not result.success:
                click.secho(f"Download trigger failed: {result.error}", fg="red")
                return False
    click.secho("Download triggered.", fg="cyan")
    return True


def do_discard(ctx: OrchestratorContext) -> bool:
    """Discard last recorded episode."""
    if not ctx.last_record:
        click.echo("\nNo episode to discard.")
        return False

    lr_run, lr_ep = ctx.last_record
    results = [client.discard(lr_run, lr_ep) for client in ctx.clients]

    failed = [(c.name, r.error) for c, r in zip(ctx.clients, results) if not r.success]
    if failed:
        click.secho(f"\nDiscard failed: {failed}", fg="red")
        return False

    click.secho(f"\nDISCARDED {lr_run} ep{format_episode(lr_ep)}", fg="red")
    ctx.last_record = None
    return True


def do_validate(ctx: OrchestratorContext) -> bool:
    """Validate last recorded episode (download first if needed)."""
    if not ctx.last_record:
        click.echo("\nNo episode to validate.")
        return False

    lr_run, lr_ep = ctx.last_record

    # First trigger download and wait for completion
    click.echo(f"\nDownloading videos before validation...")
    for client in ctx.clients:
        if client.name == "go_pro_node":
            client.download()

    # Wait for download to complete with timeout
    click.echo("Waiting for download to complete...")
    timeout = 120  # 2 minutes max
    start_time = time.time()
    while time.time() - start_time < timeout:
        time.sleep(1)
        info = ctx.nodes.get("go_pro_node", {})
        status = info.get("status", "")
        pending = info.get("pending_tasks", 0)
        is_downloading = info.get("is_downloading", False)

        # Done when not saving and no pending tasks
        if status != NodeStatus.SAVING.value and not is_downloading and pending == 0:
            break
    else:
        click.secho("Download timeout! Skipping validation.", fg="red")
        return False

    click.echo(f"Validating {lr_run} ep{format_episode(lr_ep)}...")

    # Restore terminal for validation output
    if ctx.old_settings:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, ctx.old_settings)

    try:
        ok = validate(lr_run, lr_ep)
        if ok:
            click.secho(f"VALIDATION PASSED for {lr_run} ep{format_episode(lr_ep)}", fg="green")
        else:
            click.secho(f"VALIDATION FAILED for {lr_run} ep{format_episode(lr_ep)}", fg="red")
        return ok
    finally:
        tty.setcbreak(sys.stdin.fileno())


def do_quit(ctx: OrchestratorContext) -> bool:
    """Handle quit with pending check."""
    pending = get_pending_count(ctx.nodes)

    if pending > 0:
        if ctx.quit_confirmed:
            # Already confirmed, quit anyway
            return True

        click.secho(f"\nWARNING: {pending} video(s) not yet saved!", fg="yellow", bold=True)
        click.echo("Press 'q' again to quit anyway, or 's' to save first.")
        ctx.quit_confirmed = True
        return False

    return True  # Safe to quit


# =============================================================================
# Key Handler
# =============================================================================

def handle_key(key: str, state: OrchestratorState, ctx: OrchestratorContext) -> Optional[str]:
    """
    Handle key press based on current state.
    Returns: "quit" to exit, "continue" to continue loop, None for no action.
    """
    allowed = ALLOWED_ACTIONS.get(state, {})
    action = allowed.get(key)

    if action is None:
        # Show appropriate message based on state
        if state == OrchestratorState.RECORDING:
            click.secho(f"\nCannot '{key}' while recording. Press Enter to stop first.", fg="red")
        elif state == OrchestratorState.SAVING:
            click.secho(f"\nNodes saving, please wait...", fg="yellow")
        elif state == OrchestratorState.RECOVERING:
            click.secho(f"\nNodes recovering, please wait...", fg="yellow")
        elif state in (OrchestratorState.ERROR, OrchestratorState.OFFLINE):
            click.secho(f"\nNodes in {state.value}. Only 'q' to quit is allowed.", fg="red")
        return "continue"

    # Dispatch to action function
    if action == "prepare":
        do_prepare(ctx)
    elif action == "start":
        do_start(ctx)
    elif action == "stop":
        do_stop(ctx)
    elif action == "download":
        do_download(ctx)
    elif action == "discard":
        do_discard(ctx)
    elif action == "validate":
        do_validate(ctx)
    elif action == "quit":
        if do_quit(ctx):
            return "quit"

    return "continue"


# =============================================================================
# Main
# =============================================================================

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

    zmq_ctx = zmq.Context()
    status_socket = zmq_ctx.socket(zmq.SUB)
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
                    "pending_tasks": msg.get("pending_tasks", 0),
                    "is_downloading": msg.get("is_downloading", False),
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
    click.echo("Press Enter to Start/Stop, s to save, x to discard last, v to validate last, q to quit.")

    # Wait for initial node heartbeats
    click.echo("\nWaiting for nodes...")
    time.sleep(2)

    # Check initial state and warn user
    initial_state = get_orchestrator_state(nodes, expected_names)
    if initial_state == OrchestratorState.RECORDING:
        click.secho("WARNING: Nodes still recording!", fg="yellow", bold=True)
        click.echo("Press Enter to stop, or 'q' to quit.")
    elif initial_state == OrchestratorState.SAVING:
        click.secho("WARNING: Nodes still saving!", fg="yellow", bold=True)
        click.echo("Wait for completion...")
    elif initial_state == OrchestratorState.ERROR:
        click.secho("WARNING: Nodes in ERROR state!", fg="red", bold=True)
    elif initial_state == OrchestratorState.OFFLINE:
        click.secho("WARNING: Some nodes are offline!", fg="red", bold=True)

    # Create context
    ctx = OrchestratorContext(
        clients=clients,
        nodes=nodes,
        expected_names=expected_names,
        run_id=active_run_id,
        delay=delay,
        next_episode=next_episode,
    )

    refresh_interval = 0.2

    # Save terminal settings and set cbreak mode
    old_settings = termios.tcgetattr(sys.stdin)
    ctx.old_settings = old_settings

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            try:
                # Get current state
                state = get_orchestrator_state(nodes, expected_names)

                # State change detection and sound alerts
                if state != ctx.prev_state:
                    if state == OrchestratorState.READY:
                        sound.play("ready")
                    if state in (OrchestratorState.ERROR, OrchestratorState.OFFLINE):
                        sound.start_alert()
                    if ctx.prev_state in (OrchestratorState.ERROR, OrchestratorState.OFFLINE):
                        if state not in (OrchestratorState.ERROR, OrchestratorState.OFFLINE):
                            sound.stop_alert()
                    ctx.prev_state = state

                # Auto-prepare when IDLE
                if state == OrchestratorState.IDLE and not ctx.prepare_sent:
                    do_prepare(ctx)
                    ctx.prepare_sent = True

                # Display status line
                status_line = format_status_line(nodes, expected_names)
                state_display = f"[{state.value.upper()}]"
                click.echo(f"\r{state_display} Nodes: {status_line}\033[K", nl=False)
                sys.stdout.flush()

                # Wait for key input
                key_ready, _, _ = select([sys.stdin], [], [], refresh_interval)
                if not key_ready:
                    continue

                # Reset quit confirmation on any key press
                ctx.quit_confirmed = False

                c = sys.stdin.read(1)
                key = "enter" if c in ("\r", "\n") else c

                result = handle_key(key, state, ctx)
                if result == "quit":
                    break

            except KeyboardInterrupt:
                # Handle Ctrl-C: check if safe to quit
                state = get_orchestrator_state(nodes, expected_names)

                if state == OrchestratorState.RECORDING:
                    click.secho("\nCannot quit while recording! Press Enter to stop first.", fg="red")
                    continue

                if state == OrchestratorState.SAVING:
                    click.secho("\nNodes saving, please wait...", fg="yellow")
                    continue

                pending = get_pending_count(nodes)
                if pending > 0:
                    if ctx.quit_confirmed:
                        click.secho(f"\nForce quit. {pending} video(s) not saved!", fg="red", bold=True)
                        break
                    else:
                        ctx.quit_confirmed = True
                        click.secho(f"\nWARNING: {pending} video(s) not yet saved!", fg="yellow", bold=True)
                        click.echo("Press Ctrl-C again to quit anyway, or 's' to save first.")
                        continue
                else:
                    break

    finally:
        # Restore terminal settings
        sound.stop_alert()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        running = False
        status_socket.close(0)
        zmq_ctx.term()
        click.echo("\nExiting...")


if __name__ == "__main__":
    main()
