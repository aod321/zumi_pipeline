import json
import os
import re
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from select import select
from typing import Optional, List

import click
import requests
import zmq

from zumi_config import HTTP_CONF, NodeStatus, STORAGE_CONF, ZMQ_CONF
from validator import validate, ValidationResult


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
        "x": "abort",
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

    def _get(self, path: str) -> Result:
        try:
            resp = requests.get(
                f"{self.base_url}{path}",
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

    def download(self, episode=None) -> Result:
        payload = {}
        if episode is not None:
            payload["episode"] = episode
        return self._post("/download", payload)

    def redownload(self, run_id, episode) -> Result:
        return self._post("/download", {"run_id": run_id, "episode": episode, "redownload": True})

    def get_pending_downloads(self) -> Result:
        return self._get("/pending-downloads")


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


def any_recording(nodes) -> bool:
    """Check if any node is still recording."""
    return any(info.get("status") == NodeStatus.RECORDING.value for info in nodes.values())


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
    completed_count: int = 0
    recollect_episodes: List[int] = field(default_factory=list)


# =============================================================================
# Recollect List Management
# =============================================================================

def load_recollect_list(run_id: str) -> List[int]:
    """Load recollect list from {DATA_DIR}/{run_id}/.recollect.json"""
    path = STORAGE_CONF.DATA_DIR / run_id / ".recollect.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f).get("episodes", [])
        except Exception:
            return []
    return []


def save_recollect_list(run_id: str, episodes: List[int]):
    """Save recollect list to file."""
    path = STORAGE_CONF.DATA_DIR / run_id / ".recollect.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"episodes": sorted(set(episodes))}, f)


def add_to_recollect(run_id: str, episode: int):
    """Add episode to recollect list."""
    episodes = load_recollect_list(run_id)
    if episode not in episodes:
        episodes.append(episode)
        save_recollect_list(run_id, episodes)


def remove_from_recollect(run_id: str, episode: int):
    """Remove episode from recollect list."""
    episodes = load_recollect_list(run_id)
    if episode in episodes:
        episodes.remove(episode)
        save_recollect_list(run_id, episodes)


# =============================================================================
# Validation History Management
# =============================================================================

def load_validation_history(run_id: str) -> dict:
    """Load validation history from {DATA_DIR}/{run_id}/.validated.json"""
    path = STORAGE_CONF.DATA_DIR / run_id / ".validated.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_validation_result(run_id: str, episode: int, result: ValidationResult):
    """Save validation result to history."""
    history = load_validation_history(run_id)
    history[str(episode)] = {
        "success": result.success,
        "error": result.error,
        "message": result.message
    }
    path = STORAGE_CONF.DATA_DIR / run_id / ".validated.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f)


# =============================================================================
# Helper Functions for Download/Validate Flow
# =============================================================================

def _get_gopro_client(ctx: OrchestratorContext) -> NodeClient:
    """Get the GoPro node client."""
    return next(c for c in ctx.clients if c.name == "go_pro_node")


def _wait_download_complete(ctx: OrchestratorContext, timeout: int = 300) -> bool:
    """Poll GoPro status until download complete."""
    start = time.time()

    # First, wait for download to start (is_downloading becomes True)
    while time.time() - start < 10:
        info = ctx.nodes.get("go_pro_node", {})
        if info.get("is_downloading", False):
            break
        time.sleep(0.5)

    # Then wait for download to complete (is_downloading becomes False)
    while time.time() - start < timeout:
        info = ctx.nodes.get("go_pro_node", {})
        if not info.get("is_downloading", False):
            return True
        time.sleep(1)
    return False


def _is_pending(gopro: NodeClient, run_id: str, episode: int) -> bool:
    """Check if episode is in pending download queue."""
    result = gopro.get_pending_downloads()
    if not result.success:
        return False
    pending = result.data.get("pending", [])
    return any(p["run_id"] == run_id and p["episode"] == episode for p in pending)


def _collect_all_episodes(run_id: str, gopro: NodeClient) -> set:
    """Collect episode numbers from directory + pending queue."""
    episodes = set()
    # From directory
    run_dir = STORAGE_CONF.DATA_DIR / run_id
    if run_dir.exists():
        for f in run_dir.glob(f"{run_id}_ep*_*.MP4"):
            match = re.search(r"_ep(\d+)_", f.name)
            if match:
                episodes.add(int(match.group(1)))
    # From pending
    result = gopro.get_pending_downloads()
    if result.success:
        pending = result.data.get("pending", [])
        for item in pending:
            if item["run_id"] == run_id:
                episodes.add(item["episode"])
    return episodes


def _discard_episode(ctx: OrchestratorContext, run_id: str, episode: int):
    """Discard an episode's data."""
    for client in ctx.clients:
        client.discard(run_id, episode)
    click.secho(f"Discarded ep{episode:03d}", fg="red")


def get_next_episode_to_record(run_id: str) -> int:
    """Get next episode, prioritizing recollect list."""
    recollect = load_recollect_list(run_id)
    if recollect:
        return recollect[0]
    return infer_next_episode(run_id)


# =============================================================================
# Action Functions
# =============================================================================

def do_prepare(ctx: OrchestratorContext) -> bool:
    """Send prepare to all nodes."""
    ctx.current_episode = get_next_episode_to_record(ctx.run_id)
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
        ctx.current_episode = get_next_episode_to_record(ctx.run_id)

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

    ctx.completed_count += 1
    sound.play("stop")
    click.secho(f"\nSTOPPED {ctx.run_id} ep{format_episode(ctx.current_episode)}", fg="red")

    # Remove from recollect list if this was a recollect episode
    remove_from_recollect(ctx.run_id, ctx.current_episode)

    return True


def do_abort(ctx: OrchestratorContext) -> bool:
    """Abort recording and discard current episode data."""
    run_id = ctx.run_id
    episode = ctx.current_episode

    click.secho(f"\nAborting {run_id} ep{format_episode(episode)}...", fg="yellow")

    # 1. Stop recording - only target recording nodes to reduce 409 errors
    stop_ts = time.time() + ctx.delay
    recording_clients = [c for c in ctx.clients
                         if ctx.nodes.get(c.name, {}).get("status") == NodeStatus.RECORDING.value]
    targets = recording_clients or ctx.clients  # fallback to all if none found
    results = [client.stop(stop_time=stop_ts) for client in targets]

    failed = [(c.name, r.error) for c, r in zip(targets, results) if not r.success]
    if failed:
        click.secho(f"Abort stop failed (some devices): {failed}", fg="yellow")
        # Don't return - continue to discard data

    # 2. Wait for online nodes to finish SAVING
    click.echo("Waiting for nodes to finish saving...")
    timeout = 30
    start = time.time()
    while time.time() - start < timeout:
        state = get_orchestrator_state(ctx.nodes, ctx.expected_names)
        # Accept IDLE, READY, or stay in OFFLINE/ERROR (device may be down)
        if state in (OrchestratorState.IDLE, OrchestratorState.READY,
                     OrchestratorState.OFFLINE, OrchestratorState.ERROR):
            break
        time.sleep(0.5)

    # 3. Discard current episode data (always execute)
    for client in ctx.clients:
        client.discard(run_id, episode)

    # 4. Clear last_record to prevent duplicate discard
    ctx.last_record = None
    ctx.prepare_sent = False

    sound.play("error")
    click.secho(f"\nABORTED {run_id} ep{format_episode(episode)} - data discarded", fg="red", bold=True)

    return True


def _read_single_key(ctx: OrchestratorContext) -> str:
    """Read a single key from stdin (blocking)."""
    c = sys.stdin.read(1)
    return "enter" if c in ("\r", "\n") else c


def _prompt_failure_action(result: ValidationResult) -> str:
    """Show failure options and read user choice."""
    if result.error == "video_missing":
        click.echo("[d]iscard / [r]ecollect / re-do[w]nload / [s]kip?")
    else:
        click.echo("[d]iscard / [r]ecollect / [s]kip?")


def _handle_failure_choice(ctx: OrchestratorContext, run_id: str, episode: int, choice: str):
    """Handle user's choice for a failed validation."""
    gopro = _get_gopro_client(ctx)

    if choice == "d":
        _discard_episode(ctx, run_id, episode)
    elif choice == "r":
        add_to_recollect(run_id, episode)
        click.secho(f"Added ep{episode:03d} to recollect list", fg="yellow")
    elif choice == "w":
        # Re-download
        gopro.redownload(run_id, episode)
        click.echo("Re-downloading...")
        _wait_download_complete(ctx)


def do_download_and_validate(ctx: OrchestratorContext) -> bool:
    """'s' key: process pending downloads with validation."""
    gopro = _get_gopro_client(ctx)

    click.echo("\nProcessing pending downloads...")

    while True:
        # Get pending list
        pending_result = gopro.get_pending_downloads()
        if not pending_result.success:
            click.secho(f"Failed to get pending list: {pending_result.error}", fg="red")
            return False

        pending = pending_result.data.get("pending", [])
        if not pending:
            click.secho("All pending downloads processed.", fg="green")
            return True

        item = pending[0]
        run_id, episode = item["run_id"], item["episode"]
        click.echo(f"\nDownloading ep{episode:03d}...")

        # Download this episode
        gopro.download(episode=episode)
        if not _wait_download_complete(ctx):
            click.secho("Download timeout!", fg="red")
            continue

        # Validate
        click.echo(f"Validating ep{episode:03d}...")
        result = validate(run_id, episode)

        if result.success:
            save_validation_result(run_id, episode, result)
            click.secho(f"[PASS] ep{episode:03d}", fg="green")
            continue

        # Validation failed
        click.secho(f"[FAIL] ep{episode:03d}: {result.message}", fg="red")
        _prompt_failure_action(result)

        choice = _read_single_key(ctx)

        if choice == "s":
            # Skip - save failed result
            save_validation_result(run_id, episode, result)
            continue

        _handle_failure_choice(ctx, run_id, episode, choice)

        # After handling, if redownload, validate again
        if choice == "w":
            result = validate(run_id, episode)
            if result.success:
                save_validation_result(run_id, episode, result)
                click.secho(f"[PASS] ep{episode:03d}", fg="green")
            else:
                save_validation_result(run_id, episode, result)
                click.secho(f"[FAIL] ep{episode:03d}: {result.message}", fg="red")
        else:
            save_validation_result(run_id, episode, result)


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


def _validate_single_episode(ctx: OrchestratorContext, gopro: NodeClient, episode: int, record: dict) -> Optional[str]:
    """Process a single episode in validate-all flow. Returns 'quit' to exit, None to continue."""
    # Show status and options based on history
    if record and record.get("success"):
        click.echo(f"ep{episode:03d}: validated. [r]e-validate / [s]kip / [q]uit?")
    elif record and not record.get("success"):
        click.secho(f"ep{episode:03d}: failed ({record.get('error')}). [d]elete / [r]e-validate / [s]kip / [q]uit?", fg="red")
    elif _is_pending(gopro, ctx.run_id, episode):
        click.echo(f"ep{episode:03d}: pending. [d]ownload+validate / [s]kip / [q]uit?")
    else:
        click.echo(f"ep{episode:03d}: not validated. [v]alidate / [s]kip / [q]uit?")

    action = _read_single_key(ctx)

    if action == "s":
        return None
    if action == "q":
        return "quit"
    if action == "d" and record and not record.get("success"):
        # Delete failed episode
        _discard_episode(ctx, ctx.run_id, episode)
        return None

    # Download if pending
    if _is_pending(gopro, ctx.run_id, episode):
        if action not in ("d", "v", "r"):
            return None
        click.echo(f"Downloading ep{episode:03d}...")
        gopro.download(episode=episode)
        if not _wait_download_complete(ctx):
            click.secho("Download timeout!", fg="red")
            return None

    # Validate
    if action in ("v", "r", "d"):
        click.echo(f"Validating ep{episode:03d}...")
        result = validate(ctx.run_id, episode)

        if result.success:
            save_validation_result(ctx.run_id, episode, result)
            click.secho(f"[PASS] ep{episode:03d}", fg="green")
        else:
            click.secho(f"[FAIL] ep{episode:03d}: {result.message}", fg="red")
            _prompt_failure_action(result)

            choice = _read_single_key(ctx)

            if choice == "s":
                save_validation_result(ctx.run_id, episode, result)
                return None

            _handle_failure_choice(ctx, ctx.run_id, episode, choice)

            # After handling, if redownload, validate again
            if choice == "w":
                result = validate(ctx.run_id, episode)
                if result.success:
                    save_validation_result(ctx.run_id, episode, result)
                    click.secho(f"[PASS] ep{episode:03d}", fg="green")
                else:
                    save_validation_result(ctx.run_id, episode, result)
                    click.secho(f"[FAIL] ep{episode:03d}: {result.message}", fg="red")
            else:
                save_validation_result(ctx.run_id, episode, result)

    return None


def do_validate_all(ctx: OrchestratorContext) -> bool:
    """'v' key: iterate all episodes, unvalidated first, then validated."""
    gopro = _get_gopro_client(ctx)

    history = load_validation_history(ctx.run_id)
    episodes = _collect_all_episodes(ctx.run_id, gopro)

    if not episodes:
        click.echo("\nNo episodes found.")
        return True

    # Split into unvalidated and validated
    unvalidated = [ep for ep in sorted(episodes) if str(ep) not in history]
    validated = [ep for ep in sorted(episodes) if str(ep) in history]

    click.echo(f"\nValidating run {ctx.run_id} ({len(episodes)} episodes)...")

    # Process unvalidated first
    if unvalidated:
        click.echo(f"\n--- Unvalidated ({len(unvalidated)}) ---")
        for episode in unvalidated:
            result = _validate_single_episode(ctx, gopro, episode, None)
            if result == "quit":
                return True

    # Then process validated from beginning
    if validated:
        click.echo(f"\n--- Validated ({len(validated)}) ---")
        for episode in validated:
            record = history.get(str(episode))
            result = _validate_single_episode(ctx, gopro, episode, record)
            if result == "quit":
                return True

    click.secho("Validation complete.", fg="green")
    return True


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
    # Allow stop/abort if fault but still recording
    fault_with_rec = state in (OrchestratorState.ERROR, OrchestratorState.OFFLINE) and any_recording(ctx.nodes)
    if fault_with_rec:
        allowed = {"enter": "stop", "x": "abort", "q": "quit"}
    else:
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
    elif action == "abort":
        do_abort(ctx)
    elif action == "download":
        do_download_and_validate(ctx)
    elif action == "discard":
        do_discard(ctx)
    elif action == "validate":
        do_validate_all(ctx)
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
    next_episode = get_next_episode_to_record(active_run_id)
    recollect = load_recollect_list(active_run_id)

    click.clear()
    click.secho("=== Zumi Orchestrator 2.0 (HTTP) ===", fg="cyan", bold=True)
    click.echo(f"[+] Data Dir: {STORAGE_CONF.DATA_DIR}")
    click.echo(f"[+] Run ID: {active_run_id} | Next episode: ep{format_episode(next_episode)}")
    if recollect:
        click.secho(f"[+] Recollect episodes: {recollect}", fg="yellow")
    click.echo(f"[+] Validation mode: {validation_mode}")

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

                # Force abort if fault while recording (not depending on prev_state)
                fault_with_recording = state in (OrchestratorState.ERROR, OrchestratorState.OFFLINE) and any_recording(nodes)
                if fault_with_recording and not getattr(ctx, "emergency_abort_sent", False):
                    click.secho("\nFault while recording -> forcing abort...", fg="red", bold=True)
                    do_abort(ctx)
                    ctx.emergency_abort_sent = True
                if not fault_with_recording:
                    ctx.emergency_abort_sent = False

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

                # Display dual-line status
                pending = get_pending_count(nodes)
                ep_num = ctx.current_episode if ctx.current_episode else ctx.next_episode
                ep_info = f"ep{format_episode(ep_num)} done:{ctx.completed_count} pend:{pending}"

                if fault_with_recording:
                    help_line = "[Enter]Stop [x]Abort [q]Quit - FAULT while recording!"
                else:
                    help_line = "[Enter]Start/Stop [s]Save [x]Discard [v]Validate [q]Quit"
                status_line = format_status_line(nodes, expected_names)
                state_str = f"[{state.value.upper()}]"

                line1 = help_line
                line2 = f"{state_str} {ep_info} | {status_line}"
                sys.stdout.write(f"\r\033[K{line1}\n\r\033[K{line2}\033[F")
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
