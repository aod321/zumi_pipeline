import json
import multiprocessing
import os
import time
from multiprocessing import Queue
from pathlib import Path

import numpy as np

from motor_interface import MotorState
from motor_mock import MockMotorDriver
from zumi_config import HTTP_CONF, MOTOR_CONF, NodeStatus, STORAGE_CONF
from zumi_core import NodeHTTPService
from zumi_util import RateLimiter


def writer_process(queue: Queue, data_dir: str):
    """
    Dedicated process for disk I/O.
    """
    data_dir = Path(data_dir)
    current_run_id = None
    current_episode = None
    buffer = []
    meta = {}

    while True:
        item = queue.get()
        if item is None:
            break

        try:
            msg_type, payload = item
        except Exception:
            continue

        try:
            if msg_type == "START":
                current_run_id = payload["run_id"]
                current_episode = payload.get("episode", 1)
                meta = payload["meta"]
                buffer = []
                print(f"[Writer] Start session {current_run_id} ep{current_episode:03d}")

            elif msg_type == "DATA":
                buffer.extend(payload)

            elif msg_type == "DISCARD":
                # Discard current buffer without saving
                print(f"[Writer] DISCARDING data for {current_run_id} ep{current_episode}")
                buffer = []
                current_run_id = None
                current_episode = None
                meta = {}

            elif msg_type == "STOP":
                if not buffer:
                    print("[Writer] Warning: No data to save.")
                    current_run_id = None
                    meta = {}
                    continue

                # Create run_id directory
                run_dir = data_dir / current_run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                ep_tag = f"ep{current_episode:03d}" if current_episode is not None else "ep001"
                filename = run_dir / f"{current_run_id}_{ep_tag}_motor.npz"
                data_array = np.array(buffer, dtype=np.float64)
                col_names = np.array(["ts", "pos", "vel", "tau", "iter"])
                print(f"[Writer] Saving {len(buffer)} frames to {filename}...")
                np.savez_compressed(filename, data=data_array, columns=col_names)

                meta["total_samples"] = len(buffer)
                meta["duration"] = buffer[-1][0] - buffer[0][0] if buffer else 0
                meta["episode"] = current_episode
                meta_path = run_dir / f"{current_run_id}_{ep_tag}_motor_meta.json"
                with open(meta_path, "w") as fh:
                    json.dump(meta, fh, indent=2)

                print("[Writer] Save complete.")
                buffer = []
                current_run_id = None
                current_episode = None
                meta = {}
        except Exception as exc:
            print(f"[Writer] Error: {exc}")


class MotorNode(NodeHTTPService):
    # Motor uses smaller backoff due to high frequency communication
    RECOVERY_BACKOFF_BASE = 1.0
    RECOVERY_BACKOFF_MAX = 10.0

    def __init__(self, name="DM3510"):
        self.target_freq = 200.0
        self.local_buffer = []
        self.batch_size = 50
        self.iter_idx = 0
        super().__init__(name=name, host=HTTP_CONF.MOTOR_HOST, port=HTTP_CONF.MOTOR_PORT)

    def on_init(self):
        # Init driver first - if this fails, no need to start writer
        self.driver = self._create_driver()

        self.write_queue = Queue()
        self.writer = multiprocessing.Process(
            target=writer_process, args=(self.write_queue, str(STORAGE_CONF.DATA_DIR))
        )
        self.writer.start()
        self.iter_idx = 0
        self.local_buffer = []

    def _create_driver(self, auto_set_zero: bool = True):
        driver_sel = MOTOR_CONF.DRIVER.lower()
        slave_id = MOTOR_CONF.SLAVE_ID
        master_id = MOTOR_CONF.MASTER_ID
        serial_port = MOTOR_CONF.SERIAL_PORT

        if driver_sel == "mock":
            self.logger.info("Using MockMotorDriver (MOTOR_DRIVER=mock).")
            driver = MockMotorDriver()
            driver.enable()
            return driver

        from motor_dm import DMMotorDriver

        driver = DMMotorDriver(serial_port, slave_id, master_id, logger=self.logger,
                               auto_set_zero=auto_set_zero)
        self.logger.info(
            f"DM driver ready on {serial_port}, SlaveID={hex(slave_id)}, MasterID={hex(master_id)}"
        )
        return driver

    def on_prepare(self, run_id, episode=None):
        try:
            self.driver.command(0, 0, 0, 0, 0)
            state = self.driver.get_state()

            if state.position >= 0.1:
                self.logger.error(
                    f"Gripper position ({state.position:.3f}) >= 0.1, "
                    "please close the gripper before starting recording!"
                )
                return False

            self._initialize_gripper_position()
            return True
        except Exception as exc:
            self.logger.error(f"Prepare failed: {exc}")
            return False

    def _initialize_gripper_position(self):
        """Initialize gripper: set zero position."""
        self.driver.set_zero()
        self.logger.info("Gripper zero position set")

    def on_start_recording(self, run_id, episode=None):
        self.current_episode = episode
        self.recording_start_time = time.time()
        meta = {
            "run_id": run_id,
            "episode": episode,
            "driver": self.driver.__class__.__name__,
            "target_freq": self.target_freq,
            "start_time_iso": self.get_iso_timestamp(),
        }
        self.write_queue.put(("START", {"run_id": run_id, "episode": episode, "meta": meta}))
        self.iter_idx = 0
        self.local_buffer = []
        ep_tag = f"ep{int(episode):03d}" if episode is not None else "ep001"
        self.logger.info(f"Recording started: {run_id} {ep_tag}")

    def on_stop_recording(self):
        self.is_recording = False
        self.status = NodeStatus.SAVING
        if self.local_buffer:
            self.write_queue.put(("DATA", self.local_buffer))
            self.local_buffer = []
        self.write_queue.put(("STOP", None))
        self.logger.info("Recording stopped.")
        self.status = NodeStatus.IDLE

    def check_hardware_health(self):
        """Check motor communication by sending a zero command."""
        self.driver.command(0.0, 0.0, 0.0, 0.0, 0.0)

    def main_loop(self):
        rate = RateLimiter(self.target_freq)
        get_time = time.time
        consecutive_failures = 0
        max_failures = 10  # ~50ms at 200Hz
        lock_duration = 0.5

        while self.is_running:
            should_lock = not self.is_recording or (
                get_time() - getattr(self, 'recording_start_time', 0) < lock_duration
            )

            try:
                if should_lock:
                    self.driver.command(0.0, 0.0, 0.0, 0.8, 0.05)
                else:
                    self.driver.command(0.0, 0.0, 0.0, 0.0, 0.0)
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    if self.is_recording:
                        self._discard_current_recording(f"Motor communication lost: {exc}")
                    raise RuntimeError(f"Motor communication lost after {consecutive_failures} consecutive failures: {exc}") from exc

            if self.is_recording:
                try:
                    state: MotorState = self.driver.get_state()
                except Exception as exc:
                    self.logger.error(f"State read failed: {exc}")
                    state = MotorState(0.0, 0.0, 0.0)

                frame = (get_time(), float(state.position), float(state.velocity), float(state.torque), int(self.iter_idx))
                self.local_buffer.append(frame)
                self.iter_idx += 1

                if len(self.local_buffer) >= self.batch_size:
                    self.write_queue.put(("DATA", self.local_buffer))
                    self.local_buffer = []

            rate.sleep()

        if self.is_recording:
            self.on_stop_recording()
        elif self.local_buffer:
            self.write_queue.put(("DATA", self.local_buffer))
            self.local_buffer = []

    def _discard_current_recording(self, reason: str):
        """Discard current recording data due to error and notify user loudly."""
        run_id = self.run_id
        episode = self.episode
        self.logger.error("=" * 60)
        self.logger.error("!!! RECORDING ABORTED - DATA DISCARDED !!!")
        self.logger.error(f"Run: {run_id}, Episode: {episode}")
        self.logger.error(f"Reason: {reason}")
        self.logger.error("=" * 60)

        # Clear buffers without saving
        self.local_buffer = []
        self.write_queue.put(("DISCARD", None))  # Tell writer to discard
        self.is_recording = False

        # Delete any already-written files for this episode
        if run_id and episode is not None:
            self.on_discard_run(run_id, episode)

    def on_discard_run(self, run_id, episode=None):
        # 1. Stop recording if active (discard takes over stop's role)
        if self.is_recording:
            self.is_recording = False
            self.local_buffer = []

        # 2. Tell writer to discard buffer (not save)
        self.write_queue.put(("DISCARD", None))

        # 3. Delete any existing files
        try:
            run_dir = STORAGE_CONF.DATA_DIR / run_id
            targets = []
            if episode is not None:
                ep_tag = f"ep{int(episode):03d}"
                targets.extend(run_dir.glob(f"{run_id}_{ep_tag}_*motor.npz"))
                targets.extend(run_dir.glob(f"{run_id}_{ep_tag}_*motor_meta.json"))
            else:
                targets.extend(run_dir.glob(f"{run_id}_*motor.npz"))
                targets.extend(run_dir.glob(f"{run_id}_*motor_meta.json"))

            for path in targets:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    continue
            self.logger.info(f"Discarded motor data for {run_id}, episode={episode}")
        except Exception as exc:
            self.logger.error(f"Discard failed: {exc}")

    def on_shutdown(self):
        try:
            if hasattr(self, "driver"):
                self.driver.shutdown()
        except Exception as exc:
            self.logger.error(f"Shutdown error: {exc}")

        if hasattr(self, "write_queue"):
            self.write_queue.put(None)
        if hasattr(self, "writer"):
            self.writer.join(timeout=5)

    # Recovery methods -------------------------------------------------------
    def can_recover(self, exc: Exception) -> bool:
        """Only attempt recovery for communication-related errors."""
        return isinstance(exc, (TimeoutError, RuntimeError, OSError))

    def _cleanup_for_recovery(self):
        """Shutdown old driver, but keep writer process running."""
        try:
            if hasattr(self, "driver"):
                self.driver.shutdown()
        except Exception:
            pass

    def on_recover(self):
        """Recreate motor driver without resetting zero position."""
        self.logger.info("Recreating motor driver...")
        self.driver = self._create_driver(auto_set_zero=False)

        state = self.driver.get_state()
        if state.position >= 0.1:
            raise RuntimeError(
                f"Gripper position ({state.position:.3f}) >= 0.1 during recovery. "
                "Please close the gripper manually and retry!"
            )

        self.driver.command(0.0, 0.0, 0.0, 0.8, 0.05)
        self.logger.info("Motor driver recovered")

    def after_recover(self):
        """Reset timing variables and buffers."""
        self.iter_idx = 0
        self.local_buffer = []
        self.is_recording = False
        self.run_id = None
        self.episode = None


if __name__ == "__main__":
    node = MotorNode(name="DM3510")
    node.start()
