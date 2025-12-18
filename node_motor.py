import json
import multiprocessing
import os
import time
from multiprocessing import Queue

import numpy as np
from dotenv import load_dotenv

from motor_interface import MotorState
from motor_mock import MockMotorDriver
from zumi_config import NodeStatus, STORAGE_CONF
from zumi_core import ZMQService
from zumi_util import RateLimiter


def writer_process(queue: Queue, data_dir: str):
    """
    Dedicated process for disk I/O.
    """
    os.makedirs(data_dir, exist_ok=True)
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

            elif msg_type == "STOP":
                if not buffer:
                    print("[Writer] Warning: No data to save.")
                    current_run_id = None
                    meta = {}
                    continue

                ep_tag = f"ep{current_episode:03d}" if current_episode is not None else "ep001"
                filename = os.path.join(data_dir, f"{current_run_id}_{ep_tag}_motor.npz")
                data_array = np.array(buffer, dtype=np.float64)
                col_names = np.array(["ts", "pos", "vel", "tau", "iter"])
                print(f"[Writer] Saving {len(buffer)} frames to {filename}...")
                np.savez_compressed(filename, data=data_array, columns=col_names)

                meta["total_samples"] = len(buffer)
                meta["duration"] = buffer[-1][0] - buffer[0][0] if buffer else 0
                meta["episode"] = current_episode
                meta_path = os.path.join(data_dir, f"{current_run_id}_{ep_tag}_motor_meta.json")
                with open(meta_path, "w") as fh:
                    json.dump(meta, fh, indent=2)

                print("[Writer] Save complete.")
                buffer = []
                current_run_id = None
                current_episode = None
                meta = {}
        except Exception as exc:
            print(f"[Writer] Error: {exc}")


class MotorNode(ZMQService):
    def on_init(self):
        self.target_freq = 200.0
        load_dotenv()

        # Init driver first - if this fails, no need to start writer
        self.driver = self._create_driver()

        self.write_queue = Queue()
        self.writer = multiprocessing.Process(
            target=writer_process, args=(self.write_queue, str(STORAGE_CONF.DATA_DIR))
        )
        self.writer.start()
        self.iter_idx = 0

    def _create_driver(self):
        driver_sel = os.getenv("MOTOR_DRIVER", "dm").lower()
        slave_id = int(os.getenv("MOTOR_SLAVE_ID", "0x07"), 16)
        master_id = int(os.getenv("MOTOR_MASTER_ID", "0x17"), 16)
        serial_port = os.getenv("MOTOR_SERIAL_PORT", "/dev/ttyACM0")
        if "MOTOR_PORT" in os.environ:
            serial_port = os.environ["MOTOR_PORT"]

        if driver_sel == "mock":
            self.logger.info("Using MockMotorDriver (MOTOR_DRIVER=mock).")
            driver = MockMotorDriver()
            driver.enable()
            return driver

        from motor_dm import DMMotorDriver

        driver = DMMotorDriver(serial_port, slave_id, master_id, logger=self.logger)
        self.logger.info(
            f"DM driver ready on {serial_port}, SlaveID={hex(slave_id)}, MasterID={hex(master_id)}"
        )
        return driver

    def on_prepare(self, run_id, episode=None):
        try:
            self.driver.command(0, 0, 0, 0, 0)
            _ = self.driver.get_state()
            return True
        except Exception as exc:
            self.logger.error(f"Prepare failed: {exc}")
            return False

    def on_start_recording(self, run_id, episode=None):
        self.current_episode = episode
        meta = {
            "run_id": run_id,
            "episode": episode,
            "driver": self.driver.__class__.__name__,
            "target_freq": self.target_freq,
            "start_time_iso": self.get_iso_timestamp(),
        }
        self.write_queue.put(("START", {"run_id": run_id, "episode": episode, "meta": meta}))
        self.iter_idx = 0
        ep_tag = f"ep{int(episode):03d}" if episode is not None else "ep001"
        self.logger.info(f"Recording started: {run_id} {ep_tag}")

    def on_stop_recording(self):
        self.logger.info("Recording stopped.")

    def main_loop(self):
        rate = RateLimiter(self.target_freq)
        local_buffer = []
        batch_size = 50

        get_time = time.time

        while self.is_running:
            try:
                self.driver.command(0.0, 0.0, 0.0, 0.0, 0.0)
            except Exception:
                pass

            if self.is_recording and self.scheduled_stop_time:
                if get_time() >= self.scheduled_stop_time:
                    self.is_recording = False
                    self.status = NodeStatus.SAVING
                    if local_buffer:
                        self.write_queue.put(("DATA", local_buffer))
                        local_buffer = []
                    self.write_queue.put(("STOP", None))
                    self.scheduled_stop_time = None
                    self.status = NodeStatus.IDLE
                    self.on_stop_recording()

            if self.is_recording:
                try:
                    state: MotorState = self.driver.get_state()
                except Exception as exc:
                    self.logger.error(f"State read failed: {exc}")
                    state = MotorState(0.0, 0.0, 0.0)

                frame = (get_time(), float(state.position), float(state.velocity), float(state.torque), int(self.iter_idx))
                local_buffer.append(frame)
                self.iter_idx += 1

                if len(local_buffer) >= batch_size:
                    self.write_queue.put(("DATA", local_buffer))
                    local_buffer = []

            rate.sleep()

        if self.is_recording:
            self.status = NodeStatus.SAVING
            if local_buffer:
                self.write_queue.put(("DATA", local_buffer))
            self.write_queue.put(("STOP", None))
            self.status = NodeStatus.IDLE
            self.on_stop_recording()

    def on_discard_run(self, run_id, episode=None):
        try:
            targets = []
            if episode is not None:
                ep_tag = f"ep{int(episode):03d}"
                targets.extend(STORAGE_CONF.DATA_DIR.glob(f"{run_id}_{ep_tag}_*motor.npz"))
                targets.extend(STORAGE_CONF.DATA_DIR.glob(f"{run_id}_{ep_tag}_*motor_meta.json"))
            else:
                targets.extend(STORAGE_CONF.DATA_DIR.glob(f"{run_id}_*motor.npz"))
                targets.extend(STORAGE_CONF.DATA_DIR.glob(f"{run_id}_*motor_meta.json"))

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

        self.write_queue.put(None)
        self.writer.join(timeout=5)


if __name__ == "__main__":
    node = MotorNode(name="DM3510")
    node.start()
