import logging
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import zmq

from zumi_config import Commands, NET_CONF, NodeStatus
from zumi_util import precise_wait

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')


class ZMQService(ABC):
    def __init__(self, name: str):
        self.name = name
        self.context = zmq.Context()
        self.status = NodeStatus.INIT
        self.run_id = None
        self.episode = None

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{NET_CONF.ORCHESTRATOR_IP}:{NET_CONF.CMD_PORT}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{NET_CONF.ORCHESTRATOR_IP}:{NET_CONF.STATUS_PORT}")

        self.is_running = True
        self.is_recording = False
        self.scheduled_stop_time = None

        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.logger = logging.getLogger(self.name)

    def start(self):
        self.logger.info("Initializing...")
        try:
            self.on_init()
            self.status = NodeStatus.IDLE
            self.logger.info("Ready.")
        except Exception as exc:
            self.status = NodeStatus.ERROR
            self.logger.error(f"Init failed: {exc}")
            sys.exit(1)

        self.control_thread.start()
        self.heartbeat_thread.start()

        def sigint_handler(signum, frame):
            self.logger.info("Keyboard interrupt, stopping...")
            self.is_running = False

        signal.signal(signal.SIGINT, sigint_handler)

        self.main_loop()
        self.shutdown()

    def shutdown(self):
        self.is_running = False
        try:
            self.on_shutdown()
        finally:
            self.context.term()
            self.logger.info("Shutdown complete.")

    def _heartbeat_loop(self):
        while self.is_running:
            try:
                msg = {
                    "node": self.name,
                    "status": self.status.value,
                    "ts": time.time(),
                }
                self.pub_socket.send_json(msg)
                time.sleep(1.0)
            except Exception:
                pass

    def _control_loop(self):
        while self.is_running:
            try:
                if not self.sub_socket.poll(100):
                    continue

                msg = self.sub_socket.recv_json()
                cmd = msg.get("cmd")
                payload = msg.get("payload", {})

                if cmd == Commands.PREPARE:
                    self.run_id = payload.get("run_id")
                    self.episode = payload.get("episode")
                    self.logger.info(f"Prepare: {self.run_id}, episode={self.episode}")
                    self.status = (
                        NodeStatus.READY if self.on_prepare(self.run_id, self.episode) else NodeStatus.ERROR
                    )

                elif cmd == Commands.START_SYNC:
                    target_ts = payload.get("start_time")
                    self.run_id = payload.get("run_id")
                    self.episode = payload.get("episode")
                    if self.status not in (NodeStatus.READY, NodeStatus.IDLE) or self.is_recording:
                        self.logger.warning(
                            f"START ignored; current status={self.status} recording={self.is_recording}"
                        )
                        continue
                    if target_ts is not None:
                        precise_wait(target_ts, time_func=time.time)

                    self.scheduled_stop_time = None
                    self.on_start_recording(self.run_id, self.episode)
                    self.is_recording = True
                    self.status = NodeStatus.RECORDING

                elif cmd == Commands.STOP_SYNC:
                    stop_ts = payload.get("stop_time")
                    self.logger.info(f"Stop scheduled in {(stop_ts - time.time())*1000:.1f}ms")
                    self.scheduled_stop_time = stop_ts
                    self.on_schedule_stop(stop_ts)

                elif cmd == Commands.START_DOWNLOAD:
                    self.on_start_download(payload)

                elif cmd == Commands.DISCARD_RUN:
                    run_id = payload.get("run_id")
                    episode = payload.get("episode")
                    self.logger.info(f"Discard run: {run_id}, episode={episode}")
                    self.on_discard_run(run_id, episode)

                elif cmd == Commands.EXIT:
                    self.is_running = False
                    break
            except zmq.ZMQError:
                continue
            except Exception as exc:
                self.logger.error(f"Control loop error: {exc}")

    def get_iso_timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    @abstractmethod
    def on_init(self):
        ...

    @abstractmethod
    def main_loop(self):
        ...

    def on_prepare(self, run_id, episode=None):
        return True

    @abstractmethod
    def on_start_recording(self, run_id, episode=None):
        ...

    @abstractmethod
    def on_stop_recording(self):
        ...

    @abstractmethod
    def on_shutdown(self):
        ...

    def on_schedule_stop(self, stop_ts):
        pass

    def on_discard_run(self, run_id, episode=None):
        pass

    def on_start_download(self, payload=None):
        pass
