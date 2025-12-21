import logging
import signal
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict

import uvicorn
from fastapi import Body, FastAPI, HTTPException
import zmq

from zumi_config import NodeStatus, ZMQ_CONF
from zumi_util import precise_wait

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


class NodeHTTPService(ABC):
    """
    Base class for node FastAPI services with a minimal state machine.
    """

    # Health check configuration
    HEALTH_CHECK_INTERVAL = 5  # seconds between health checks
    HEALTH_CHECK_MAX_FAILURES = 3  # consecutive failures before ERROR

    # Recovery configuration (can be overridden by subclass)
    AUTO_RECOVERY_ENABLED = True  # whether to attempt auto recovery
    MAX_RECOVERY_ATTEMPTS = 5  # max recovery attempts before giving up
    RECOVERY_BACKOFF_BASE = 2.0  # base backoff time in seconds
    RECOVERY_BACKOFF_MAX = 60.0  # max backoff time in seconds

    def __init__(self, name: str, host: str = "0.0.0.0", port: int = 8000):
        self.name = name
        self.host = host
        self.port = port

        self.status = NodeStatus.INIT
        self.is_running = False
        self.is_recording = False
        self.run_id = None
        self.episode = None
        self.last_error = None
        self._health_check_failures = 0
        self._recovery_attempts = 0
        self._in_recovery = False

        self.app = FastAPI(title=f"{name} service")
        self.logger = logging.getLogger(self.name)

        self._setup_routes()
        self._setup_lifecycle()

        # ZMQ heartbeat publisher
        self.zmq_context = zmq.Context()
        self.status_pub = self.zmq_context.socket(zmq.PUB)
        self.status_pub.connect(f"tcp://{ZMQ_CONF.ORCHESTRATOR_IP}:{ZMQ_CONF.STATUS_PORT}")
        self._hb_thread = None
        self._shutdown_triggered = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM by publishing OFFLINE before uvicorn shuts down."""
        if self._shutdown_triggered:
            return  # Avoid duplicate handling
        self._shutdown_triggered = True
        self.logger.info(f"Signal {signum} received, publishing OFFLINE...")
        self.publish_status(force_status=NodeStatus.OFFLINE)
        time.sleep(0.05)  # Give ZMQ time to flush
        # Re-raise to let uvicorn handle normal shutdown
        raise KeyboardInterrupt

    # FastAPI bootstrap --------------------------------------------------
    def _setup_lifecycle(self):
        @self.app.on_event("startup")
        async def _startup():
            self.logger.info("Initializing...")
            try:
                self.on_init()
                self.status = NodeStatus.IDLE
                self.is_running = True
            except Exception as exc:
                self.status = NodeStatus.ERROR
                self.last_error = str(exc)
                self.logger.error(f"Init failed: {exc}")
                self.publish_status()  # Notify orchestrator of ERROR immediately
                raise

            self.loop_thread = threading.Thread(target=self._main_loop_wrapper, daemon=True)
            self.loop_thread.start()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()

        @self.app.on_event("shutdown")
        async def _shutdown():
            self.logger.info("Shutdown signal received.")
            # Notify orchestrator immediately before stopping
            self.publish_status(force_status=NodeStatus.OFFLINE)
            time.sleep(0.05)  # Give ZMQ time to flush
            self.is_running = False
            try:
                self.on_shutdown()
            except Exception as exc:
                self.logger.error(f"Shutdown error: {exc}")
            if hasattr(self, "loop_thread"):
                self.loop_thread.join(timeout=3)
            if self._hb_thread:
                self._hb_thread.join(timeout=2)
            try:
                self.status_pub.close(0)
            except Exception:
                pass
            self.zmq_context.term()
            self.logger.info("Shutdown complete.")

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            try:
                self.check_hardware_health()
                return {"status": "ok", "node": self.name}
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Hardware check failed: {exc}")

        @self.app.get("/status")
        async def status():
            return self.status_payload()

        @self.app.post("/prepare")
        async def prepare(payload: Dict[str, Any] = Body(default={})):
            run_id = payload.get("run_id")
            episode = payload.get("episode")
            try:
                ok = self.on_prepare(run_id, episode)
            except Exception as exc:
                self.status = NodeStatus.ERROR
                self.last_error = str(exc)
                self.logger.error(f"Prepare failed: {exc}")
                raise HTTPException(status_code=500, detail=str(exc))

            if not ok:
                self.status = NodeStatus.ERROR
                raise HTTPException(status_code=500, detail="Prepare rejected.")

            self.status = NodeStatus.READY
            self.run_id = run_id
            self.episode = episode
            return {"status": "ok"}

        @self.app.post("/start")
        async def start(payload: Dict[str, Any] = Body(default={})):
            if self.status != NodeStatus.READY:
                raise HTTPException(status_code=409, detail=f"Node not ready (status={self.status}).")

            run_id = payload.get("run_id")
            episode = payload.get("episode")
            start_time = payload.get("start_time")

            self.run_id = run_id
            self.episode = episode

            threading.Thread(
                target=self._exec_start, args=(run_id, episode, start_time), daemon=True
            ).start()
            return {"status": "accepted", "start_time": start_time}

        @self.app.post("/stop")
        async def stop(payload: Dict[str, Any] = Body(default={})):
            if not self.is_recording:
                raise HTTPException(status_code=409, detail="Not recording.")

            stop_time = payload.get("stop_time")
            threading.Thread(target=self._exec_stop, args=(stop_time,), daemon=True).start()
            return {"status": "accepted", "stop_time": stop_time}

        @self.app.post("/discard")
        async def discard(payload: Dict[str, Any] = Body(default={})):
            run_id = payload.get("run_id")
            episode = payload.get("episode")
            try:
                self.on_discard_run(run_id, episode)
            except Exception as exc:
                self.logger.error(f"Discard failed: {exc}")
                raise HTTPException(status_code=500, detail=str(exc))
            return {"status": "ok"}

        @self.app.post("/download")
        async def download():
            try:
                self.on_start_download()
            except Exception as exc:
                self.logger.error(f"Download trigger failed: {exc}")
                raise HTTPException(status_code=500, detail=str(exc))
            return {"status": "ok"}

        @self.app.post("/recover")
        async def recover():
            if self.status not in (NodeStatus.ERROR, NodeStatus.OFFLINE):
                raise HTTPException(
                    status_code=400,
                    detail=f"Node not in ERROR/OFFLINE state (status={self.status})"
                )
            if self._in_recovery:
                raise HTTPException(status_code=409, detail="Recovery already in progress")

            # Reset attempts to allow manual trigger
            self._recovery_attempts = 0
            # Run recovery in background thread
            threading.Thread(
                target=self._attempt_recovery,
                args=(Exception("Manual recovery trigger"),),
                daemon=True,
            ).start()
            return {"status": "recovery_started"}

    # Control helpers ----------------------------------------------------
    def _exec_start(self, run_id, episode, start_time=None):
        try:
            if start_time:
                precise_wait(float(start_time), time_func=time.time)
            self.is_recording = True
            self.status = NodeStatus.RECORDING
            self.publish_status()  # Notify orchestrator immediately
            self.on_start_recording(run_id, episode)
        except Exception as exc:
            self.is_recording = False
            self.status = NodeStatus.ERROR
            self.last_error = str(exc)
            self.logger.error(f"Start failed: {exc}")
            self.publish_status()  # Notify orchestrator of ERROR immediately

    def _exec_stop(self, stop_time=None):
        try:
            if stop_time:
                precise_wait(float(stop_time), time_func=time.time)
            self.on_stop_recording()
        except Exception as exc:
            self.status = NodeStatus.ERROR
            self.last_error = str(exc)
            self.logger.error(f"Stop failed: {exc}")
            self.publish_status()  # Notify orchestrator of ERROR immediately
        finally:
            self.is_recording = False
            if self.status not in (NodeStatus.ERROR, NodeStatus.SAVING):
                self.status = NodeStatus.IDLE
            self.run_id = None
            self.episode = None
            self.publish_status()  # Notify orchestrator immediately

    def _main_loop_wrapper(self):
        while self.is_running:
            try:
                self.main_loop()
                break  # Normal exit
            except Exception as exc:
                # Discard recording immediately
                if self.is_recording:
                    self._discard_current_recording(f"Main loop error: {exc}")

                self.status = NodeStatus.ERROR
                self.last_error = str(exc)
                self.logger.error(f"Main loop error: {exc}")
                self.publish_status()

                # Attempt recovery
                if self._attempt_recovery(exc):
                    self.logger.info("Restarting main loop after recovery...")
                    continue  # Recovery succeeded, restart main_loop
                else:
                    break  # Recovery failed, exit loop

    def _heartbeat_loop(self):
        last_health_check = 0
        while self.is_running:
            try:
                self.publish_status()
            except Exception:
                pass

            # Skip health check during recovery
            if self._in_recovery:
                time.sleep(0.5)
                continue

            # Periodic hardware health check
            now = time.time()
            if now - last_health_check >= self.HEALTH_CHECK_INTERVAL:
                last_health_check = now
                try:
                    self.check_hardware_health()
                    self._health_check_failures = 0
                    self._recovery_attempts = 0  # Continuous health resets recovery count

                    # Auto-recover from ERROR if hardware is now healthy (external fix)
                    if self.status == NodeStatus.ERROR and not self.is_recording:
                        self.logger.info("Hardware healthy again, resetting to IDLE")
                        self.status = NodeStatus.IDLE
                        self.last_error = None
                        self.publish_status()

                except Exception as exc:
                    self._health_check_failures += 1
                    self.logger.warning(
                        f"Health check failed ({self._health_check_failures}/{self.HEALTH_CHECK_MAX_FAILURES}): {exc}"
                    )
                    if self._health_check_failures >= self.HEALTH_CHECK_MAX_FAILURES:
                        # Discard recording immediately on hardware failure
                        if self.is_recording:
                            self._discard_current_recording(f"Health check failed: {exc}")

                        self.status = NodeStatus.ERROR
                        self.last_error = f"Hardware health check failed: {exc}"
                        self.logger.error(self.last_error)
                        self.publish_status()

                        # Attempt recovery
                        self._attempt_recovery(exc)

            time.sleep(0.5)

    def publish_status(self, force_status: NodeStatus = None):
        """
        Publish current status to orchestrator.
        Can be called from heartbeat loop, exception handlers, or shutdown.

        Args:
            force_status: Override the current status (e.g., OFFLINE for shutdown)
        """
        try:
            payload = self.status_payload()
            if force_status is not None:
                payload["status"] = force_status.value if isinstance(force_status, NodeStatus) else str(force_status)
            payload["ts"] = time.time()
            self.status_pub.send_json(payload)
        except Exception:
            pass

    # Public API ---------------------------------------------------------
    def status_payload(self) -> Dict[str, Any]:
        payload = {
            "node": self.name,
            "status": self.status.value if isinstance(self.status, NodeStatus) else str(self.status),
            "is_recording": self.is_recording,
            "run_id": self.run_id,
            "episode": self.episode,
            "last_error": self.last_error,
        }
        payload.update(self.extra_status())
        return payload

    def start(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

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
    def check_hardware_health(self):
        """
        Check if hardware is healthy and reachable.
        Subclasses must implement this to verify their specific hardware.
        Should raise an exception if hardware is not healthy.
        """
        ...

    def on_start_download(self):
        pass

    def on_discard_run(self, run_id, episode=None):  # noqa: ARG002
        pass

    def on_shutdown(self):
        pass

    def extra_status(self) -> Dict[str, Any]:
        return {}

    def get_iso_timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    # Recovery hooks (optional, subclass can override) -----------------------
    def can_recover(self, exc: Exception) -> bool:  # noqa: ARG002
        """
        Determine if recovery should be attempted for this exception.
        Subclass can override to filter recoverable exceptions.
        Default: return AUTO_RECOVERY_ENABLED.
        """
        return self.AUTO_RECOVERY_ENABLED

    def on_recover(self):
        """
        Re-initialize hardware / reconnect.
        Subclass implements: recreate driver, rediscover camera, reset modes, etc.
        Should raise exception if recovery fails.
        """
        pass

    def after_recover(self):
        """
        Post-recovery cleanup.
        Subclass implements: reset timing variables, clear buffers, reset counters.
        """
        pass

    def _cleanup_for_recovery(self):
        """
        Lightweight cleanup before recovery (less than on_shutdown).
        Release old resources but keep ZMQ/FastAPI running.
        """
        pass

    def _discard_current_recording(self, reason: str):
        """
        Discard current recording due to error.
        Subclass can override for specific cleanup.
        """
        if not self.is_recording:
            return

        self.logger.error("=" * 60)
        self.logger.error("!!! RECORDING ABORTED - DATA DISCARDED !!!")
        self.logger.error(f"Run: {self.run_id}, Episode: {self.episode}")
        self.logger.error(f"Reason: {reason}")
        self.logger.error("=" * 60)

        run_id = self.run_id
        episode = self.episode
        self.is_recording = False

        if run_id:
            try:
                self.on_discard_run(run_id, episode)
            except Exception as exc:
                self.logger.error(f"Discard cleanup failed: {exc}")

    def _attempt_recovery(self, exc: Exception) -> bool:
        """
        Unified recovery flow with exponential backoff.
        Returns True if recovery succeeded, False otherwise.
        """
        if not self.can_recover(exc):
            self.logger.info("Recovery not enabled for this error type")
            return False

        # Discard any in-progress recording FIRST
        if self.is_recording:
            self._discard_current_recording(f"Hardware failure triggered recovery: {exc}")

        if self._recovery_attempts >= self.MAX_RECOVERY_ATTEMPTS:
            self.logger.error(
                f"Max recovery attempts ({self.MAX_RECOVERY_ATTEMPTS}) reached, giving up"
            )
            return False

        self._in_recovery = True
        self._recovery_attempts += 1
        self.status = NodeStatus.RECOVERING
        self.publish_status()

        # Exponential backoff
        backoff = min(
            self.RECOVERY_BACKOFF_BASE * (2 ** (self._recovery_attempts - 1)),
            self.RECOVERY_BACKOFF_MAX,
        )
        self.logger.info(
            f"Recovery attempt {self._recovery_attempts}/{self.MAX_RECOVERY_ATTEMPTS}, "
            f"waiting {backoff:.1f}s..."
        )
        time.sleep(backoff)

        try:
            self._cleanup_for_recovery()
            self.on_recover()
            self.after_recover()

            # Recovery succeeded
            self._recovery_attempts = 0
            self._health_check_failures = 0
            self._in_recovery = False
            self.status = NodeStatus.IDLE
            self.last_error = None
            self.run_id = None
            self.episode = None
            self.logger.info("=" * 40)
            self.logger.info("Recovery successful!")
            self.logger.info("=" * 40)
            self.publish_status()
            return True

        except Exception as recover_exc:
            self.logger.error(f"Recovery failed: {recover_exc}")
            self._in_recovery = False
            self.status = NodeStatus.ERROR
            self.last_error = f"Recovery failed: {recover_exc}"
            self.publish_status()
            return False
