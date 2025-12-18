import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

from zumi_config import GOPRO_CONF, HTTP_CONF, NodeStatus, STORAGE_CONF
from zumi_core import NodeHTTPService

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("GoPro")


class QueueManager:
    def __init__(self, db_file=None):
        self.db_file = Path(db_file) if db_file else STORAGE_CONF.DATA_DIR / "gopro_queue.json"
        self.queue = self._load()

    def _load(self):
        if self.db_file.exists():
            try:
                with open(self.db_file, "r") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning(f"Failed to load queue file: {exc}. Starting new.")
        return []

    def save(self):
        try:
            self.db_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_file, "w") as fh:
                json.dump(self.queue, fh, indent=2)
        except Exception as exc:
            logger.error(f"Failed to save queue: {exc}")

    def add_task(self, run_id, episode, folder, filename):
        ep_val = episode or 1
        self.queue = [t for t in self.queue if not (t.get("run_id") == run_id and t.get("episode") == ep_val)]
        task = {
            "run_id": run_id,
            "episode": ep_val,
            "folder": folder,
            "filename": filename,
            "status": "pending",
            "timestamp": time.time(),
        }
        self.queue.append(task)
        self.save()
        logger.info(f"[Queue] Added: {run_id} -> {filename}")

    def discard_run(self, run_id, episode=None):
        found = False
        for task in self.queue:
            if task["run_id"] == run_id and (episode is None or task.get("episode") == episode):
                task["status"] = "discarded"
                found = True
                logger.info(f"[Queue] Run {run_id} ep={task.get('episode')} marked as discarded.")

                orig_name = task.get("filename")
                if orig_name:
                    ep_val = task.get("episode", 1) or 1
                    ep_tag = f"ep{int(ep_val):03d}"
                    save_name = f"{run_id}_{ep_tag}_{orig_name}"
                    save_path = STORAGE_CONF.VIDEO_DIR / save_name
                    if save_path.exists():
                        try:
                            save_path.unlink()
                            logger.info(f"[Queue] Deleted file: {save_path}")
                        except Exception as exc:
                            logger.error(f"[Queue] Failed to delete file {save_path}: {exc}")

                    imu_path = save_path.with_name(save_path.stem + "_imu.json")
                    if imu_path.exists():
                        try:
                            imu_path.unlink()
                            logger.info(f"[Queue] Deleted IMU file: {imu_path}")
                        except Exception:
                            pass
        if not found:
            logger.warning(f"[Queue] Could not find Run {run_id} to discard.")
        self.save()

    def mark_done(self, run_id, episode=None):
        for task in self.queue:
            if task["run_id"] == run_id and (episode is None or task.get("episode") == episode):
                task["status"] = "done"
        self.save()

    def get_pending_tasks(self):
        return [t for t in self.queue if t.get("status") == "pending"]


class GoProController:
    def __init__(self, ip=None, sn=None):
        self.ip = ip
        self.sn = sn
        self.session = requests.Session()

        if not self.ip:
            if self.sn:
                self.ip = self._get_ip_from_sn(self.sn)
                logger.info(f"Derived IP {self.ip} from SN {self.sn}")
            else:
                self.ip = self._discover_ip()
        if not self.ip:
            raise RuntimeError("GoPro IP not found. Please connect camera via USB.")

        self.base_url = f"http://{self.ip}:8080"
        self._check_connection()

    def _get_ip_from_sn(self, sn):
        if len(sn) < 3 or not sn[-3:].isdigit():
            return None
        return f"172.2{sn[-3:-2]}.1{sn[-2:]}.51"

    def _discover_ip(self):
        logger.info("Auto-discovering GoPro...")
        try:
            cmd = "ip -4 --oneline link | grep -v 'state DOWN' | grep -v LOOPBACK | grep -v 'NO-CARRIER'"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8")
            for line in output.strip().split("\n"):
                if "inet" not in line:
                    try:
                        dev = line.split(":")[1].strip()
                        ip_out = subprocess.check_output(f"ip -4 addr show dev {dev}", shell=True).decode("utf-8")
                    except Exception:
                        continue
                else:
                    ip_out = line
                match = re.search(r"inet (172\.2\d\.1\d\d)\.\d+", ip_out)
                if match:
                    subnet = match.group(1)
                    return f"{subnet}.51"
        except Exception:
            pass
        return None

    def _check_connection(self):
        try:
            self.session.get(f"{self.base_url}/gopro/camera/control/wired_usb?p=1", timeout=2)
            resp = self.session.get(f"{self.base_url}/gopro/camera/info", timeout=2)
            resp.raise_for_status()
            logger.info(f"GoPro Connected: {self.ip}")
        except Exception as exc:
            raise RuntimeError(f"GoPro connection failed: {exc}")

    def sync_time(self):
        logger.info("Synchronizing camera time with system time...")
        try:
            now = datetime.now()
            date_str = now.strftime("%Y_%m_%d")
            time_str = now.strftime("%H_%M_%S")
            tz_offset_sec = time.timezone
            is_dst = 1 if (time.localtime().tm_isdst > 0 and time.daylight) else 0
            tzone = int(-tz_offset_sec / 60)

            url = f"{self.base_url}/gopro/camera/set_date_time"
            params = {"date": date_str, "time": time_str, "tzone": tzone, "dst": is_dst}
            resp = self.session.get(url, params=params, timeout=5)
            resp.raise_for_status()
            logger.info("Camera time synchronized successfully.")
        except Exception as exc:
            logger.error(f"Error synchronizing camera time: {exc}")

    def send_labs_command(self, code):
        logger.info(f"Sending Labs command: {code}")
        try:
            url = f"{self.base_url}/gopro/qrcode"
            params = {"labs": 1, "code": code}
            resp = self.session.get(url, params=params, timeout=5)
            resp.raise_for_status()
            logger.info(f"Labs command sent successfully. Response: {resp.text}")
            return True
        except Exception as exc:
            logger.error(f"Error sending Labs command: {exc}")
            return False

    def mute_audio(self, mask=15):
        return self.send_labs_command(f"oMMUTE={mask}")

    def start(self):
        try:
            resp = self.session.get(f"{self.base_url}/gopro/camera/shutter/start", timeout=2)
            if resp.status_code != 200:
                time.sleep(0.5)
                self.session.get(f"{self.base_url}/gopro/camera/shutter/start", timeout=2)
        except Exception as exc:
            logger.error(f"Start Recording Failed: {exc}")

    def stop(self):
        try:
            self.session.get(f"{self.base_url}/gopro/camera/shutter/stop", timeout=2)
        except Exception as exc:
            logger.error(f"Stop Recording Failed: {exc}")

    def get_last_media_info(self):
        try:
            resp = self.session.get(f"{self.base_url}/gopro/media/last_captured", timeout=3)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error(f"Get last media failed: {exc}")
            return None

    def download_file(self, folder, filename, save_path):
        url = f"{self.base_url}/videos/DCIM/{folder}/{filename}"
        with self.session.get(url, stream=True, timeout=10) as resp:
            resp.raise_for_status()
            with open(save_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)


class GoProNode(NodeHTTPService):
    # GoPro reconnection is slower, use larger backoff
    RECOVERY_BACKOFF_BASE = 3.0
    RECOVERY_BACKOFF_MAX = 30.0

    def __init__(self, name="go_pro_node", mute_on_start=True):
        self.mute_on_start = mute_on_start
        self.current_run_id = None
        self.current_episode = None
        self.is_downloading = False
        super().__init__(name, host=HTTP_CONF.GOPRO_HOST, port=HTTP_CONF.GOPRO_PORT)

    def on_init(self):
        sn = GOPRO_CONF.SN
        ip = GOPRO_CONF.IP
        try:
            self.cam = GoProController(ip=ip, sn=sn)
        except RuntimeError as exc:
            logger.critical(str(exc))
            raise

        self.cam.sync_time()
        if self.mute_on_start:
            logger.info("Attempting to mute all audio channels (Labs feature)...")
            self.cam.mute_audio(15)
        else:
            logger.info("Skipping audio mute (disabled by user).")

        self.qm = QueueManager()
        pending_count = len(self.qm.get_pending_tasks())
        logger.info(f"Node Initialized. {pending_count} videos pending in queue.")

    def on_prepare(self, run_id, episode=None):
        try:
            self.cam._check_connection()
            return True
        except Exception as exc:
            logger.error(f"Prepare failed: {exc}")
            return False

    def on_start_recording(self, run_id, episode=None):
        self.current_run_id = run_id
        self.current_episode = episode
        threading.Thread(target=self.cam.start, daemon=True).start()
        ep_tag = f"ep{episode:03d}" if episode is not None else "ep001"
        logger.info(f"[Record] STARTED Run: {run_id} {ep_tag}")

    def on_stop_recording(self):
        if not self.current_run_id:
            logger.warning("[Record] Stop called but no active run; ignoring.")
            return
        logger.info("[Record] Stopping...")
        self.status = NodeStatus.SAVING
        try:
            self.cam.stop()
            time.sleep(1.5)
            info = self.cam.get_last_media_info()
            if info and info.get("file"):
                folder = info["folder"]
                filename = info["file"]
                self.qm.add_task(self.current_run_id, self.current_episode, folder, filename)
                if not self.is_downloading:
                    threading.Thread(target=self._download_worker, daemon=True).start()
            else:
                logger.error("[Record] Could not retrieve filename for this run!")
                self.status = NodeStatus.ERROR
        except Exception as exc:
            logger.error(f"[Record] Error during stop sequence: {exc}")
            self.status = NodeStatus.ERROR
        finally:
            self.current_run_id = None
            self.current_episode = None

    def on_start_download(self):
        if not self.is_downloading:
            if self.status != NodeStatus.RECORDING:
                self.status = NodeStatus.SAVING
            threading.Thread(target=self._download_worker, daemon=True).start()
        else:
            logger.info("Download already in progress.")

    def _download_worker(self):
        self.is_downloading = True
        tasks = self.qm.get_pending_tasks()
        if not tasks:
            logger.info("[Download] No pending tasks.")
            self.is_downloading = False
            if self.status == NodeStatus.SAVING:
                self.status = NodeStatus.IDLE
            return

        STORAGE_CONF.VIDEO_DIR.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            if task.get("status") != "pending":
                continue
            run_id = task["run_id"]
            episode = task.get("episode", 1)
            ep_val = episode or 1
            orig_name = task["filename"]
            folder = task["folder"]
            ep_tag = f"ep{int(ep_val):03d}"
            save_name = f"{run_id}_{ep_tag}_{orig_name}"
            save_path = STORAGE_CONF.VIDEO_DIR / save_name
            logger.info(f"[Download] Downloading {orig_name} -> {save_name} ...")
            try:
                self.cam.download_file(folder, orig_name, save_path)

                gopro_basename = os.path.splitext(orig_name)[0]
                old_motor_npz = STORAGE_CONF.DATA_DIR / f"{run_id}_{ep_tag}_motor.npz"
                new_motor_npz = STORAGE_CONF.DATA_DIR / f"{run_id}_{ep_tag}_{gopro_basename}_motor.npz"
                if old_motor_npz.exists():
                    try:
                        old_motor_npz.rename(new_motor_npz)
                        logger.info(f"[Sync] Renamed motor file: {old_motor_npz} -> {new_motor_npz}")
                    except Exception as exc:
                        logger.error(f"[Sync] Failed to rename motor file: {exc}")

                old_motor_meta = STORAGE_CONF.DATA_DIR / f"{run_id}_{ep_tag}_motor_meta.json"
                new_motor_meta = STORAGE_CONF.DATA_DIR / f"{run_id}_{ep_tag}_{gopro_basename}_motor_meta.json"
                if old_motor_meta.exists():
                    try:
                        old_motor_meta.rename(new_motor_meta)
                    except Exception:
                        pass

                self.qm.mark_done(run_id, ep_val)
                logger.info(f"[Download] Success: {save_name}")
            except Exception as exc:
                logger.error(f"[Download] Failed {save_name}: {exc}")

        self.is_downloading = False
        if self.status == NodeStatus.SAVING:
            self.status = NodeStatus.IDLE

    def check_hardware_health(self):
        """Check GoPro connection by verifying camera is reachable."""
        self.cam._check_connection()

    def main_loop(self):
        while self.is_running:
            # Check pending downloads
            if not self.is_downloading and self.qm.get_pending_tasks():
                threading.Thread(target=self._download_worker, daemon=True).start()
            time.sleep(1)

    def on_discard_run(self, run_id, episode=None):
        self.qm.discard_run(run_id, episode)
        try:
            if episode is not None:
                ep_val = episode or 1
                ep_tag = f"ep{int(ep_val):03d}"
                patterns = [
                    f"{run_id}_{ep_tag}_*.MP4",
                    f"{run_id}_{ep_tag}_*_imu.json",
                ]
            else:
                patterns = [f"{run_id}_*.MP4", f"{run_id}_*_imu.json"]
            for pattern in patterns:
                for path in STORAGE_CONF.VIDEO_DIR.glob(pattern):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        continue
        except Exception as exc:
            logger.error(f"[Queue] Discard cleanup failed: {exc}")

    def on_shutdown(self):
        logger.info("Shutdown.")

    def extra_status(self):
        pending = len(self.qm.get_pending_tasks()) if hasattr(self, "qm") else 0
        return {"is_downloading": self.is_downloading, "pending_tasks": pending}

    # Recovery methods -------------------------------------------------------
    def can_recover(self, exc: Exception) -> bool:
        """Recover from network/connection errors."""
        return isinstance(exc, (RuntimeError, requests.RequestException, OSError))

    def _cleanup_for_recovery(self):
        """Close old session."""
        try:
            if hasattr(self, "cam") and hasattr(self.cam, "session"):
                self.cam.session.close()
        except Exception:
            pass

    def on_recover(self):
        """Reconnect to GoPro with full re-initialization."""
        logger.info("Reconnecting to GoPro...")
        sn = GOPRO_CONF.SN
        ip = GOPRO_CONF.IP
        self.cam = GoProController(ip=ip, sn=sn)
        self.cam.sync_time()
        if self.mute_on_start:
            self.cam.mute_audio(15)
        logger.info("GoPro reconnected")

    def after_recover(self):
        """Reset download state."""
        self.is_downloading = False
        self.is_recording = False
        self.current_run_id = None
        self.current_episode = None

    def _discard_current_recording(self, reason: str):
        """Discard GoPro recording on error."""
        if not self.is_recording:
            return

        run_id = self.current_run_id
        episode = self.current_episode

        logger.error("=" * 60)
        logger.error("!!! RECORDING ABORTED - VIDEO DATA DISCARDED !!!")
        logger.error(f"Run: {run_id}, Episode: {episode}")
        logger.error(f"Reason: {reason}")
        logger.error("=" * 60)

        # Try to stop camera recording (best effort)
        try:
            self.cam.stop()
        except Exception:
            pass

        self.is_recording = False

        # Mark task as discarded in queue
        if run_id:
            self.qm.discard_run(run_id, episode)

        self.current_run_id = None
        self.current_episode = None


if __name__ == "__main__":
    mute = True
    if "--no-mute" in sys.argv:
        mute = False
    node = GoProNode(name="go_pro_node", mute_on_start=mute)
    node.start()
