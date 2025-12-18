import time
import threading
import os
import sys
import json
import requests
import re
import subprocess
import logging
import datetime
import zmq
import argparse
from dotenv import load_dotenv

# Ensure core library import
try:
    from zumi_core import ZMQService
    from zumi_util import precise_wait
except ImportError:
    print("Error: zumi_core.py or zumi_util.py not found.")
    sys.exit(1)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("GoPro")

# -------------------------------------------------------------------
# 1. Queue Manager (File persistence for tasks)
# -------------------------------------------------------------------
class QueueManager:
    """
    Manages recording queue, run_id<->filename mappings,
    and status: pending/done/discarded.
    """
    def __init__(self, db_file="gopro_queue.json"):
        self.db_file = db_file
        self.queue = self._load()

    def _load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load queue file: {e}. Starting new.")
                return []
        return []

    def save(self):
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.queue, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save queue: {e}")

    def add_task(self, run_id, folder, filename):
        # Remove preexisting (should not happen, but proactive)
        for t in self.queue:
            if t['run_id'] == run_id:
                logger.warning(f"RunID {run_id} already in queue, overwriting.")
                self.queue.remove(t)
                break
        task = {
            "run_id": run_id,
            "folder": folder,
            "filename": filename,  # GoPro .MP4 name
            "status": "pending",
            "timestamp": time.time()
        }
        self.queue.append(task)
        self.save()
        logger.info(f"[Queue] Added: {run_id} -> {filename}")

    def discard_run(self, run_id):
        found = False
        for t in self.queue:
            if t['run_id'] == run_id:
                t['status'] = 'discarded'
                found = True
                logger.info(f"[Queue] Run {run_id} marked as DISCARDED.")
                
                # Attempt to delete downloaded files if they exist
                orig_name = t.get('filename')
                if orig_name:
                    save_name = f"{run_id}_{orig_name}"
                    save_path = os.path.join("data/videos", save_name)
                    # Delete Video
                    if os.path.exists(save_path):
                        try:
                            os.remove(save_path)
                            logger.info(f"[Queue] Deleted file: {save_path}")
                        except Exception as e:
                            logger.error(f"[Queue] Failed to delete file {save_path}: {e}")
                    
                    # Delete IMU JSON (if extracted)
                    imu_path = save_path.replace(".MP4", "_imu.json")
                    if os.path.exists(imu_path):
                        try:
                            os.remove(imu_path)
                            logger.info(f"[Queue] Deleted IMU file: {imu_path}")
                        except Exception as e:
                             pass

        if not found:
            logger.warning(f"[Queue] Could not find Run {run_id} to discard.")
        self.save()

    def mark_done(self, run_id):
        for t in self.queue:
            if t['run_id'] == run_id:
                t['status'] = 'done'
        self.save()

    def get_pending_tasks(self):
        return [t for t in self.queue if t['status'] == 'pending']

# -------------------------------------------------------------------
# 2. GoPro Controller (API abstraction)
# -------------------------------------------------------------------
class GoProController:
    """
    Encapsulate GoPro HTTP-API,
    auto-discover IP, connection check, API ops, download.
    """
    def __init__(self, ip=None, sn=None):
        self.ip = ip
        self.sn = sn
        self.session = requests.Session()

        # Auto discover IP (prefer env-provided, or SN-based)
        if not self.ip:
            if self.sn:
                self.ip = self._get_ip_from_sn(self.sn)
                logger.info(f"Derived IP {self.ip} from SN {self.sn}")
            else:
                self.ip = self._discover_ip()
        if not self.ip:
            raise RuntimeError("GoPro IP not found. Please connect camera via USB.")

        self.base_url = f"http://{self.ip}:8080"

        # Enable wired control, verify camera present:
        self._check_connection()

    def _get_ip_from_sn(self, sn):
        # GoPro rule: 172.2X.1YZ.51 from last 3 SN digits
        if len(sn) < 3 or not sn[-3:].isdigit():
            return None
        return f"172.2{sn[-3:-2]}.1{sn[-2:]}.51"

    def _discover_ip(self):
        logger.info("Auto-discovering GoPro...")
        try:
            # Use `ip` to list up interfaces, exclude loopback
            cmd = "ip -4 --oneline link | grep -v 'state DOWN' | grep -v LOOPBACK | grep -v 'NO-CARRIER'"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8")
            for line in output.strip().split('\n'):
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
            # Enable Wired USB Control
            self.session.get(f"{self.base_url}/gopro/camera/control/wired_usb?p=1", timeout=2)
            r = self.session.get(f"{self.base_url}/gopro/camera/info", timeout=2)
            r.raise_for_status()
            logger.info(f"GoPro Connected: {self.ip}")
        except Exception as e:
            raise RuntimeError(f"GoPro connection failed: {e}")

    def sync_time(self):
        """Synchronize camera time with system time."""
        logger.info("Synchronizing camera time with system time...")
        try:
            # Get Local time (wall clock) because we are sending the local timezone offset.
            # If we send UTC time with a non-zero timezone offset, the camera will shift it BACK.
            now = datetime.datetime.now()
            date_str = now.strftime("%Y_%m_%d")
            time_str = now.strftime("%H_%M_%S")
            
            # Calculate timezone offset
            # Using standard timezone offset (not DST-adjusted) for tzone
            # and letting dst flag handle the shift if supported.
            tz_offset_sec = time.timezone
            
            # Check if DST is currently active
            is_dst = 1 if (time.localtime().tm_isdst > 0 and time.daylight) else 0
            
            # API expects offset in minutes. Python gives seconds west of UTC.
            # Example: UTC-8 -> time.timezone is 28800. -28800/60 = -480.
            tzone = int(-tz_offset_sec / 60)

            logger.info(f"Setting date: {date_str}, time: {time_str}, tzone: {tzone}, dst: {is_dst}")
            
            # /gopro/camera/set_date_time
            url = f"{self.base_url}/gopro/camera/set_date_time"
            params = {
                'date': date_str,
                'time': time_str,
                'tzone': tzone,
                'dst': is_dst
            }
            
            r = self.session.get(url, params=params, timeout=5)
            r.raise_for_status()
            logger.info("Camera time synchronized successfully.")
            
        except Exception as e:
            logger.error(f"Error synchronizing camera time: {e}")

    def send_labs_command(self, code):
        """
        Send a GoPro Labs command via the QR code API endpoint.
        Requires GoPro Labs firmware.
        """
        logger.info(f"Sending Labs command: {code}")
        try:
            url = f"{self.base_url}/gopro/qrcode"
            params = {
                'labs': 1,
                'code': code
            }
            r = self.session.get(url, params=params, timeout=5)
            r.raise_for_status()
            logger.info(f"Labs command sent successfully. Response: {r.text}")
            return True
        except Exception as e:
            logger.error(f"Error sending Labs command: {e}")
            return False

    def mute_audio(self, mask=15):
        """
        Mute one or more channels of audio.
        mask: Binary mask for channels 4321. Default 15 = mute all.
        """
        return self.send_labs_command(f"oMMUTE={mask}")

    def start(self):
        try:
            r = self.session.get(f"{self.base_url}/gopro/camera/shutter/start", timeout=2)
            if r.status_code != 200:
                time.sleep(0.5)
                self.session.get(f"{self.base_url}/gopro/camera/shutter/start", timeout=2)
        except Exception as e:
            logger.error(f"Start Recording Failed: {e}")

    def stop(self):
        try:
            self.session.get(f"{self.base_url}/gopro/camera/shutter/stop", timeout=2)
        except Exception as e:
            logger.error(f"Stop Recording Failed: {e}")

    def get_last_media_info(self):
        try:
            r = self.session.get(f"{self.base_url}/gopro/media/last_captured", timeout=3)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Get last media failed: {e}")
            return None

    def download_file(self, folder, filename, save_path):
        url = f"{self.base_url}/videos/DCIM/{folder}/{filename}"
        with self.session.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)

# -------------------------------------------------------------------
# 3. GoPro Node (ZMQ/logic glue)
# -------------------------------------------------------------------
class GoProNode(ZMQService):
    def __init__(self, name, mute_on_start=True):
        super().__init__(name)
        self.mute_on_start = mute_on_start

    def on_init(self):
        load_dotenv()
        sn = os.getenv("GOPRO_SN")
        ip = os.getenv("GOPRO_IP")
        try:
            self.cam = GoProController(ip=ip, sn=sn)
        except RuntimeError as e:
            logger.critical(str(e))
            if __name__ != "__main__":
                sys.exit(1)
        
        # Sync camera time at startup
        self.cam.sync_time()

        # Optional: Mute all audio channels (GoPro Labs)
        if self.mute_on_start:
            # NOTE: If you want to enable this mute control feature, you need to ensure that the
            # GoPro Labs firmware version is not lower than 2.04.70 released on August 25, 2025.
            # In addition, it has only been tested on the HERO13 Black model.
            # Other older models can choose to turn off this feature.
            logger.info("Attempting to mute all audio channels (Labs feature)...")
            self.cam.mute_audio(15)
        else:
            logger.info("Skipping audio mute (disabled by user).")
        
        self.qm = QueueManager()
        pending_count = len(self.qm.get_pending_tasks())
        logger.info(f"Node Initialized. {pending_count} videos pending in queue.")

        self.current_run_id = None
        self.is_downloading = False

    def _control_loop(self):
        """
        Handle ZMQ commands:
         - START_SYNC
         - STOP_SYNC
         - DISCARD_RUN
         - START_DOWNLOAD
        """
        while self.is_running:
            try:
                if self.sub_socket.poll(timeout=100):
                    msg = self.sub_socket.recv_json()
                    cmd = msg.get("cmd")
                    payload = msg.get("payload", {})
                    if cmd == "START_SYNC":
                        target_ts = payload.get("start_time")
                        self.run_id = payload.get("run_id")
                        precise_wait(target_ts, time_func=time.time)
                        self.scheduled_stop_time = None
                        self.is_recording = True
                        self.on_start_recording(self.run_id)
                    elif cmd == "STOP_SYNC":
                        stop_ts = payload.get("stop_time")
                        self.on_schedule_stop(stop_ts)
                    elif cmd == "DISCARD_RUN":
                        run_id = payload.get("run_id")
                        self.qm.discard_run(run_id)
                    elif cmd == "START_DOWNLOAD":
                        if not self.is_downloading:
                            t = threading.Thread(target=self._batch_download_process, daemon=True)
                            t.start()
                        else:
                            logger.warning("Download already in progress.")
                    elif cmd == "EXIT":
                        self.is_running = False
                        break
            except zmq.ZMQError:
                pass

    def on_start_recording(self, run_id):
        self.current_run_id = run_id
        threading.Thread(target=self.cam.start, daemon=True).start()
        logger.info(f"[Record] STARTED Run: {run_id}")

    def on_schedule_stop(self, stop_ts):
        threading.Thread(target=self._exec_sync_stop, args=(stop_ts,), daemon=True).start()

    def _exec_sync_stop(self, stop_ts):
        precise_wait(stop_ts, time_func=time.time)
        self.on_stop_recording()

    def on_stop_recording(self):
        logger.info("[Record] Stopping...")
        self.is_recording = False
        try:
            self.cam.stop()
            time.sleep(1.5)
            info = self.cam.get_last_media_info()
            if info and info.get('file'):
                folder = info['folder']
                filename = info['file']
                self.qm.add_task(self.current_run_id, folder, filename)
            else:
                logger.error("[Record] Could not retrieve filename for this run!")
        except Exception as e:
            logger.error(f"[Record] Error during stop sequence: {e}")

        self.current_run_id = None

    def _batch_download_process(self):
        self.is_downloading = True
        tasks = self.qm.get_pending_tasks()
        if not tasks:
            logger.info("[Download] No pending tasks.")
            self.is_downloading = False
            return
        logger.info(f"[Download] Starting batch download for {len(tasks)} files...")

        save_dir = "data/videos"
        os.makedirs(save_dir, exist_ok=True)

        for task in tasks:
            if task['status'] != 'pending':
                continue
            run_id = task['run_id']
            orig_name = task['filename']
            folder = task['folder']
            save_name = f"{run_id}_{orig_name}"
            save_path = os.path.join(save_dir, save_name)
            logger.info(f"[Download] Downloading {orig_name} -> {save_name} ...")
            try:
                self.cam.download_file(folder, orig_name, save_path)
                
                # --- RENAME MOTOR FILE ---
                # Naming Standard: run_xxx_GX01..._motor.npz
                # Current Motor File: data/run_xxx_motor.npz
                # We want to insert the GoPro filename (without extension) into the motor filename
                gopro_basename = os.path.splitext(orig_name)[0] # GX011234
                
                old_motor_path = f"data/{run_id}_motor.npz"
                new_motor_path = f"data/{run_id}_{gopro_basename}_motor.npz"
                
                if os.path.exists(old_motor_path):
                     try:
                         os.rename(old_motor_path, new_motor_path)
                         logger.info(f"[Sync] Renamed motor file: {old_motor_path} -> {new_motor_path}")
                     except Exception as e:
                         logger.error(f"[Sync] Failed to rename motor file: {e}")
                
                # Check JSON too just in case
                old_motor_json = f"data/{run_id}_motor.json"
                new_motor_json = f"data/{run_id}_{gopro_basename}_motor.json"
                if os.path.exists(old_motor_json):
                     try:
                         os.rename(old_motor_json, new_motor_json)
                     except Exception: pass
                
                self.qm.mark_done(run_id)
                logger.info(f"[Download] Success: {save_name}")
            except Exception as e:
                logger.error(f"[Download] Failed {save_name}: {e}")
                # retry next time
        logger.info("[Download] Batch process finished.")
        self.is_downloading = False

    def main_loop(self):
        while self.is_running:
            time.sleep(1)

    def on_shutdown(self):
        logger.info("Shutdown.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoPro Node Service")
    parser.add_argument("--no-mute", action="store_true", help="Disable auto-mute of audio on startup")
    args = parser.parse_args()

    mute_on_start = not args.no_mute

    node = GoProNode(name="go_pro_node", mute_on_start=mute_on_start)
    node.start()