import time
import json
import threading
from zumi_core import ZMQService
from zumi_util import precise_wait

class GoProNode(ZMQService):
    def on_init(self):
        # Mock connection
        self.ip = "172.xx.xx.xx"
        print(f"[GoPro] Connected to {self.ip}")

    def on_start_recording(self, run_id):
        # The parent class has already waited for the exact start time.
        # So we trigger the API immediately here.
        ts = self.get_iso_timestamp()
        print(f"[GoPro] SHUTTER ON at {ts}")
        # requests.get(f"http://{self.ip}/gopro/camera/shutter/start")

    def on_schedule_stop(self, stop_ts):
        """
        GoPro doesn't have a high-freq loop. 
        We spawn a thread to wait for the exact stop time.
        """
        threading.Thread(target=self._exec_sync_stop, args=(stop_ts,)).start()

    def _exec_sync_stop(self, stop_ts):
        # 1. Wait precisely until stop time
        precise_wait(stop_ts, time_func=time.time)
        
        # 2. Trigger API
        self.is_recording = False
        self.on_stop_recording()

    def on_stop_recording(self):
        ts = self.get_iso_timestamp()
        print(f"[GoPro] SHUTTER OFF at {ts}")
        # requests.get(f"http://{self.ip}/gopro/camera/shutter/stop")
        # Logic to download file or tag metadata...

    def main_loop(self):
        # Low frequency heartbeat loop
        while self.is_running:
            time.sleep(1)

    def on_shutdown(self):
        pass

if __name__ == "__main__":
    node = GoProNode(name="GoPro10")
    node.start()