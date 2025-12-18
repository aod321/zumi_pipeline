import zmq
import time
import threading
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from zumi_util import precise_wait

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

class ZMQService(ABC):
    def __init__(self, name, sub_port=5555):
        self.name = name
        self.context = zmq.Context()
        
        # Subscribe to Orchestrator commands
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://localhost:{sub_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.is_running = True
        self.is_recording = False
        self.run_id = None
        self.scheduled_stop_time = None
        
        # Background thread for receiving commands
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)

    def start(self):
        logging.info(f"Service '{self.name}' Initializing...")
        self.on_init()
        logging.info(f"Service '{self.name}' Ready.")
        
        self.control_thread.start()
        
        try:
            self.main_loop()
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        self.is_running = False
        self.on_shutdown()
        logging.info(f"Service '{self.name}' Shutdown.")

    def _control_loop(self):
        """
        Listens for START_SYNC, STOP_SYNC, EXIT commands.
        Handles the precise wait for START commands internally.
        """
        while self.is_running:
            try:
                # Blocking receive
                msg = self.sub_socket.recv_json()
                cmd = msg.get("cmd")
                payload = msg.get("payload", {})
                
                if cmd == "START_SYNC":
                    target_ts = payload.get("start_time")
                    self.run_id = payload.get("run_id")
                    
                    # --- SYNC START LOGIC ---
                    # Calculate wait time
                    now = time.time()
                    wait_ms = (target_ts - now) * 1000
                    
                    if wait_ms > 0:
                        logging.info(f"Sync Start: Waiting {wait_ms:.2f}ms...")
                        # Block this thread until exact start time
                        precise_wait(target_ts, time_func=time.time)
                    else:
                        logging.warning(f"Sync Start: MISSED target by {abs(wait_ms):.2f}ms! Starting Now.")
                    
                    # Reset state and flag start
                    self.scheduled_stop_time = None
                    self.is_recording = True
                    self.on_start_recording(self.run_id)
                    
                elif cmd == "STOP_SYNC":
                    stop_ts = payload.get("stop_time")
                    logging.info(f"Sync Stop: Scheduled for T+{ (stop_ts - time.time())*1000 :.2f}ms")
                    self.scheduled_stop_time = stop_ts
                    # Notify subclass to handle stop wait (useful for low-freq nodes)
                    self.on_schedule_stop(stop_ts)
                        
                elif cmd == "EXIT":
                    self.is_running = False
                    break
            except zmq.ZMQError:
                pass

    def get_iso_timestamp(self):
        """Returns ISO 8601 timestamp with microseconds for data logging."""
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z'

    # --- Abstract Methods ---
    @abstractmethod
    def on_init(self): pass
    
    @abstractmethod
    def main_loop(self): pass 
    
    @abstractmethod
    def on_start_recording(self, run_id): pass
    
    @abstractmethod
    def on_stop_recording(self): pass

    @abstractmethod
    def on_shutdown(self): pass
    
    def on_schedule_stop(self, stop_ts):
        """Optional hook for subclasses (e.g., GoPro) to handle stop wait"""
        pass