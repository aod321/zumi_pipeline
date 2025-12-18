import time
import json
import threading
import os
import sys
import gc
import numpy as np

from zumi_core import ZMQService
from zumi_util import precise_wait

# Ensure we can import DM_CAN and serial
try:
    from DM_CAN import *
    import serial
    from dotenv import load_dotenv
except ImportError as e:
    print(f"[Motor] Critical Import Error: {e}")
    sys.exit(1)

class MotorNode(ZMQService):
    def on_init(self):
        self.data_buffer = []
        self.lock = threading.Lock()
        
        # --- Config ---
        self.target_freq = 200.0
        self.dt = 1.0 / self.target_freq
        
        # --- Hardware Setup ---
        print("[Motor] Hardware Init (MIT Mode)...")
        
        # 1. Get Parameters from .env
        load_dotenv()
        
        # Use defaults if not in env
        self.slave_id = int(os.getenv("MOTOR_SLAVE_ID", "0x07"), 16)
        self.master_id = int(os.getenv("MOTOR_MASTER_ID", "0x17"), 16)
        self.serial_port = os.getenv("MOTOR_SERIAL_PORT", "/dev/ttyACM0")

        if "MOTOR_PORT" in os.environ:
            self.serial_port = os.environ["MOTOR_PORT"]

        print(f"[Motor] Config: Port={self.serial_port}, SlaveID={hex(self.slave_id)}, MasterID={hex(self.master_id)}")

        # 2. Initialize Serial
        try:
            self.ser = serial.Serial(self.serial_port, 921600, timeout=0.5)
            if not self.ser.is_open:
                self.ser.open()
        except Exception as e:
            msg = f"[Motor] Failed to open serial port '{self.serial_port}': {e}"
            print(msg)
            raise RuntimeError(msg)

        # 3. Initialize Motor Objects
        self.motor = Motor(DM_Motor_Type.DMH3510, self.slave_id, self.master_id)
        self.ctrl = MotorControl(self.ser)
        self.ctrl.addMotor(self.motor)

        # 4. Check Connection & Enable
        print(f"[Motor] Connecting to motor SlaveID={hex(self.slave_id)}...")
        if self.ctrl.switchControlMode(self.motor, Control_Type.MIT):
            print("[Motor] Switch MIT mode SUCCESS")
        else:
            msg = f"[Motor] Failed to switch to MIT mode. Motor {hex(self.slave_id)} might be offline."
            print(msg)
            self.ser.close()
            raise RuntimeError(msg)

        # Enable Motor
        self.ctrl.enable(self.motor)
        self.ctrl.set_zero_position(self.motor)
        print("[Motor] Motor Enabled and Zero Position Set.")

    def on_start_recording(self, run_id):
        self.iter_idx = 0

        # Disable GC
        # Prevent "stop-the-world" garbage collection pauses during critical recording
        gc.disable()

        print(f"[Motor] Recording STARTED: {run_id} (GC Disabled)")

    def on_stop_recording(self):
        # We don't re-enable GC here immediately to avoid jitter during the "Falling Edge" phase.
        # It will be re-enabled in _save_to_disk.
        print("[Motor] Recording STOPPED. Saving scheduled...")

    def _save_to_disk(self):
        """
        Runs in a background thread.
        Optimized to save data as Compressed NumPy Binary (.npz).
        """
        os.makedirs("data", exist_ok=True)
        # Using .npz extension for compressed numpy archive
        filename = f"data/{self.run_id}_motor.npz"
        
        with self.lock:
            if not self.data_buffer:
                print("[Motor] Warning: No data to save.")
                # Even if empty, re-enable GC
                gc.enable()
                return
            # Atomic swap
            raw_tuples = self.data_buffer
            self.data_buffer = []

        print(f"[Motor] Converting {len(raw_tuples)} frames to NumPy...")
        
        try:
            # OPTIMIZATION 2: Save as NumPy Binary
            # Much faster to write and compact on disk compared to JSON.
            # 1. Convert list of tuples to numpy array (N, 5)
            # Columns: [timestamp, pos, vel, tau, iter_idx]
            data_array = np.array(raw_tuples, dtype=np.float64)
            
            # 2. Define column names for reference
            col_names = np.array(["ts", "pos", "vel", "tau", "iter"])
            
            # 3. Save compressed
            np.savez_compressed(filename, data=data_array, columns=col_names)
            
            print(f"[Motor] Saved {data_array.shape} to {filename}")

        except Exception as e:
            print(f"[Motor] Failed to save data to {filename}: {e}")
        
        finally:
            # OPTIMIZATION 1 (Restore): Re-enable and force collect GC
            # Clean up the large temporary lists created during recording
            gc.enable()
            gc.collect()
            print("[Motor] GC Re-enabled and Collected.")

    def main_loop(self):
        """
        High-Performance Control Loop (200Hz)
        Optimizations:
        1. Decoupled timing logic.
        2. Zero-Lock Recording with Tuple storage.
        3. Local Variable Caching (Loop Invariant Code Motion).
        """
        print(f"[Motor] Loop running at {self.target_freq}Hz")
        
        motor_ref = self.motor
        ctrl_mit = self.ctrl.controlMIT
        
        get_pos = motor_ref.getPosition
        get_vel = motor_ref.getVelocity
        get_tau = motor_ref.getTorque
        
        get_time = time.time
        get_monotonic = time.monotonic
        
        # Initialize timing
        next_wake_time = get_monotonic()
        
        # Local state
        local_buffer = []
        local_append = None # Will be bound when recording starts
        was_recording = False 

        while self.is_running:
            # --- 1. Hardware Communication (Heartbeat) ---
            # Always send command to keep motor active and read state
            try:
                ctrl_mit(motor_ref, 0.0, 0.0, 0.0, 0.0, 0.0)
            except Exception as e:
                print(f"[Motor] Control Error: {e}")
            
            # --- 2. Auto-Stop Logic ---
            if self.is_recording and self.scheduled_stop_time:
                # Use cached get_time
                if get_time() >= self.scheduled_stop_time:
                    self.is_recording = False
                    self.on_stop_recording()
                    self.scheduled_stop_time = None

            # --- 3. Data Recording (Hot Path) ---
            if self.is_recording:
                # Rising Edge
                if not was_recording:
                    local_buffer = []
                    # Cache the append method of this specific list instance
                    local_append = local_buffer.append 
                    was_recording = True
                
                # Fast Tuple Creation with Local Vars
                frame = (
                    get_time(),           # Cached time.time
                    float(get_pos()),     # Cached getter
                    float(get_vel()),     # Cached getter
                    float(get_tau()),     # Cached getter
                    int(self.iter_idx)
                )
                
                # Fast Append
                local_append(frame)
                self.iter_idx += 1

            # --- 4. Handle Stop (Falling Edge) ---
            elif was_recording:
                print(f"[Motor] Committing {len(local_buffer)} frames...")
                with self.lock:
                    self.data_buffer = local_buffer
                
                threading.Thread(target=self._save_to_disk).start()
                
                local_buffer = [] 
                local_append = None # clear reference
                was_recording = False

            # --- 5. Precise Timing ---
            next_wake_time += self.dt
            now = get_monotonic() # Cached time.monotonic
            
            # Anti-Lag
            if now > next_wake_time:
                if now - next_wake_time > 0.005: 
                    # print(f"[Motor] Lag: {(now - next_wake_time)*1000:.2f}ms") 
                    next_wake_time = now + self.dt
                    
            precise_wait(next_wake_time)

    def on_shutdown(self):
        print("[Motor] Shutting down...")
        # Ensure GC is back on if we crash/exit mid-recording
        gc.enable() 
        try:
            if hasattr(self, 'ctrl') and hasattr(self, 'motor'):
                self.ctrl.disable(self.motor)
            if hasattr(self, 'ser') and self.ser.is_open:
                self.ser.close()
        except Exception as e:
            print(f"[Motor] Shutdown Error: {e}")

if __name__ == "__main__":
    node = MotorNode(name="DM3510")
    node.start()
