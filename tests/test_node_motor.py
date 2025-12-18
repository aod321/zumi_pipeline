import zmq
import time
import os
import glob
import subprocess
import sys
import numpy as np
from datetime import datetime

# Configuration
ZMQ_PORT = "5555"
TARGET_FREQ = 200.0  # Hz
MIN_ACCEPTABLE_FREQ = 120.0  # Hz (Lower bound for "Pass")
RECORD_DURATION = 3.0  # seconds

def run_test():
    # 1. Setup ZMQ Orchestrator Mock
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://*:{ZMQ_PORT}")
    
    print("=== Motor Node Performance Test ===")
    print(f"Target Frequency: {TARGET_FREQ} Hz")
    
    # 2. System Time Check
    print(f"[System] Local Time: {datetime.now()}")
    
    # 3. Start node_motor.py as subprocess
    print(">>> Starting node_motor.py...")
    # Using sys.executable ensures we use the same python interpreter (venv friendly)
    motor_process = subprocess.Popen([sys.executable, "node_motor.py"], cwd=os.getcwd())
    
    # Wait for hardware initialization (Serial open + MIT Handshake takes ~2-3s)
    print(">>> Waiting 4s for hardware init...")
    time.sleep(4)
    
    run_id = f"test_{int(time.time())}"
    
    try:
        # 4. Prepare Test Sequence
        # We start T+1.0s from now to give ZMQ time to sync
        start_time = time.time() + 1.0
        stop_time = start_time + RECORD_DURATION

        # 5. Send START_SYNC
        print(f">>> Sending START_SYNC (Duration: {RECORD_DURATION}s)...")
        pub_socket.send_json({
            "cmd": "START_SYNC",
            "payload": {
                "run_id": run_id,
                "start_time": start_time,
                # Optional: The node supports auto-stop if we send scheduled_stop_time
                # "scheduled_stop_time": stop_time
            }
        })

        # Wait for the recording duration + buffer
        time.sleep(1.0 + RECORD_DURATION + 0.5)

        # 6. Send STOP_SYNC
        # Even if scheduled stop is used, sending STOP is good practice
        print(">>> Sending STOP_SYNC...")
        pub_socket.send_json({
            "cmd": "STOP_SYNC",
            "payload": {
                "stop_time": time.time()
            }
        })

        # Wait for file save and GC (Give it 2s to flush buffer to disk)
        print(">>> Waiting for disk I/O...")
        time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        # Clean up process
        print(">>> Stopping node_motor.py...")
        motor_process.terminate()
        try:
            motor_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print("Force killing process...")
            motor_process.kill()
            
    # 7. Verification & Analysis
    print("\n=== Data Verification ===")
    
    # Look for .npz file
    expected_file = f"data/{run_id}_motor.npz"
    
    if not os.path.exists(expected_file):
        print(f"FAIL: File {expected_file} not found!")
        return
    
    print(f"Reading file: {expected_file}")
    
    try:
        with np.load(expected_file) as npz:
            # Check keys
            if 'data' not in npz or 'columns' not in npz:
                print("FAIL: Invalid .npz structure (missing 'data' or 'columns')")
                return

            raw_data = npz['data']  # Shape: (N, 5)
            columns = npz['columns']  # ['ts', 'pos', 'vel', 'tau', 'iter']

            print(f"Data Shape: {raw_data.shape}")
            print(f"Columns: {columns}")

    except Exception as e:
        print(f"FAIL: Could not load numpy file: {e}")
        return

    # 8. Performance Analysis
    count = len(raw_data)
    if count < 10:
        print(f"FAIL: Not enough frames ({count}) to analyze.")
        return

    # Extract timestamps (Column 0)
    timestamps = raw_data[:, 0]

    # Check monotonicity
    if np.any(np.diff(timestamps) <= 0):
        print("WARNING: Timestamps are not strictly monotonic! (Clock reset?)")

    # Calculate dt (time delta between frames)
    dt = np.diff(timestamps)

    # Filter out absurdly large gaps (e.g., if there was a pause before start)
    # Usually not needed if logic is tight, but good for safety
    valid_dt = dt[1:]

    if len(valid_dt) == 0:
        print("FAIL: No valid time intervals.")
        return

    # Frequencies
    freqs = 1.0 / valid_dt

    avg_freq = np.mean(freqs)
    std_freq = np.std(freqs)
    min_freq = np.min(freqs)
    max_freq = np.max(freqs)

    # Jitter (Standard Deviation of dt)
    jitter_ms = np.std(valid_dt) * 1000.0

    print("-" * 30)
    print(f"STATS (N={count})")
    print("-" * 30)
    print(f"Avg Freq:      {avg_freq:.2f} Hz")
    print(f"Std Dev Freq:  {std_freq:.2f} Hz")
    print(f"Min Instant:   {min_freq:.2f} Hz")
    print(f"Max Instant:   {max_freq:.2f} Hz")
    print(f"Loop Jitter:   {jitter_ms:.3f} ms")
    print("-" * 30)
    
    # 9. Pass/Fail Logic
    # Criteria 1: Average Frequency close to target
    # Allow 5% deviation on average (190-210Hz)
    freq_ok = abs(avg_freq - TARGET_FREQ) < (TARGET_FREQ * 0.05)

    # Criteria 2: Stability (No drops below threshold)
    # A drop below 120Hz implies a gap > 8.33ms (target is 5ms)
    drops = np.sum(freqs < MIN_ACCEPTABLE_FREQ)
    stability_ok = (drops == 0)

    if freq_ok and stability_ok:
        print("\n✅ RESULT: PASS")
    else:
        print("\n❌ RESULT: FAIL")
        if not freq_ok:
            print(f"   Reason: Average frequency {avg_freq:.2f}Hz is too far from {TARGET_FREQ}Hz")
        if not stability_ok:
            print(f"   Reason: {drops} frames dropped below {MIN_ACCEPTABLE_FREQ}Hz (Potential Lag Spikes)")

if __name__ == "__main__":
    run_test()
