import os
import socket
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import zmq

# Configuration
TARGET_FREQ = 200.0  # Hz
MIN_ACCEPTABLE_FREQ = 120.0  # Hz (Lower bound for "Pass")
MAX_ALLOWED_DROPS = 1  # Allow a single lag spike around start/stop
RECORD_DURATION = 3.0  # seconds


def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run_test(driver: str = "mock"):
    # 1. Setup ZMQ Orchestrator Mock
    cmd_port = get_free_port()
    status_port = get_free_port()

    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://*:{cmd_port}")
    
    print("=== Motor Node Performance Test ===")
    print(f"Target Frequency: {TARGET_FREQ} Hz")
    
    # 2. System Time Check
    print(f"[System] Local Time: {datetime.now()}")
    
    # 3. Start node_motor.py as subprocess
    print(">>> Starting node_motor.py...")
    env = os.environ.copy()
    env["MOTOR_DRIVER"] = driver
    env["ZUMI_CMD_PORT"] = str(cmd_port)
    env["ZUMI_STATUS_PORT"] = str(status_port)
    motor_process = subprocess.Popen([sys.executable, "node_motor.py"], cwd=os.getcwd(), env=env)
    
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
        pub_socket.send_json(
            {
                "cmd": "START_SYNC",
                "payload": {
                    "run_id": run_id,
                    "start_time": start_time,
                },
            }
        )

        # 6. Schedule STOP_SYNC slightly in the future to avoid past timestamps
        stop_schedule = stop_time
        print(">>> Scheduling STOP_SYNC...")
        pub_socket.send_json(
            {
                "cmd": "STOP_SYNC",
                "payload": {
                    "stop_time": stop_schedule,
                },
            }
        )

        # Wait until after stop + small buffer
        wait_time = max(0, stop_schedule - time.time() + 1.0)
        time.sleep(wait_time)

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
    stability_ok = drops <= MAX_ALLOWED_DROPS

    if freq_ok and stability_ok:
        print("\n✅ RESULT: PASS")
    else:
        print("\n❌ RESULT: FAIL")
        if not freq_ok:
            print(f"   Reason: Average frequency {avg_freq:.2f}Hz is too far from {TARGET_FREQ}Hz")
        if not stability_ok:
            print(
                f"   Reason: {drops} frames dropped below {MIN_ACCEPTABLE_FREQ}Hz "
                f"(allowed {MAX_ALLOWED_DROPS}, Potential Lag Spikes)"
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Motor node integration test")
    parser.add_argument(
        "--driver",
        choices=["mock", "dm"],
        default=os.getenv("MOTOR_DRIVER", "mock"),
        help="Motor driver to use (mock for hardware-free, dm for real motor)",
    )
    args = parser.parse_args()
    run_test(driver=args.driver)
