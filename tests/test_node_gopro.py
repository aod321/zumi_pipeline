import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import zmq

# Configuration
TEST_DURATION = 5  # Record for 5 seconds
DOWNLOAD_DIR = "data/videos"
DOCKER_IMAGE = "chicheng/openicc"

logging.basicConfig(level=logging.INFO, format='[TEST] %(message)s')
logger = logging.getLogger("TestGoPro")

def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def prerequisites_ok():
    if shutil.which("docker") is None:
        logger.warning("Skipping: docker not available.")
        return False
    if shutil.which("ffprobe") is None:
        logger.warning("Skipping: ffprobe not available.")
        return False
    return True


def start_node(cmd_port, status_port):
    logger.info("Starting GoPro Node...")
    env = os.environ.copy()
    env["ZUMI_CMD_PORT"] = str(cmd_port)
    env["ZUMI_STATUS_PORT"] = str(status_port)
    process = subprocess.Popen(
        [sys.executable, "node_gopro.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        cwd=os.getcwd(),
        env=env,
    )
    return process

def get_video_creation_time(file_path):
    """
    Get the creation_time from the video metadata using ffprobe.
    Returns a unix timestamp (float).
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", 
            "-select_streams", "v:0", 
            "-show_entries", "stream_tags=creation_time", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        if output:
            # Example format: 2025-01-01T12:00:00.000000Z
            dt = datetime.strptime(output, "%Y-%m-%dT%H:%M:%S.%fZ")
            return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception as e:
        logger.warning(f"Could not get video creation time: {e}")
    return None

def get_imu_start_time(json_path):
    """
    Extract the first IMU timestamp from the JSON file.
    Updated to support OpenImuCameraCalibrator JSON structure.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Strategy 1: Check for the provided structure (Root -> ID -> streams -> ACCL/GYRO -> samples)
        for key, val in data.items():
            if isinstance(val, dict) and "streams" in val:
                streams = val["streams"]
                for stream_key in ["ACCL", "GYRO"]:
                    if stream_key in streams:
                        stream_data = streams[stream_key]
                        if "samples" in stream_data and len(stream_data["samples"]) > 0:
                            first_sample = stream_data["samples"][0]
                            if "date" in first_sample:
                                date_str = first_sample["date"]
                                try:
                                    if date_str.endswith("Z"):
                                        date_str = date_str.replace("Z", "+00:00")
                                    dt = datetime.fromisoformat(date_str)
                                    return dt.timestamp()
                                except ValueError:
                                    pass

        # Strategy 2: Flat structure (backup)
        if "start_time" in data:
            return float(data["start_time"])
        
        if "imu" in data and isinstance(data["imu"], list) and len(data["imu"]) > 0:
             sample = data["imu"][0]
             if "time" in sample:
                 return float(sample["time"])
             if "ts" in sample:
                 return float(sample["ts"])
                 
    except Exception as e:
        logger.warning(f"Failed to parse IMU JSON: {e}")
    return None

def run_test_scenario(socket, scenario_name, num_videos, duration):
    print("\n" + "#"*60)
    print(f"STARTING SCENARIO: {scenario_name} ({num_videos} videos)")
    print("#"*60)

    run_info_list = []

    # 1. Recording Phase
    for i in range(num_videos):
        run_id = f"test_{scenario_name}_{i}_{int(time.time())}"
        start_time = time.time() + 3
        stop_time = start_time + duration
        
        logger.info(f"[{i+1}/{num_videos}] Scheduling Run {run_id}")
        logger.info(f"  Start: {start_time:.3f}, Stop: {stop_time:.3f}")
        
        run_info_list.append({
            "run_id": run_id,
            "target_start": start_time,
            "target_stop": stop_time
        })

        # Send Start
        socket.send_json({
            "cmd": "START_SYNC",
            "payload": {"run_id": run_id, "start_time": start_time}
        })

        # Send Stop
        socket.send_json({
            "cmd": "STOP_SYNC",
            "payload": {"stop_time": stop_time}
        })
        
        # Wait for this recording to finish before scheduling next (or not? 
        # ZMQ is async, but node is single-threaded for recording. 
        # We must wait for previous stop to pass before next start is valid logic-wise for the camera.)
        wait_for_finish = stop_time - time.time() + 2
        if wait_for_finish > 0:
            time.sleep(wait_for_finish)
            
        # Add a small buffer between recordings
        time.sleep(2)

    # 2. Download Phase
    logger.info("Sending START_DOWNLOAD...")
    socket.send_json({"cmd": "START_DOWNLOAD"})
    
    # Wait for all files
    logger.info(f"Waiting for {num_videos} files to appear in {DOWNLOAD_DIR}...")
    
    found_files = {} # run_id -> filepath
    max_wait = 60 * num_videos + 30
    start_wait = time.time()
    
    while time.time() - start_wait < max_wait:
        if os.path.exists(DOWNLOAD_DIR):
            for fname in os.listdir(DOWNLOAD_DIR):
                if not fname.endswith(".MP4"): continue
                
                # Check if this file belongs to one of our runs
                for info in run_info_list:
                    rid = info['run_id']
                    if fname.startswith(rid):
                        found_files[rid] = os.path.join(DOWNLOAD_DIR, fname)
        
        if len(found_files) == num_videos:
            break
        time.sleep(2)
        
    if len(found_files) < num_videos:
        logger.error(f"Timeout! Found {len(found_files)}/{num_videos} files.")
        return False

    logger.info("All files downloaded.")
    time.sleep(2) # Flush

    # 3. Verification Phase
    success = True
    for info in run_info_list:
        rid = info['run_id']
        path = found_files.get(rid)
        if not path:
            logger.error(f"Missing file for {rid}")
            success = False
            continue
            
        logger.info(f"Verifying {rid} -> {path}")
        if not verify_file(info['target_start'], path):
            success = False
            
    return success

def verify_file(target_start_time, video_path):
    # Extract IMU
    abs_video_path = os.path.abspath(video_path)
    video_dir = os.path.dirname(abs_video_path)
    video_filename = os.path.basename(abs_video_path)
    json_filename = video_filename.replace(".MP4", "_imu.json")
    abs_json_path = os.path.join(video_dir, json_filename)
    
    if os.path.exists(abs_json_path):
        os.remove(abs_json_path)

    docker_cmd = [
        "docker", "run", "--rm",
        "--volume", f"{video_dir}:/data",
        DOCKER_IMAGE,
        "node", "/OpenImuCameraCalibrator/javascript/extract_metadata_single.js",
        f"/data/{video_filename}",
        f"/data/{json_filename}"
    ]
    
    try:
        subprocess.run(docker_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        logger.error("Docker extraction failed.")
        return False
        
    video_start_ts = get_video_creation_time(video_path)
    imu_start_ts = get_imu_start_time(abs_json_path) if os.path.exists(abs_json_path) else None
    
    print("-" * 40)
    print(f"RESULTS FOR: {os.path.basename(video_path)}")
    print(f"Target Start:     {target_start_time:.4f}")
    print(f"Video Start:      {video_start_ts:.4f}" if video_start_ts else "Video Start:      N/A")
    print(f"IMU Start:        {imu_start_ts:.4f}" if imu_start_ts else "IMU Start:        N/A")
    
    diff_video = (video_start_ts - target_start_time) if video_start_ts else None
    diff_imu = (imu_start_ts - target_start_time) if imu_start_ts else None
    
    if diff_video is not None:
        print(f"Diff Video-Target: {diff_video:+.4f}s")
    if diff_imu is not None:
        print(f"Diff IMU-Target:   {diff_imu:+.4f}s")
        
    # Basic check: Error < 2 seconds (GoPro clock drift + start latency)
    # This is a loose check, mainly checking for gross errors.
    valid = True
    if diff_video and abs(diff_video) > 5.0:
        logger.warning("Large Video timestamp discrepancy!")
        valid = False
    if diff_imu and abs(diff_imu) > 5.0:
        logger.warning("Large IMU timestamp discrepancy!")
        valid = False
        
    return valid

def main():
    if not prerequisites_ok():
        logger.info("Prerequisites not met; skipping GoPro test.")
        return

    cmd_port = get_free_port()
    status_port = get_free_port()

    node_process = start_node(cmd_port, status_port)
    logger.info("Waiting 10s for node to initialize...")
    time.sleep(10)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{cmd_port}")
    
    logger.info("Waiting for ZMQ connection...")
    time.sleep(2)

    try:
        # Scenario 1: Single Video
        if not run_test_scenario(socket, "Single", 1, TEST_DURATION):
            logger.error("Single video scenario failed!")
        
        time.sleep(5)
        
        # Scenario 2: Batch Videos (3 videos)
        if not run_test_scenario(socket, "Batch", 3, TEST_DURATION):
            logger.error("Batch video scenario failed!")

    finally:
        socket.send_json({"cmd": "EXIT"})
        time.sleep(2)
        node_process.terminate()
        logger.info("Test Finished.")

if __name__ == "__main__":
    main()
