import sys
import os
import json
import numpy as np
import subprocess
import logging
import shlex
from datetime import datetime, timezone
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
VIDEO_DIR = Path("data/videos")
DOCKER_IMAGE = "chicheng/openicc"

logging.basicConfig(level=logging.INFO, format='[VALIDATOR] %(message)s')
logger = logging.getLogger("Validator")

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
            str(file_path)
        ]
        output = subprocess.check_output(cmd).decode().strip()
        if output:
            dt = datetime.strptime(output, "%Y-%m-%dT%H:%M:%S.%fZ")
            return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception as e:
        logger.warning(f"Could not get video creation time: {e}")
    return None

def extract_imu(video_path, json_path):
    video_path = Path(video_path).resolve()
    json_path = Path(json_path).resolve()
    
    # Logic: map the user's home directory to /data in docker
    # This preserves the relative path structure inside the container
    home_dir = Path.home()
    
    try:
        if video_path.is_relative_to(home_dir):
            mount_source = home_dir
            rel_video_path = video_path.relative_to(home_dir)
            
            # Docker paths
            docker_video_path = Path("/data") / rel_video_path
            
            # For JSON, we want it explicitly where requested, assuming it's also under home
            if json_path.is_relative_to(home_dir):
                 rel_json_path = json_path.relative_to(home_dir)
                 docker_json_path = Path("/data") / rel_json_path
            else:
                 # Fallback if JSON path is outside home (unlikely but safe)
                 docker_json_path = Path("/data") / json_path.name
                 
        else:
            # Fallback: Mount the video's parent directory directly
            mount_source = video_path.parent
            docker_video_path = Path("/data") / video_path.name
            docker_json_path = Path("/data") / json_path.name

        docker_cmd = [
            "docker", "run", "--rm",
            "--volume", f"{mount_source}:/data",
            DOCKER_IMAGE,
            "node", "/OpenImuCameraCalibrator/javascript/extract_metadata_single.js",
            str(docker_video_path),
            str(docker_json_path)
        ]
        
        subprocess.run(docker_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        logger.error("Docker extraction failed.")
        return False
    except FileNotFoundError:
        logger.error("Docker not found.")
        return False
    except Exception as e:
        logger.error(f"Error preparing docker command: {e}")
        return False

def get_imu_start_time(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Strategy 1: OpenImuCameraCalibrator structure
        for key, val in data.items():
            if isinstance(val, dict) and "streams" in val:
                streams = val["streams"]
                if "ACCL" in streams and "samples" in streams["ACCL"]:
                     samples = streams["ACCL"]["samples"]
                     if samples:
                         date_str = samples[0].get("date")
                         if date_str:
                             if date_str.endswith("Z"): date_str = date_str.replace("Z", "+00:00")
                             return datetime.fromisoformat(date_str).timestamp()
                             
        # Strategy 2: Flat structure
        if "start_time" in data:
            return float(data["start_time"])
            
    except Exception as e:
        logger.warning(f"Failed to parse IMU JSON: {e}")
    return None

def check_video_decoding(video_path):
    cmd = [
        "ffmpeg", "-v", "error", 
        "-i", str(video_path), 
        "-t", "5", 
        "-an", 
        "-f", "null", "-"
    ]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg decoding failed: {e.stderr.decode()}")
        return False

def normalize_run_files(run_id, video_path):
    """
    Renames generic 'run_id_motor.npz' to 'run_id_GX..._motor.npz' if video exists.
    Returns the resolved motor path.
    """
    if not video_path:
        # Just look for generic files if no video
        generic_npz = DATA_DIR / f"{run_id}_motor.npz"
        generic_json = DATA_DIR / f"{run_id}_motor.json"
        if generic_npz.exists(): return generic_npz
        if generic_json.exists(): return generic_json
        
        # Look for already renamed files
        existing = list(DATA_DIR.glob(f"{run_id}_*_motor.npz"))
        if existing: return existing[0]
        return None

    # We have video, verify name pattern
    video_name = video_path.name # run_id_GX...MP4
    prefix = f"{run_id}_"
    
    if not video_name.startswith(prefix):
        return None # Should not happen based on glob pattern

    gopro_tag = video_name[len(prefix):].replace(".MP4", "") # e.g. GX011234
    
    target_motor_npz = DATA_DIR / f"{run_id}_{gopro_tag}_motor.npz"
    target_motor_json = DATA_DIR / f"{run_id}_{gopro_tag}_motor.json"
    
    generic_motor_npz = DATA_DIR / f"{run_id}_motor.npz"
    generic_motor_json = DATA_DIR / f"{run_id}_motor.json"
    
    # Rename NPZ
    if generic_motor_npz.exists() and not target_motor_npz.exists():
        try:
            generic_motor_npz.rename(target_motor_npz)
            print(f"[INFO] Renamed generic motor file to: {target_motor_npz.name}")
        except Exception as e:
            print(f"[WARN] Failed to rename motor file: {e}")
            return generic_motor_npz
            
    # Rename JSON (Best effort)
    if generic_motor_json.exists() and not target_motor_json.exists():
        try:
            generic_motor_json.rename(target_motor_json)
        except Exception: pass

    if target_motor_npz.exists():
        return target_motor_npz
    elif generic_motor_npz.exists():
        return generic_motor_npz
    
    # Fallback: check if it was already named correctly previously
    renamed_existing = list(DATA_DIR.glob(f"{run_id}_{gopro_tag}_motor.npz"))
    if renamed_existing:
        return renamed_existing[0]
        
    return None

def validate(run_id):
    print(f"\n=== Validating Run: {run_id} ===")
    
    # 1. Locate Video first (needed for normalization naming)
    video_pattern = f"{run_id}_*.MP4"
    video_files = list(VIDEO_DIR.glob(video_pattern))
    
    video_path = None
    if video_files:
        video_path = video_files[0]
        if video_path.stat().st_size < 1024:
             print(f"[FAIL] Video file too small ({video_path.stat().st_size} bytes)")
             return False
        print(f"[PASS] Video file found: {video_path}")
    else:
        print("[WARN] GoPro video missing (might be queued for download).")
    
    # 2. Normalize and Locate Motor Data
    motor_path = normalize_run_files(run_id, video_path)
    
    if not motor_path or not motor_path.exists():
        print("[FAIL] Motor data missing.")
        return False
        
    if motor_path.stat().st_size < 1024:
        print(f"[FAIL] Motor data too small ({motor_path.stat().st_size} bytes)")
        return False
        
    print(f"[PASS] Motor data found: {motor_path}")

    # 3. Validity Check (Motor Content)
    motor_start_ts = None
    try:
        if motor_path.suffix == '.npz':
            with np.load(motor_path) as data:
                arr = data['data']
                if len(arr) == 0:
                     print("[FAIL] Motor data empty.")
                     return False
                motor_start_ts = arr[0, 0]
                motor_end_ts = arr[-1, 0]
                count = len(arr)
        else:
            with open(motor_path, 'r') as f:
                data = json.load(f)
                if not data:
                    print("[FAIL] Motor data empty.")
                    return False
                motor_start_ts = data[0][0]
                motor_end_ts = data[-1][0]
                count = len(data)

        duration = motor_end_ts - motor_start_ts
        freq = count / duration if duration > 0 else 0
        print(f"[INFO] Motor: {freq:.1f}Hz, {duration:.2f}s")
        
    except Exception as e:
        print(f"[FAIL] Motor data invalid: {e}")
        return False

    # 4. Video & Sync Checks
    if video_path:
        if not check_video_decoding(video_path):
             print("[FAIL] Video corrupted (ffmpeg check failed).")
             return False
        print("[PASS] Video integrity verified (ffmpeg).")
        
        # Check IMU Extraction
        imu_json_path = video_path.with_name(video_path.name.replace(".MP4", "_imu.json"))
        if not imu_json_path.exists():
            print("[INFO] Extracting IMU data...")
            if not extract_imu(video_path, imu_json_path):
                print("[FAIL] IMU extraction failed.")
                return False
        
        imu_start_ts = get_imu_start_time(imu_json_path)
        if not imu_start_ts:
             print("[FAIL] IMU data invalid.")
             return False
        print(f"[PASS] IMU data extracted: {imu_json_path.name}")
        
        # Synchronization Check
        video_start_ts = get_video_creation_time(video_path)
        
        print(f"\nTimestamps:")
        print(f"  Motor Start: {motor_start_ts:.4f}")
        if video_start_ts: 
            print(f"  Video Start: {video_start_ts:.4f} (Diff: {video_start_ts - motor_start_ts:+.4f}s)")
        print(f"  IMU Start:   {imu_start_ts:.4f}   (Diff: {imu_start_ts - motor_start_ts:+.4f}s)")
        
        diff = abs(imu_start_ts - motor_start_ts)
        if diff > 1.0: 
            print(f"[WARN] Large sync offset between Motor and IMU ({diff:.3f}s > 1.0s)")
        else:
            print(f"[PASS] Synchronization within tolerance ({diff:.3f}s <= 1.0s).")

    print(f"\n=== Run {run_id}: VALIDATED ===")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate(sys.argv[1])
    else:
        print("Usage: python validator.py <run_id>")
