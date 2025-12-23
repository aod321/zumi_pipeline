#!/usr/bin/env python3
# %%
import sys
import os
import json
import pathlib
import pickle
import collections
from typing import Dict, Tuple, Optional, List

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.cv_util import get_gripper_width
from umi.common.timecode_util import mp4_get_start_datetime


def _load_motor_data(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load motor timestamps and positions from .npz.

    Returns
    -------
    ts : (N,) float64 - Absolute timestamps (seconds).
    pos : (N,) float64 - Motor positions (radians).
    """
    data = np.load(path)
    arr = data["data"]
    columns = [str(c) for c in data["columns"]]
    col_idx: Dict[str, int] = {name: idx for idx, name in enumerate(columns)}
    if "ts" not in col_idx or "pos" not in col_idx:
        raise RuntimeError(f"{path} missing required columns ts/pos (found {columns})")
    return arr[:, col_idx["ts"]].astype(np.float64), arr[:, col_idx["pos"]].astype(np.float64)


def _resample(ts: np.ndarray, pos: np.ndarray, target_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample motor positions to target_fps using linear interpolation.
    Returns relative timestamps starting from 0.
    """
    rel_ts = ts - ts.min()
    if target_fps is None or target_fps <= 0:
        return rel_ts, pos
    dt = 1.0 / float(target_fps)
    t_new = np.arange(0.0, rel_ts[-1] + dt * 0.5, dt, dtype=np.float64)
    return t_new, np.interp(t_new, rel_ts, pos)


def _smooth_signal(x: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size <= 1:
        return x
    pad = window_size // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:-pad]


def _infer_open_close_and_direction(
    t_resampled: np.ndarray,
    pos_resampled: np.ndarray,
    initial_close_window: float = 0.5,
) -> Tuple[float, float, float, float, float, float]:
    """Infer open/close positions and sign of opening direction.

    Assumptions (matches our validation procedure):
    - Gripper starts fully closed and stays closed for the first ~0.5s.
    - Then it opens and closes repeatedly.

    Returns: (close_pos, open_pos, sign, pos_span, min_pos, max_pos)
    """
    if len(pos_resampled) == 0:
        raise RuntimeError("No samples to infer open/close positions.")

    min_pos, max_pos = float(np.min(pos_resampled)), float(np.max(pos_resampled))

    # 1) Closed position: median of the initial window (gripper is closed there)
    close_mask = (t_resampled - float(t_resampled.min())) <= float(initial_close_window)
    if not np.any(close_mask):
        raise RuntimeError("Unable to find samples in the initial closed window.")
    close_pos = float(np.median(pos_resampled[close_mask]))

    # 2) Open position: sample furthest away from the closed position
    idx_open = int(np.argmax(np.abs(pos_resampled - close_pos)))
    open_pos = float(pos_resampled[idx_open])

    pos_span = float(abs(open_pos - close_pos))
    sign = 1.0 if (pos_span <= 1e-6 or open_pos > close_pos) else -1.0

    return close_pos, open_pos, sign, pos_span, min_pos, max_pos


def _identify_gripper_from_tags(
    tag_detection_results: List[dict], tag_per_gripper: int = 6, det_thresh: float = 0.5
) -> Optional[int]:
    n_frames = len(tag_detection_results)
    if n_frames == 0:
        return None

    tag_counts = collections.defaultdict(lambda: 0)
    for frame in tag_detection_results:
        for key in frame["tag_dict"].keys():
            tag_counts[key] += 1
    tag_stats = {k: v / n_frames for k, v in tag_counts.items()}
    if len(tag_stats) == 0:
        return None

    max_tag_id = np.max(list(tag_stats.keys()))
    max_gripper_id = max_tag_id // tag_per_gripper

    gripper_prob_map = dict()
    for gripper_id in range(max_gripper_id + 1):
        left_id = gripper_id * tag_per_gripper
        right_id = left_id + 1
        left_prob = tag_stats.get(left_id, 0.0)
        right_prob = tag_stats.get(right_id, 0.0)
        gripper_prob = min(left_prob, right_prob)
        if gripper_prob <= 0:
            continue
        gripper_prob_map[gripper_id] = gripper_prob
    if len(gripper_prob_map) == 0:
        return None
    gripper_probs = sorted(gripper_prob_map.items(), key=lambda x: x[1])
    gripper_id, gripper_prob = gripper_probs[-1]
    if gripper_prob < det_thresh:
        return None
    return gripper_id


def _extract_width_series(
    tag_detection_results: List[dict],
    left_id: int,
    right_id: int,
    nominal_z: float,
    z_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    times = np.array([float(x["time"]) for x in tag_detection_results], dtype=np.float64)
    widths = []
    for td in tag_detection_results:
        w = get_gripper_width(td["tag_dict"], left_id=left_id, right_id=right_id, nominal_z=nominal_z, z_tolerance=z_tolerance)
        widths.append(np.nan if w is None else float(w))
    widths = np.asarray(widths, dtype=np.float64)
    det_ratio = float(np.isfinite(widths).mean()) if len(widths) > 0 else 0.0
    w_min = float(np.nanmin(widths)) if np.isfinite(widths).any() else float("nan")
    w_max = float(np.nanmax(widths)) if np.isfinite(widths).any() else float("nan")
    return times, widths, det_ratio, w_min, w_max


def _fill_and_normalize(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = values.copy()
    finite_mask = np.isfinite(x)
    if not finite_mask.any():
        return x, float("nan"), float("nan")
    idx = np.arange(len(x), dtype=np.float64)
    x[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], x[finite_mask])
    x = _smooth_signal(x, window_size=5)
    vmin = float(np.nanmin(x))
    vmax = float(np.nanmax(x))
    if vmax - vmin < 1e-9:
        norm = np.zeros_like(x)
    else:
        norm = (x - vmin) / (vmax - vmin)
    return norm, vmin, vmax


def _find_edges(t: np.ndarray, signal: np.ndarray, grad_threshold: float = 0.1, min_separation: float = 0.2) -> List[float]:
    """Detect rising edges using gradient threshold and minimum time separation."""
    grad = np.gradient(signal, t)
    edge_times: List[float] = []
    last_t = -1e9
    for ti, gi in zip(t, grad):
        if gi > grad_threshold and (ti - last_t) >= min_separation:
            edge_times.append(float(ti))
            last_t = ti
    return edge_times


def _estimate_offset_and_drift(
    motor_ts_abs: np.ndarray,
    motor_pos: np.ndarray,
    video_ts: np.ndarray,
    video_width: np.ndarray,
    video_start_ts_abs: float,
) -> Tuple[Optional[float], float, dict]:
    """Return (offset_ros_minus_gopro, drift_ppm, debug_info).

    offset_ros_minus_gopro is the clock delta such that:
        ros_time = video_time_abs + offset + drift * (video_time_rel)
    where video_time_abs is in GoPro clock (from mp4 EXIF) and video_time_rel
    is relative to the first frame timestamp in tag_detection.
    """
    debug = {}
    # 1) Normalize motor signal to "open is high"
    motor_rel_ts = motor_ts_abs - motor_ts_abs[0]
    close_pos, open_pos, sign, pos_span, _, _ = _infer_open_close_and_direction(motor_rel_ts, motor_pos, initial_close_window=0.5)
    motor_width = sign * (motor_pos - close_pos)
    motor_norm, motor_min, motor_max = _fill_and_normalize(motor_width)

    # 2) Normalize video width
    video_norm, video_min, video_max = _fill_and_normalize(video_width)

    # resample to common timeline for cross-corr (relative timelines)
    # choose dt from video median frame interval
    if len(video_ts) < 2:
        return None, 0.0, debug
    dt = float(np.median(np.diff(video_ts)))
    if dt <= 0:
        return None, 0.0, debug
    t_end = min(video_ts[-1], motor_rel_ts[-1])
    if t_end <= 0:
        return None, 0.0, debug
    t_uniform = np.arange(0.0, t_end, dt, dtype=np.float64)
    motor_uniform = np.interp(t_uniform, motor_rel_ts, motor_norm)
    video_uniform = np.interp(t_uniform, video_ts, video_norm)

    a = motor_uniform - motor_uniform.mean()
    b = video_uniform - video_uniform.mean()
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    best_idx = int(np.argmax(corr))
    corr_lag = lags[best_idx] * dt
    corr_peak = float(corr[best_idx]) if corr.size > 0 else 0.0
    # diff_rel = motor_edge_time - video_edge_time (relative)
    diff_rel_corr = -float(corr_lag)
    debug["corr_lag_sec"] = float(corr_lag)
    debug["corr_peak"] = corr_peak

    # 3) Edge-based refinement
    motor_edges = _find_edges(t_uniform, motor_uniform, grad_threshold=0.2, min_separation=0.2)
    video_edges = _find_edges(t_uniform, video_uniform, grad_threshold=0.2, min_separation=0.2)
    debug["n_edges_motor"] = len(motor_edges)
    debug["n_edges_video"] = len(video_edges)

    if len(motor_edges) >= 1 and len(video_edges) >= 1:
        diffs_flat = []
        for ve in video_edges:
            nearest = min([(abs(me - ve), me - ve) for me in motor_edges], key=lambda x: x[0])
            diffs_flat.append(nearest[1])
        offset_rel = float(np.median(diffs_flat))  # motor_rel - video_rel
        # optional drift using linear fit if >1 edge pair
        drift_ppm = 0.0
        if len(video_edges) >= 2 and len(motor_edges) >= 2:
            # pair by nearest neighbor
            pairs = []
            for ve in video_edges:
                me = min(motor_edges, key=lambda x: abs(x - ve))
                pairs.append((ve, me))
            v = np.array([p[0] for p in pairs], dtype=np.float64)
            m = np.array([p[1] for p in pairs], dtype=np.float64)
            A = np.stack([np.ones_like(v), v], axis=1)
            sol, _, _, _ = np.linalg.lstsq(A, m, rcond=None)
            a0, a1 = sol  # m ~= a0 + a1*v
            offset_rel = float(a0)
            drift_ppm = float((a1 - 1.0) * 1e6)
        debug["method"] = "edges"
        start_delta = float(motor_ts_abs[0] - video_start_ts_abs)
        offset_abs = start_delta + offset_rel
        debug["offset_rel"] = offset_rel
        debug["start_delta"] = start_delta
        return offset_abs, drift_ppm, debug

    # fallback to cross-correlation lag
    start_delta = float(motor_ts_abs[0] - video_start_ts_abs)
    offset_abs = start_delta + diff_rel_corr
    debug["method"] = "xcorr"
    debug["offset_rel"] = diff_rel_corr
    debug["start_delta"] = start_delta
    return offset_abs, 0.0, debug


@click.command()
@click.option("-m", "--motor", "motor_path", required=True, type=click.Path(exists=True, dir_okay=False), help="Motor data .npz path.")
@click.option("-md", "--meta", "meta_path", default=None, type=click.Path(exists=True, dir_okay=False), help="Motor meta data json path.")
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False), help="Output gripper_range.json path.")
@click.option("-g", "--gripper-id", default=0, show_default=True, type=int, help="Gripper hardware id.")
@click.option("--target-fps", default=0.0, show_default=True, type=float, help="Resample motor data to this FPS. Use 0 to keep original rate.")
@click.option("-cs", "--cam-serial", default=None, type=str, help="Camera serial number associated with this gripper.")
@click.option("-td", "--tag-detection", default=None, type=click.Path(exists=True, dir_okay=False), help="Tag detection pkl for time sync.")
@click.option("--nominal-z", default=0.03, show_default=True, type=float, help="Nominal Z (m) for finger tags.")
@click.option("--z-tolerance", default=0.01, show_default=True, type=float, help="Z tolerance (m) for finger tag depth filter.")
@click.option("-v", "--video", "video_path", default=None, type=click.Path(exists=True, dir_okay=False), help="Raw video mp4 path for this calibration (to get absolute start time).")
def main(motor_path, meta_path, output, gripper_id, target_fps, cam_serial, tag_detection, nominal_z, z_tolerance, video_path):
    motor_path = pathlib.Path(motor_path)
    meta_path = pathlib.Path(meta_path) if meta_path is not None else None
    if meta_path is None:
        candidate = motor_path.parent / "motor_meta_data.json"
        if candidate.is_file():
            meta_path = candidate

    ts, pos = _load_motor_data(motor_path)
    if len(ts) == 0:
        raise RuntimeError("Motor data is empty.")

    t_resampled, pos_resampled = _resample(ts, pos, target_fps)
    if len(pos_resampled) == 0:
        raise RuntimeError("No valid samples from motor data.")

    close_pos, open_pos, sign, pos_span, min_pos, max_pos = _infer_open_close_and_direction(t_resampled, pos_resampled)

    # Run-level linear mapping: width = sign * (pos - close_pos)
    scale, offset = float(sign), float(-sign * close_pos)

    meta = {}
    if meta_path is not None and meta_path.is_file():
        meta = json.load(open(meta_path, "r"))

    result = {
        "gripper_id": gripper_id,
        "cam_serial": cam_serial,
        "min_width": 0.0,
        "max_width": float(pos_span),
        "calibration_method": "motor_range_direction_only",
        "min_position": float(min_pos),
        "max_position": float(max_pos),
        "open_position": float(open_pos),
        "close_position": float(close_pos),
        "pos_span": float(pos_span),
        "position_to_width": {"scale": float(scale), "offset": float(offset), "units": "width_equals_scale_times_pos_plus_offset"},
        "stats": {
            "n_samples_raw": int(len(ts)),
            "n_samples_used": int(len(pos_resampled)),
            "n_seconds": float(ts.max() - ts.min()),
            "ts_start": float(ts.min()),
            "ts_end": float(ts.max()),
            "effective_freq_hz": float(len(ts) / (ts.max() - ts.min())) if len(ts) > 1 else 0.0,
            "resample_fps": float(target_fps),
        },
    }
    if meta:
        result["motor_meta_data"] = meta
    # Optional: compute time sync with tag detection
    if tag_detection is not None:
        td_path = pathlib.Path(tag_detection)
        tag_data = pickle.load(open(td_path, "rb"))
        gripper_id_auto = _identify_gripper_from_tags(tag_data)
        if gripper_id_auto is not None:
            gripper_id = gripper_id_auto
        left_id = gripper_id * 6
        right_id = left_id + 1

        times, widths, det_ratio, w_min, w_max = _extract_width_series(
            tag_detection_results=tag_data,
            left_id=left_id,
            right_id=right_id,
            nominal_z=nominal_z,
            z_tolerance=z_tolerance,
        )
        # Fallback scan if nothing detected
        if not np.isfinite(widths).any():
            candidate_z = [nominal_z, 0.025, 0.03, 0.035, 0.04]
            tried = []
            for z in candidate_z:
                t_tmp, w_tmp, r_tmp, wmin_tmp, wmax_tmp = _extract_width_series(
                    tag_detection_results=tag_data,
                    left_id=left_id,
                    right_id=right_id,
                    nominal_z=z,
                    z_tolerance=z_tolerance,
                )
                tried.append((r_tmp, z, t_tmp, w_tmp, wmin_tmp, wmax_tmp))
            tried.sort(key=lambda x: x[0], reverse=True)
            best = tried[0]
            det_ratio, nominal_z, times, widths, w_min, w_max = best[0], best[1], best[2], best[3], best[4], best[5]

        # only proceed if we have some finite widths
        if np.isfinite(widths).any():
            # video start time required to produce absolute offset
            video_start_ts_abs = None
            if video_path is not None:
                video_start_ts_abs = mp4_get_start_datetime(video_path).timestamp()
            else:
                # try to infer from tag_detection location
                candidate_mp4 = td_path.parent.joinpath("raw_video.mp4")
                if candidate_mp4.is_file():
                    video_start_ts_abs = mp4_get_start_datetime(str(candidate_mp4)).timestamp()
            if video_start_ts_abs is None:
                print("Warning: Unable to determine video start time; skipping time sync.")
            else:
                offset, drift_ppm, dbg = _estimate_offset_and_drift(
                    motor_ts_abs=ts,
                    motor_pos=pos,
                    video_ts=times,
                    video_width=widths,
                    video_start_ts_abs=video_start_ts_abs,
                )
                if offset is not None:
                    result["time_sync"] = {
                        "offset_ros_minus_gopro": float(offset),
                        "drift_ppm": float(drift_ppm),
                        "method": dbg.get("method", "unknown"),
                        "n_edges_motor": int(dbg.get("n_edges_motor", 0)),
                        "n_edges_video": int(dbg.get("n_edges_video", 0)),
                        "corr_peak": float(dbg.get("corr_peak", 0.0)),
                        "corr_lag_sec": float(dbg.get("corr_lag_sec", 0.0)),
                        "offset_rel": float(dbg.get("offset_rel", 0.0)),
                        "start_delta": float(dbg.get("start_delta", 0.0)),
                        "nominal_z": float(nominal_z),
                        "z_tolerance": float(z_tolerance),
                        "tag_width_min": float(w_min),
                        "tag_width_max": float(w_max),
                        "tag_det_ratio": float(det_ratio),
                        "video_start_timestamp": float(video_start_ts_abs),
                    }
                else:
                    print("Warning: Unable to estimate time sync (offset).")

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(output_path, "w"), indent=2)
    print(f"Saved gripper range to {output_path}")


if __name__ == "__main__":
    main()
