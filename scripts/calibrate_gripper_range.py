# %%
import sys
import os
import json
import pathlib
from typing import Dict, Tuple

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def _load_motor_data(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    arr = data["data"]
    columns = [str(c) for c in data["columns"]]
    col_idx: Dict[str, int] = {name: idx for idx, name in enumerate(columns)}
    if "ts" not in col_idx or "pos" not in col_idx:
        raise RuntimeError(f"{path} missing required columns ts/pos (found {columns})")
    ts = arr[:, col_idx["ts"]]
    pos = arr[:, col_idx["pos"]]
    return ts, pos


def _resample(ts: np.ndarray, pos: np.ndarray, target_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample motor positions to target_fps using linear interpolation.
    If target_fps <= 0, return original timestamps/positions (shifted to start at 0).
    """
    rel_ts = ts - ts.min()
    if target_fps is None or target_fps <= 0:
        return rel_ts, pos

    dt = 1.0 / float(target_fps)
    # include endpoint
    t_new = np.arange(0.0, rel_ts[-1] + dt * 0.5, dt, dtype=np.float64)
    pos_new = np.interp(t_new, rel_ts, pos)
    return t_new, pos_new


@click.command()
@click.option(
    "-m",
    "--motor",
    "motor_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Motor data .npz path (ts,pos,vel,tau,iter).",
)
@click.option(
    "-md",
    "--meta",
    "meta_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Motor meta data json path. Defaults to motor_dir/motor_meta_data.json if present.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output gripper_range.json path.",
)
@click.option(
    "-g",
    "--gripper-id",
    default=0,
    show_default=True,
    type=int,
    help="Gripper hardware id (relative width only; set -1 if unknown).",
)
@click.option(
    "--target-fps",
    default=0.0,
    show_default=True,
    type=float,
    help="Resample motor data to this FPS before computing range. Use 0 to keep original rate.",
)
@click.option(
    "-cs",
    "--cam-serial",
    default=None,
    type=str,
    help="Camera serial number associated with this gripper.",
)
def main(motor_path, meta_path, output, gripper_id, target_fps, cam_serial):
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

    min_pos = float(np.min(pos_resampled))
    max_pos = float(np.max(pos_resampled))
    pos_span = float(max_pos - min_pos)

    # Auto-detect direction based on recording convention:
    # Recording starts with gripper open, so pos[0] is the open position.
    first_pos = float(pos_resampled[0])
    if abs(first_pos - min_pos) < abs(first_pos - max_pos):
        # Open position is near min_pos
        open_pos, close_pos = min_pos, max_pos
    else:
        # Open position is near max_pos
        open_pos, close_pos = max_pos, min_pos

    # Mapping: width = 0 when closed, width = pos_span when open
    # width = scale * pos + offset
    if close_pos == min_pos:
        # Closing means position decreases
        scale = 1.0
        offset = -min_pos
    else:
        # Closing means position increases
        scale = -1.0
        offset = max_pos

    meta = {}
    if meta_path is not None and meta_path.is_file():
        meta = json.load(open(meta_path, "r"))

    result = {
        "gripper_id": gripper_id,
        "cam_serial": cam_serial,
        "min_width": 0.0,
        "max_width": pos_span,
        "calibration_method": "motor_range_auto_direction",
        "min_position": min_pos,
        "max_position": max_pos,
        "open_position": open_pos,
        "close_position": close_pos,
        "pos_span": pos_span,
        "position_to_width": {
            "scale": scale,
            "offset": offset,
            "units": "width_equals_scale_times_pos_plus_offset",
        },
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

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(output_path, "w"), indent=2)
    print(f"Saved gripper range to {output_path}")


if __name__ == "__main__":
    main()
