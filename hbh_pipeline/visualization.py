from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from preprocess_pipeline.calib import load_camera_profile

SKELETON = [
    ("right_shoulder", "left_shoulder"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_hip", "left_hip"),
    ("neck", "right_shoulder"),
    ("neck", "left_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("left_shoulder", "left_elbow"),
    ("right_elbow", "right_hand"),
    ("left_elbow", "left_hand"),
    ("right_hip", "right_knee"),
    ("left_hip", "left_knee"),
    ("right_knee", "right_ankle"),
    ("left_knee", "left_ankle"),
    ("right_ankle", "right_foot"),
    ("left_ankle", "left_foot"),
]


def _frame_index(path: Path) -> int:
    m = re.findall(r"\d+", path.stem)
    return int(m[-1]) if m else -1


def _build_projection_matrix(cam_profile: dict) -> np.ndarray:
    intr = np.asarray(cam_profile.get("intrinsics_cam", []), dtype=float)
    if intr.shape != (3, 3):
        raise ValueError("Invalid intrinsics_cam in data_cam1.json")
    R = np.asarray(cam_profile.get("extrinsic_cam", []), dtype=float)
    if R.shape != (3, 3):
        raise ValueError("Invalid extrinsic_cam in data_cam1.json")
    t = np.asarray(cam_profile.get("xyz", []), dtype=float).reshape(-1)
    if t.size != 3:
        raise ValueError("Invalid xyz in data_cam1.json")

    # Keep parity with HBH triangulation camera model in hbh_pipeline.execute.build_projection_matrix.
    R_use = R.copy()
    R_use[1:, :] *= -1.0
    Rt = np.hstack([R_use, (-R_use @ t.reshape(3, 1))])
    return intr @ Rt


def _project_point(xyz, P: np.ndarray):
    arr = np.asarray(xyz, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    if not np.isfinite(arr[:3]).all():
        return None
    h = P @ np.array([arr[0], arr[1], arr[2], 1.0], dtype=float)
    if not np.isfinite(h).all() or abs(h[2]) <= 1e-9:
        return None
    return int(round(h[0] / h[2])), int(round(h[1] / h[2]))


def _draw_overlay(img_bgr: np.ndarray, joints: dict, P: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    proj = {}
    for name, xyz in joints.items():
        p = _project_point(xyz, P)
        if p is not None:
            proj[name] = p
    for a, b in SKELETON:
        if a in proj and b in proj:
            cv2.line(out, proj[a], proj[b], (200, 200, 200), 2, cv2.LINE_AA)
    for p in proj.values():
        cv2.circle(out, p, 3, (0, 255, 255), -1, cv2.LINE_AA)
    return out


def _load_method_frame_map(method_dir: Path) -> dict[int, dict]:
    out = {}
    for p in sorted(method_dir.glob("hbh_data_*.json"), key=_frame_index):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        kp = data.get("keypoints3d", {}).get("camera1", {})
        if isinstance(kp, dict):
            out[_frame_index(p)] = kp
    return out


def _load_pa_map(csv_path: Path) -> dict[int, dict[str, float]]:
    out = {}
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() in {"sep=,", "frame", "average"}:
                continue
            try:
                frame_id = int(row[0].strip())
            except ValueError:
                continue
            out[frame_id] = {
                "all": float(row[1]) if len(row) > 1 and row[1].strip() else float("nan"),
                "arm_leg": float(row[2]) if len(row) > 2 and row[2].strip() else float("nan"),
                "reliable": float(row[3]) if len(row) > 3 and row[3].strip() else float("nan"),
            }
    return out


def run_hbh_visualization(config: dict) -> None:
    hbh_cfg = config.get("hbh", {})
    if not hbh_cfg.get("enabled", False):
        return

    paths = config.get("paths", {})
    output_root = Path(paths.get("hbh_output_dir", "output/hbh_results"))
    method_dirs = sorted([p for p in output_root.glob("method_*") if p.is_dir()])[:5]
    if not method_dirs:
        print(f"[HBH-Vis] No method folders under {output_root}")
        return

    video1 = Path(paths["camera1_video"])
    cap = cv2.VideoCapture(str(video1))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for HBH visualization: {video1}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    cam_profile = load_camera_profile(config, "cam1")
    P = _build_projection_matrix(cam_profile)

    method_names = [d.name.replace("method_", "") for d in method_dirs]
    method_kp_maps = {m: _load_method_frame_map(d) for m, d in zip(method_names, method_dirs)}
    method_pa_maps = {m: _load_pa_map(output_root / f"pa_mpjpe_hbh_{m}.csv") for m in method_names}

    common_ids = None
    for m in method_names:
        ids = set(method_kp_maps[m].keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted([i for i in (common_ids or set()) if i >= 1 and i <= n_frames])
    if not common_ids:
        print("[HBH-Vis] No common frames across selected methods")
        cap.release()
        return

    ret, first = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame for HBH visualization")
    h, w = first.shape[:2]

    # Avoid codec limits for very large frame sizes (5-column layout can exceed MPEG-4 constraints).
    max_canvas_w = int(hbh_cfg.get("vis_max_width", 3840))
    max_canvas_h = int(hbh_cfg.get("vis_max_height", 2160))
    cols = len(method_names)
    canvas_w, canvas_h = w * cols, h
    scale = min(max_canvas_w / float(canvas_w), max_canvas_h / float(canvas_h), 1.0)
    out_w = max(2, int(round(canvas_w * scale)))
    out_h = max(2, int(round(canvas_h * scale)))
    if out_w % 2 == 1:
        out_w -= 1
    if out_h % 2 == 1:
        out_h -= 1

    out_path = output_root / "hbh_project_5col.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open VideoWriter for {out_path} with size {(out_w, out_h)}. "
            f"Try lowering hbh.vis_max_width/vis_max_height."
        )

    for frame_id in common_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        cols = []
        for method in method_names:
            overlay = _draw_overlay(frame, method_kp_maps[method][frame_id], P)
            metrics = method_pa_maps.get(method, {}).get(frame_id, {})
            t1 = f"{method} | Frame {frame_id}"
            t2 = f"PA: {metrics.get('all', float('nan')):.2f} | AL: {metrics.get('arm_leg', float('nan')):.2f} | RH: {metrics.get('reliable', float('nan')):.2f}"
            cv2.putText(overlay, t1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, t2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cols.append(overlay)
        canvas = np.hstack(cols)
        if (out_w, out_h) != (canvas.shape[1], canvas.shape[0]):
            canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(canvas)

    writer.release()
    cap.release()
    print(f"[HBH-Vis] Saved: {out_path} | size={out_w}x{out_h} | scale={scale:.4f}")
