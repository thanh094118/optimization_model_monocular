from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np


OPENPOSE25_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]

# The GT files store a richer 27-joint schema. For OpenPose 25 evaluation we
# keep only the joints that have a clear semantic match on both sides.
EVAL_JOINT_MAP = [
    ("head", "Nose", "head"),
    ("neck", "Neck", "neck"),
    ("right_shoulder", "RShoulder", "right_shoulder"),
    ("right_elbow", "RElbow", "right_elbow"),
    ("right_wrist", "RWrist", "right_wrist"),
    ("left_shoulder", "LShoulder", "left_shoulder"),
    ("left_elbow", "LElbow", "left_elbow"),
    ("left_wrist", "LWrist", "left_wrist"),
    ("pelvis", "MidHip", "pelvis"),
    ("right_hip", "RHip", "right_hip"),
    ("right_knee", "RKnee", "right_knee"),
    ("right_ankle", "RAnkle", "right_ankle"),
    ("left_hip", "LHip", "left_hip"),
    ("left_knee", "LKnee", "left_knee"),
    ("left_ankle", "LAnkle", "left_ankle"),
    ("left_toe", "LBigToe", "left_toe"),
    ("left_foot", "LHeel", "left_foot"),
    ("right_toe", "RBigToe", "right_toe"),
    ("right_foot", "RHeel", "right_foot"),
]

PRIORITY1_NAMES = [
    "head",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
]

PRIORITY2_NAMES = PRIORITY1_NAMES + [
    "neck",
    "pelvis",
    "left_toe",
    "left_foot",
    "right_toe",
    "right_foot",
]


def _frame_index_from_path(path: Path) -> int:
    match = re.findall(r"\d+", path.stem)
    if not match:
        raise ValueError(f"Cannot extract frame index from {path}")
    return int(match[-1])


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_frame_map(frame_dir: Path) -> dict[int, Path]:
    frame_map: dict[int, Path] = {}
    for path in sorted(frame_dir.glob("*.json")):
        frame_idx = _frame_index_from_path(path)
        if frame_idx in frame_map:
            raise ValueError(f"Duplicate frame index {frame_idx} in {frame_dir}")
        frame_map[frame_idx] = path
    if not frame_map:
        raise ValueError(f"No JSON frames found in {frame_dir}")
    return frame_map


def _select_person(records: object, frame_path: Path, target_id: int = 0) -> dict:
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty list in {frame_path}")
    if len(records) == 1:
        record = records[0]
    else:
        record = next((item for item in records if isinstance(item, dict) and item.get("id") == target_id), records[0])
    if not isinstance(record, dict):
        raise ValueError(f"Invalid person record in {frame_path}")
    return record


def _normalize_vector(value: object, source_path: Path, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3D vector for {label} in {source_path}, got shape {arr.shape}")
    return arr


def _load_prediction_joints(frame_path: Path) -> dict[str, np.ndarray]:
    record = _select_person(_load_json(frame_path), frame_path)
    keypoints = record.get("keypoints3d")
    if keypoints is None:
        raise ValueError(f"Missing keypoints3d in {frame_path}")
    if len(keypoints) != len(OPENPOSE25_NAMES):
        raise ValueError(
            f"Expected {len(OPENPOSE25_NAMES)} OpenPose joints in {frame_path}, got {len(keypoints)}"
        )
    return {
        name: _normalize_vector(value, frame_path, name)
        for name, value in zip(OPENPOSE25_NAMES, keypoints)
    }


def _load_ground_truth_joints(frame_path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = _load_json(frame_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid ground-truth payload in {frame_path}")

    if any(str(key).startswith("camera") for key in raw.keys()):
        frame_payload = {key: value for key, value in raw.items() if str(key).startswith("camera")}
    elif len(raw) == 1:
        frame_payload = next(iter(raw.values()))
        if not isinstance(frame_payload, dict):
            raise ValueError(f"Invalid ground-truth payload in {frame_path}")
        frame_payload = {key: value for key, value in frame_payload.items() if str(key).startswith("camera")}
    else:
        raise ValueError(f"Expected camera* keys or a single wrapper entry in {frame_path}")

    gt_cameras: dict[str, dict[str, np.ndarray]] = {}
    for cam_key, joints in frame_payload.items():
        if not isinstance(joints, dict):
            raise ValueError(f"Invalid joint payload for {cam_key} in {frame_path}")
        gt_cameras[cam_key] = {
            name: _normalize_vector(value, frame_path, f"{cam_key}.{name}")
            for name, value in joints.items()
        }
    return gt_cameras


def _project_eval_joints(
    joints: dict[str, np.ndarray],
    source_path: Path,
    mapping: Iterable[tuple[str, str, str]],
    source_kind: str,
) -> dict[str, np.ndarray]:
    projected: dict[str, np.ndarray] = {}
    for out_name, pred_src_name, gt_src_name in mapping:
        src_name = pred_src_name if source_kind == "pred" else gt_src_name
        if src_name not in joints:
            raise ValueError(f"Missing source joint {src_name} in {source_path}")
        projected[out_name] = joints[src_name]
    return projected


def _root_from_hips(joints: dict[str, np.ndarray]) -> np.ndarray:
    if "left_hip" not in joints or "right_hip" not in joints:
        raise ValueError("Missing left_hip/right_hip for root alignment")
    return (joints["left_hip"] + joints["right_hip"]) / 2.0


def _compute_mpjpe(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> float:
    pred_root = _root_from_hips(pred)
    truth_root = _root_from_hips(truth)
    errors = [np.linalg.norm((pred[k] - pred_root) - (truth[k] - truth_root)) for k in keys]
    return float(np.mean(errors) * 1000.0)


def _compute_pck_mm(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> float:
    errors = [np.linalg.norm(pred[k] - truth[k]) for k in keys]
    return float(np.mean(errors) * 1000.0)


def _compute_pa_mpjpe(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> float:
    pred_mat = np.asarray([pred[k] for k in keys], dtype=float)
    truth_mat = np.asarray([truth[k] for k in keys], dtype=float)

    pred_mean = pred_mat.mean(axis=0)
    truth_mean = truth_mat.mean(axis=0)
    pred_centered = pred_mat - pred_mean
    truth_centered = truth_mat - truth_mean

    cov = pred_centered.T @ truth_centered
    u, s, vt = np.linalg.svd(cov, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T

    pred_var = np.sum(pred_centered**2)
    if pred_var < 1e-12:
        aligned = np.broadcast_to(truth_mean, truth_mat.shape)
    else:
        scale = float(np.sum(s) / pred_var)
        aligned = scale * (pred_centered @ r) + truth_mean

    return float(np.mean(np.linalg.norm(aligned - truth_mat, axis=1)) * 1000.0)


def _evaluate_series(camera_label: str, pred_dir: Path, gt_dir: Path, output_dir: Path) -> None:
    pred_frames = _load_frame_map(pred_dir)
    gt_frames = _load_frame_map(gt_dir)

    missing_in_gt = sorted(set(pred_frames) - set(gt_frames))
    if missing_in_gt:
        raise ValueError(
            f"Frame mismatch for {camera_label}. Missing GT frames for prediction indices: "
            f"{missing_in_gt[:20]}{'...' if len(missing_in_gt) > 20 else ''}"
        )
    extra_gt = sorted(set(gt_frames) - set(pred_frames))
    if extra_gt:
        print(
            f"[{camera_label}] GT has {len(extra_gt)} extra frames beyond prediction; "
            "they will be ignored."
        )

    out_files = {
        "MPJPE": output_dir / f"MPJPE_{camera_label}.csv",
        "PA-MPJPE": output_dir / f"PA-MPJPE_{camera_label}.csv",
        "PCK": output_dir / f"PCK_{camera_label}.csv",
    }

    rows = {
        "MPJPE": [],
        "PA-MPJPE": [],
        "PCK": [],
    }

    for frame_idx in sorted(pred_frames):
        pred_path = pred_frames[frame_idx]
        gt_path = gt_frames[frame_idx]

        pred_raw = _load_prediction_joints(pred_path)
        gt_raw = _load_ground_truth_joints(gt_path)
        gt_cam_key = "camera1" if camera_label == "cam1" else "camera2"
        if gt_cam_key not in gt_raw:
            raise ValueError(f"Missing {gt_cam_key} in {gt_path}")

        pred = _project_eval_joints(pred_raw, pred_path, EVAL_JOINT_MAP, "pred")
        truth = _project_eval_joints(gt_raw[gt_cam_key], gt_path, EVAL_JOINT_MAP, "gt")

        row = {"Frame": frame_idx}
        row["priority1_mm"] = {
            "MPJPE": _compute_mpjpe(pred, truth, PRIORITY1_NAMES),
            "PA-MPJPE": _compute_pa_mpjpe(pred, truth, PRIORITY1_NAMES),
            "PCK": _compute_pck_mm(pred, truth, PRIORITY1_NAMES),
        }
        row["priority2_mm"] = {
            "MPJPE": _compute_mpjpe(pred, truth, PRIORITY2_NAMES),
            "PA-MPJPE": _compute_pa_mpjpe(pred, truth, PRIORITY2_NAMES),
            "PCK": _compute_pck_mm(pred, truth, PRIORITY2_NAMES),
        }
        for metric in rows:
            rows[metric].append(row)

    for metric, out_path in out_files.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            f.write("sep=,\n")
            writer = csv.writer(f)
            writer.writerow(["Frame", "openpose25_priority1_mm", "openpose25_priority2_mm"])
            sum_p1 = 0.0
            sum_p2 = 0.0
            for row in rows[metric]:
                p1 = row["priority1_mm"][metric]
                p2 = row["priority2_mm"][metric]
                sum_p1 += p1
                sum_p2 += p2
                writer.writerow([row["Frame"], f"{p1:.2f}", f"{p2:.2f}"])
            n = len(rows[metric])
            writer.writerow(["AVERAGE", f"{(sum_p1 / n):.2f}", f"{(sum_p2 / n):.2f}"])
        print(f"[{camera_label}] wrote {metric} -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PCK/mm, MPJPE and PA-MPJPE for EasyMocap OpenPose 25 outputs."
    )
    parser.add_argument(
        "--camera1-dir",
        type=Path,
        default=None,
        help="Prediction directory for camera1. Omit to skip camera1.",
    )
    parser.add_argument(
        "--camera2-dir",
        type=Path,
        default=None,
        help="Prediction directory for camera2. Omit to skip camera2.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("input/gtruth_results"),
        help="Ground-truth directory with ground_truth_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/easymocap"),
        help="Directory where CSV files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ground_truth_dir.exists():
        raise FileNotFoundError(f"Missing ground-truth directory: {args.ground_truth_dir}")

    if args.camera1_dir is None and args.camera2_dir is None:
        args.camera1_dir = Path("input/easymocap/video_1")
        args.camera2_dir = Path("input/easymocap/video_2")

    if args.camera1_dir is not None:
        if not args.camera1_dir.exists():
            raise FileNotFoundError(f"Missing camera1 prediction directory: {args.camera1_dir}")
        _evaluate_series("cam1", args.camera1_dir, args.ground_truth_dir, args.output_dir)

    if args.camera2_dir is not None:
        if not args.camera2_dir.exists():
            raise FileNotFoundError(f"Missing camera2 prediction directory: {args.camera2_dir}")
        _evaluate_series("cam2", args.camera2_dir, args.ground_truth_dir, args.output_dir)

    print(f"Done. CSV files are in {args.output_dir}")


if __name__ == "__main__":
    main()
