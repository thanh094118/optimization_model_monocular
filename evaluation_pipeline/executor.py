from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import procrustes

from keypoints_map import load_keypoints3d_map
from evaluation_pipeline.logs import log_disabled, log_done

MODULES = ["posed", "fused", "learnable", "optimized"]
CAMERAS = ["camera1", "camera2"]
CAMERA_FILE_NAMES = {
    "camera1": "cam1",
    "camera2": "cam2",
}


def _camera_key_from_video_path(video_path: Optional[str]) -> str:
    if not video_path:
        return "camera1"
    stem = Path(video_path).stem
    match = re.search(r"video_(\d+)", stem)
    if match:
        return f"camera{int(match.group(1))}"
    match = re.search(r"camera_(\d+)", stem)
    if match:
        return f"camera{int(match.group(1))}"
    nums = re.findall(r"\d+", stem)
    if nums:
        return f"camera{int(nums[-1])}"
    return "camera1"

def _frame_index_from_path(p: Path) -> int:
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else -1

def _resolve_root_joint(joints: dict) -> np.ndarray:
    if "left_hip" in joints and "right_hip" in joints:
        return (joints["left_hip"] + joints["right_hip"]) / 2.0
    raise ValueError("Missing left_hip or right_hip for root alignment")

def _compute_mpjpe(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> float:
    pred_root = _resolve_root_joint(pred)
    truth_root = _resolve_root_joint(truth)
    
    errors = []
    for k in keys:
        p = pred[k] - pred_root
        t = truth[k] - truth_root
        errors.append(np.linalg.norm(p - t) * 1000.0)
    return float(np.mean(errors))

def _compute_pa_mpjpe(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> float:
    pred_mat = np.array([pred[k] for k in keys], dtype=float)
    truth_mat = np.array([truth[k] for k in keys], dtype=float)
    
    truth_centered = truth_mat - np.mean(truth_mat, axis=0)
    norm_truth = np.linalg.norm(truth_centered)
    if norm_truth < 1e-6:
        return 0.0
        
    mtx1, mtx2, _ = procrustes(truth_mat, pred_mat)
    distances = np.linalg.norm(mtx1 * norm_truth - mtx2 * norm_truth, axis=1) * 1000.0
    return float(np.mean(distances))

def _compute_pck_mm(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> float:
    # PCK-mm is mean Euclidean distance (absolute, no root align)
    errors = []
    for k in keys:
        p = pred[k]
        t = truth[k]
        errors.append(np.linalg.norm(p - t) * 1000.0)
    return float(np.mean(errors))

def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_frame_map(frame_dir: Path, frame_offset: int = 0) -> dict[int, Path]:
    frame_map = {}
    for path in frame_dir.glob("*.json"):
        frame_idx = _frame_index_from_path(path)
        if frame_idx < 0:
            raise ValueError(f"Cannot extract frame index from {path}")
        frame_idx += frame_offset
        if frame_idx in frame_map:
            raise ValueError(f"Duplicate frame index {frame_idx} in {frame_dir}")
        frame_map[frame_idx] = path
    return frame_map


def _filter_frame_map_by_range(frame_map: dict[int, Path], start_frame: Optional[int], end_frame: Optional[int]) -> dict[int, Path]:
    if start_frame is None and end_frame is None:
        return frame_map
    filtered = {}
    for frame_idx, path in frame_map.items():
        if start_frame is not None and frame_idx < start_frame:
            continue
        if end_frame is not None and frame_idx > end_frame:
            continue
        filtered[frame_idx] = path
    return filtered


def _resolve_selected_frame_window(config: dict) -> tuple[Optional[int], Optional[int]]:
    runtime_cfg = config.get("runtime", {})
    preprocess_cfg = config.get("preprocess", {})

    start_frame = runtime_cfg.get("selected_frame_start", preprocess_cfg.get("start_frame"))
    end_frame = runtime_cfg.get("selected_frame_end", preprocess_cfg.get("end_frame"))

    start_frame = None if start_frame in (None, "") else int(start_frame)
    end_frame = None if end_frame in (None, "") else int(end_frame)
    return start_frame, end_frame


def _resolve_truth_frame_payload(truth_data: dict, testcase_name: Optional[str], truth_path: Path) -> dict:
    if testcase_name is not None:
        tc_data = truth_data.get(testcase_name)
        if tc_data is None:
            raise ValueError(f"Missing testcase {testcase_name} in {truth_path}")
        if not isinstance(tc_data, dict):
            raise ValueError(f"Invalid testcase payload for {testcase_name} in {truth_path}")
        return tc_data

    camera_keys = [key for key in truth_data if re.fullmatch(r"camera\d+", str(key))]
    if camera_keys:
        return truth_data

    if len(truth_data) != 1:
        raise ValueError(
            f"Expected either camera* keys or exactly one top-level testcase entry in {truth_path} "
            "when evaluation.testcase_name is null"
        )

    tc_data = next(iter(truth_data.values()))
    if not isinstance(tc_data, dict):
        raise ValueError(f"Invalid top-level testcase payload in {truth_path}")
    return tc_data

def run_evaluation(config: dict) -> None:
    eval_cfg = config.get("evaluation", {})
    if not eval_cfg.get("enabled", True):
        log_disabled()
        return

    paths = config["paths"]
    map_data = load_keypoints3d_map(paths["keypoints3d_map"])
    canonical_names = set([k["name"] for k in map_data["keypoints"]])
    priority1_names = map_data.get("priority1", [])
    priority2_names = map_data.get("priority2", [])

    truth_dir = Path(eval_cfg.get("ground_truth_dir", "input/gtruth_results"))
    out_dir = Path(paths.get("evaluation_output_dir", "output/evaluation_results"))
    testcase_name = eval_cfg.get("testcase_name", "testcase1")
    start_frame, end_frame = _resolve_selected_frame_window(config)

    if config.get("runtime", {}).get("clean_output", True):
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.csv"):
            old.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    module_dirs = {
        "posed": Path(paths["pose_output_dir"]) / "keypoints3d",
        "fused": Path(paths["fused_output_dir"]) / "keypoints3d",
        "learnable": Path(paths["learnable_output_dir"]) / "keypoints3d",
        "optimized": Path(paths["optimized_output_dir"]) / "keypoints3d",
    }

    for name, mdir in module_dirs.items():
        if not mdir.exists():
            raise FileNotFoundError(f"Missing required module directory: {mdir}")

    truth_frame_map = _load_frame_map(truth_dir, frame_offset=1)
    truth_frame_map = _filter_frame_map_by_range(truth_frame_map, start_frame, end_frame)
    if not truth_frame_map:
        raise ValueError(f"No ground truth frames found in {truth_dir}")

    gt_camera_keys = {
        "camera1": _camera_key_from_video_path(paths.get("camera1_video")),
        "camera2": _camera_key_from_video_path(paths.get("camera2_video")),
    }

    truth_frames = set(truth_frame_map)
    module_frame_maps = {}
    for name, mdir in module_dirs.items():
        module_frame_maps[name] = _filter_frame_map_by_range(_load_frame_map(mdir), start_frame, end_frame)
        mframes = set(module_frame_maps[name])
        if mframes != truth_frames:
            raise ValueError(f"Frame mismatch in module {name}. Expected {sorted(truth_frames)}, got {sorted(mframes)}")

    frames = sorted(truth_frames)
    
    # metrics: metric -> cam -> frame -> module -> priority -> value
    results = {
        "MPJPE": {"camera1": {}, "camera2": {}},
        "PA-MPJPE": {"camera1": {}, "camera2": {}},
        "PCK": {"camera1": {}, "camera2": {}},
    }

    for frame in frames:
        truth_path = truth_frame_map[frame]
        truth_data = _load_json(truth_path)
        tc_data = _resolve_truth_frame_payload(truth_data, testcase_name, truth_path)

        for cam in CAMERAS:
            gt_cam = gt_camera_keys[cam]
            if gt_cam not in tc_data:
                raise ValueError(f"Missing {gt_cam} in {truth_path}")
            
            truth_joints = {k: np.array(v, dtype=float) for k, v in tc_data[gt_cam].items()}
            if not canonical_names.issubset(truth_joints.keys()):
                raise ValueError(f"Truth {gt_cam} frame {frame} missing keys. Expected {canonical_names}")

            for metric in results:
                if frame not in results[metric][cam]:
                    results[metric][cam][frame] = {}

            for mod in MODULES:
                mod_path = module_frame_maps[mod][frame]
                mod_data = _load_json(mod_path)
                if cam not in mod_data:
                    raise ValueError(f"Missing {cam} in {mod_path}")
                
                pred_joints = {k: np.array(v, dtype=float) for k, v in mod_data[cam].items()}
                if set(pred_joints) != canonical_names:
                    raise ValueError(
                        f"Module {mod} {cam} frame {frame} must contain exactly canonical 21 keys. "
                        f"Expected {sorted(canonical_names)}, got {sorted(pred_joints)}"
                    )
                
                # compute metrics
                # MPJPE
                results["MPJPE"][cam][frame].setdefault(mod, {})
                results["MPJPE"][cam][frame][mod]["priority1_mm"] = _compute_mpjpe(pred_joints, truth_joints, priority1_names)
                results["MPJPE"][cam][frame][mod]["priority2_mm"] = _compute_mpjpe(pred_joints, truth_joints, priority2_names)
                
                # PA-MPJPE
                results["PA-MPJPE"][cam][frame].setdefault(mod, {})
                results["PA-MPJPE"][cam][frame][mod]["priority1_mm"] = _compute_pa_mpjpe(pred_joints, truth_joints, priority1_names)
                results["PA-MPJPE"][cam][frame][mod]["priority2_mm"] = _compute_pa_mpjpe(pred_joints, truth_joints, priority2_names)
                
                # PCK-mm
                results["PCK"][cam][frame].setdefault(mod, {})
                results["PCK"][cam][frame][mod]["priority1_mm"] = _compute_pck_mm(pred_joints, truth_joints, priority1_names)
                results["PCK"][cam][frame][mod]["priority2_mm"] = _compute_pck_mm(pred_joints, truth_joints, priority2_names)

    header = ["Frame"]
    for mod in MODULES:
        header.append(f"{mod}_priority1_mm")
        header.append(f"{mod}_priority2_mm")

    for metric in ["MPJPE", "PA-MPJPE", "PCK"]:
        for cam in CAMERAS:
            filename = f"{metric}_{CAMERA_FILE_NAMES[cam]}.csv"
            out_file = out_dir / filename
            with out_file.open("w", newline="", encoding="utf-8") as f:
                f.write("sep=,\n")
                writer = csv.writer(f)
                writer.writerow(header)
                
                avg_sums = {h: 0.0 for h in header[1:]}
                
                for frame in frames:
                    row = [frame]
                    for mod in MODULES:
                        v1 = results[metric][cam][frame][mod]["priority1_mm"]
                        v2 = results[metric][cam][frame][mod]["priority2_mm"]
                        row.append(f"{v1:.2f}")
                        row.append(f"{v2:.2f}")
                        avg_sums[f"{mod}_priority1_mm"] += v1
                        avg_sums[f"{mod}_priority2_mm"] += v2
                    writer.writerow(row)
                
                n_frames = len(frames)
                avg_row = ["AVERAGE"]
                for h in header[1:]:
                    avg_row.append(f"{(avg_sums[h]/n_frames):.2f}")
                writer.writerow(avg_row)

    log_done(str(out_dir))
