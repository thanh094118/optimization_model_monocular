from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes

from json_io import write_json

ARM_LEG_KEYS = {
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
}

RELIABLE_KEYS = {
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
}

STANDARD_KEYS = [
    "spine3", "spine4", "spine2", "spine", "pelvis", "neck", "head", "head_top",
    "left_clavicle", "left_shoulder", "left_elbow", "left_wrist", "left_hand",
    "right_clavicle", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "left_hip", "left_knee", "left_ankle", "left_foot", "left_toe",
    "right_hip", "right_knee", "right_ankle", "right_foot", "right_toe",
]

ALIASES = {
    "RShoulder": "right_shoulder", "LShoulder": "left_shoulder",
    "RElbow": "right_elbow", "LElbow": "left_elbow",
    "RWrist": "right_wrist", "LWrist": "left_wrist",
    "RHip": "right_hip", "LHip": "left_hip",
    "RKnee": "right_knee", "LKnee": "left_knee",
    "RAnkle": "right_ankle", "LAnkle": "left_ankle",
    "Neck": "neck", "MidHip": "pelvis",
    "RBigToe": "right_toe", "LBigToe": "left_toe",
    "RHeel": "right_foot", "LHeel": "left_foot",
    "Nose": "head", "REye": "head", "LEye": "head",
}


def _frame_index(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else -1


def _find_keypoints_path(data):
    candidates = []

    def recurse(node, current_path):
        if isinstance(node, dict):
            values = list(node.values())
            if values and all(
                isinstance(v, (list, tuple)) and len(v) == 3 and
                all(isinstance(x, (int, float)) for x in v)
                for v in values
            ):
                candidates.append(current_path)
            else:
                for k, v in node.items():
                    new_path = f"{current_path}.{k}" if current_path else k
                    recurse(v, new_path)

    recurse(data, "")

    for cand in candidates:
        if "annot3" in cand or "keypoints" in cand:
            return cand
    return candidates[0] if candidates else None


def _load_points_auto(path: Path, default_path="annotations.annot3.keypoints") -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        node = data
        for key in default_path.split("."):
            node = node[key]
        if isinstance(node, dict):
            sample = next(iter(node.values()))
            if isinstance(sample, (list, tuple)) and len(sample) == 3:
                return {k: np.asarray(v, dtype=float) for k, v in node.items()}
    except Exception:
        pass

    found = _find_keypoints_path(data)
    if found:
        node = data
        for key in found.split("."):
            node = node[key]
        return {k: np.asarray(v, dtype=float) for k, v in node.items()}

    raise ValueError(f"Cannot find 3D keypoints in {path}")


def _load_pred_with_cameras(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Standard pipeline outputs: {"camera1": {...}, "camera2": {...}}
    if isinstance(data, dict) and "camera1" in data and "camera2" in data:
        cam1 = data.get("camera1")
        cam2 = data.get("camera2")
        if isinstance(cam1, dict) and isinstance(cam2, dict):
            return (
                {k: np.asarray(v, dtype=float) for k, v in cam1.items()},
                {k: np.asarray(v, dtype=float) for k, v in cam2.items()},
            )

    # Legacy or other layouts: fallback to a single set for both cameras.
    single = _load_points_auto(path)
    return single, single


def _load_truth_with_cameras(file_path: Path, testcase_name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    testcase_data = data.get(testcase_name)
    if testcase_data is None:
        if len(data) == 1:
            testcase_data = next(iter(data.values()))
        else:
            raise KeyError(f"Missing testcase {testcase_name!r} in {file_path}")

    kp_cam1 = testcase_data.get("camera1")
    kp_cam2 = testcase_data.get("camera2")
    if kp_cam1 is None or kp_cam2 is None:
        raise ValueError(f"Missing camera1/camera2 in truth file: {file_path}")

    return (
        {k: np.asarray(v, dtype=float) for k, v in kp_cam1.items()},
        {k: np.asarray(v, dtype=float) for k, v in kp_cam2.items()},
    )


def _normalize_key(k: str) -> str:
    return ALIASES.get(k, k)


def _auto_map_keys(pred_keys: list[str], truth_keys: list[str]) -> list[tuple[str, str]]:
    std_index = {k: i for i, k in enumerate(STANDARD_KEYS)}
    truth_norm_to_raw = {_normalize_key(tk): tk for tk in truth_keys}

    mapping = []
    for pk in pred_keys:
        pk_norm = _normalize_key(pk)
        if pk_norm in truth_norm_to_raw:
            mapping.append((pk, truth_norm_to_raw[pk_norm]))
            continue
        if pk_norm in std_index:
            fi = std_index[pk_norm]
            matched = next(
                (tk for tk in truth_keys if _normalize_key(tk) in std_index and std_index[_normalize_key(tk)] == fi),
                None,
            )
            if matched:
                mapping.append((pk, matched))
    return mapping


def _compute_pa_mpjpe(pred_joints: dict, truth_joints: dict, joint_pairs: list[tuple[str, str]]) -> dict[str, float]:
    valid_pairs = [(pk, tk) for pk, tk in joint_pairs if pk in pred_joints and tk in truth_joints]
    if not valid_pairs:
        return {}

    pred_matrix = np.asarray([pred_joints[pk] for pk, _ in valid_pairs], dtype=float)
    truth_matrix = np.asarray([truth_joints[tk] for _, tk in valid_pairs], dtype=float)

    truth_centered = truth_matrix - np.mean(truth_matrix, axis=0)
    norm_truth = np.linalg.norm(truth_centered)

    mtx1, mtx2, _ = procrustes(truth_matrix, pred_matrix)

    mtx1_real = mtx1 * norm_truth
    mtx2_real = mtx2 * norm_truth
    distances = np.linalg.norm(mtx1_real - mtx2_real, axis=1) * 1000.0

    return {pk: float(dist) for (pk, _), dist in zip(valid_pairs, distances)}


def _evaluate_single_pred_dir(pred_dir: Path, truth_dir: Path, testcase_name: str, logs: list[str]) -> dict:
    pred_files = sorted([p for p in pred_dir.glob("*.json") if _frame_index(p) != -1], key=_frame_index)
    truth_files = sorted([p for p in truth_dir.glob("*.json") if _frame_index(p) != -1], key=_frame_index)

    if not pred_files:
        raise FileNotFoundError(f"No JSON files in pred dir: {pred_dir}")
    if not truth_files:
        raise FileNotFoundError(f"No JSON files in truth dir: {truth_dir}")

    pred_by_frame = {_frame_index(p): p for p in pred_files}
    truth_by_frame = {_frame_index(p): p for p in truth_files}
    common_frame_ids = sorted(set(pred_by_frame.keys()) & set(truth_by_frame.keys()))

    if not common_frame_ids:
        raise ValueError(f"No common frame_id between pred={pred_dir} and truth={truth_dir}")

    missing_in_truth = sorted(set(pred_by_frame.keys()) - set(truth_by_frame.keys()))
    missing_in_pred = sorted(set(truth_by_frame.keys()) - set(pred_by_frame.keys()))
    if missing_in_truth:
        logs.append(f"[WARN] Frames missing in truth: {missing_in_truth[:20]}{'...' if len(missing_in_truth) > 20 else ''}")
    if missing_in_pred:
        logs.append(f"[WARN] Frames missing in pred: {missing_in_pred[:20]}{'...' if len(missing_in_pred) > 20 else ''}")

    stats = {
        "CAM1": {"frame_data": [], "all": [], "arm_leg": [], "reliable": []},
        "CAM2": {"frame_data": [], "all": [], "arm_leg": [], "reliable": []},
    }
    joint_pairs_cache = {}

    for frame_id in common_frame_ids:
        pred_path = pred_by_frame[frame_id]
        truth_path = truth_by_frame[frame_id]

        try:
            pred_cam1, pred_cam2 = _load_pred_with_cameras(pred_path)
            truth_cam1, truth_cam2 = _load_truth_with_cameras(truth_path, testcase_name)
        except Exception as exc:
            logs.append(f"[ERR] {pred_path.name}: {exc}")
            continue

        for cam_name, pred_joints, truth_joints in (
            ("CAM1", pred_cam1, truth_cam1),
            ("CAM2", pred_cam2, truth_cam2),
        ):
            cache_key = (cam_name, tuple(sorted(pred_joints.keys())), tuple(sorted(truth_joints.keys())))
            if cache_key not in joint_pairs_cache:
                joint_pairs_cache[cache_key] = _auto_map_keys(sorted(pred_joints.keys()), sorted(truth_joints.keys()))
            joint_pairs = joint_pairs_cache[cache_key]
            if not joint_pairs:
                logs.append(f"[WARN] {cam_name} cannot map joints for {pred_path.name}")
                continue

            errors = _compute_pa_mpjpe(pred_joints, truth_joints, joint_pairs)
            if not errors:
                continue

            all_mean = float(np.mean(list(errors.values())))
            al_vals = [v for k, v in errors.items() if _normalize_key(k) in ARM_LEG_KEYS]
            rel_vals = [v for k, v in errors.items() if _normalize_key(k) in RELIABLE_KEYS]
            al_mean = float(np.mean(al_vals)) if al_vals else float("nan")
            rel_mean = float(np.mean(rel_vals)) if rel_vals else float("nan")

            stats[cam_name]["all"].append(all_mean)
            if al_vals:
                stats[cam_name]["arm_leg"].append(al_mean)
            if rel_vals:
                stats[cam_name]["reliable"].append(rel_mean)
            stats[cam_name]["frame_data"].append({"frame": frame_id, "all": all_mean, "arm_leg": al_mean, "reliable": rel_mean})

    final = {}
    for cam in ("CAM1", "CAM2"):
        if not stats[cam]["all"]:
            continue
        final[cam] = {
            "cam": cam,
            "mpjpe_all": float(np.mean(stats[cam]["all"])),
            "mpjpe_arm_leg": float(np.mean(stats[cam]["arm_leg"])) if stats[cam]["arm_leg"] else float("nan"),
            "mpjpe_reliable": float(np.mean(stats[cam]["reliable"])) if stats[cam]["reliable"] else float("nan"),
            "frames": len(stats[cam]["frame_data"]),
            "frame_data": stats[cam]["frame_data"],
        }
    return final


def _write_module_csv(module_name: str, cam: str, result: dict, output_dir: Path) -> None:
    csv_path = output_dir / f"pa_mpjpe_{module_name}_{cam.lower()}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        cf.write("sep=,\n")
        writer = csv.writer(cf)
        writer.writerow(["Frame", "Toan_than", "Tay_Chan_8key", "Uy_tin_Vai_Hong"])
        for fdata in result["frame_data"]:
            writer.writerow([
                fdata["frame"],
                f"{fdata['all']:.2f}",
                f"{fdata['arm_leg']:.2f}" if not np.isnan(fdata["arm_leg"]) else "",
                f"{fdata['reliable']:.2f}" if not np.isnan(fdata["reliable"]) else "",
            ])
        writer.writerow([
            "AVERAGE",
            f"{result['mpjpe_all']:.2f}",
            f"{result['mpjpe_arm_leg']:.2f}" if not np.isnan(result["mpjpe_arm_leg"]) else "",
            f"{result['mpjpe_reliable']:.2f}" if not np.isnan(result["mpjpe_reliable"]) else "",
        ])


def run_evaluation(config: dict) -> None:
    paths = config["paths"]
    runtime_cfg = config.get("runtime", {})
    eval_cfg = config.get("evaluation", {})

    if not eval_cfg.get("enabled", True):
        print("[Evaluation] Disabled by config: evaluation.enabled=false")
        return

    truth_dir = Path(eval_cfg.get("ground_truth_dir", "input/gtruth_results"))
    testcase_name = eval_cfg.get("testcase_name", "testcase1")
    output_dir = Path(paths.get("evaluation_output_dir", "output/evaluation_results"))

    module_inputs = {
        "pose": Path(paths["pose_output_dir"]) / "keypoints3d",
        "fusion": Path(paths["fused_output_dir"]) / "keypoints3d",
        "learnable": Path(paths["learnable_output_dir"]) / "keypoints3d",
    }

    if runtime_cfg.get("clean_output", True):
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob("*.csv"):
            old.unlink()
        for old in output_dir.glob("*.json"):
            old.unlink()
        log_file = output_dir / "evaluation_log.txt"
        if log_file.exists():
            log_file.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {truth_dir}")

    all_results = {}
    logs: list[str] = []

    for module_name, pred_dir in module_inputs.items():
        if not pred_dir.exists():
            print(f"[Evaluation] Skip {module_name}: missing {pred_dir}")
            continue
        print(f"[Evaluation] Evaluating {module_name} from {pred_dir}")
        result = _evaluate_single_pred_dir(pred_dir, truth_dir, testcase_name, logs)
        all_results[module_name] = result
        for cam in ("CAM1", "CAM2"):
            if cam in result:
                _write_module_csv(module_name, cam, result[cam], output_dir)

    write_json(output_dir / "summary.json", all_results)
    if logs:
        (output_dir / "evaluation_log.txt").write_text("\n".join(logs), encoding="utf-8")

    print(f"[Evaluation] Done. Output: {output_dir}")
