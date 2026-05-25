from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes

from evaluation_pipeline.executor import ALIASES, ARM_LEG_KEYS, RELIABLE_KEYS


def _frame_index(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else -1


def _normalize_key(k: str) -> str:
    return ALIASES.get(k, k)


def _auto_map_keys(pred_keys: list[str], truth_keys: list[str]) -> list[tuple[str, str]]:
    truth_norm_to_raw = {_normalize_key(tk): tk for tk in truth_keys}
    mapping = []
    for pk in pred_keys:
        pk_norm = _normalize_key(pk)
        if pk_norm in truth_norm_to_raw:
            mapping.append((pk, truth_norm_to_raw[pk_norm]))
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
    distances = np.linalg.norm((mtx1 - mtx2) * norm_truth, axis=1) * 1000.0
    return {pk: float(dist) for (pk, _), dist in zip(valid_pairs, distances)}


def _load_hbh_pred_with_cameras(path: Path) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key3d = data.get("keypoints3d", {})
    cam1 = key3d.get("camera1", {})
    cam2 = key3d.get("camera2", {})
    return cam1, cam2


def _load_truth_with_cameras(file_path: Path, testcase_name: str) -> tuple[dict, dict]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    testcase_data = data.get(testcase_name)
    if testcase_data is None:
        if len(data) == 1:
            testcase_data = next(iter(data.values()))
        else:
            raise KeyError(f"Missing testcase {testcase_name!r} in {file_path}")
    return testcase_data["camera1"], testcase_data["camera2"]


def _evaluate_method_dir(method_dir: Path, truth_dir: Path, testcase_name: str) -> list[dict]:
    pred_files = sorted([p for p in method_dir.glob("hbh_data_*.json") if _frame_index(p) != -1], key=_frame_index)
    truth_files = sorted([p for p in truth_dir.glob("*.json") if _frame_index(p) != -1], key=_frame_index)
    truth_by_frame = {_frame_index(p): p for p in truth_files}

    rows = []
    for pred_path in pred_files:
        frame_id = _frame_index(pred_path)
        truth_path = truth_by_frame.get(frame_id)
        if truth_path is None:
            continue

        pred_cam1, pred_cam2 = _load_hbh_pred_with_cameras(pred_path)
        truth_cam1, truth_cam2 = _load_truth_with_cameras(truth_path, testcase_name)

        frame_stat = {"frame": frame_id}
        for cam_name, pred_joints, truth_joints in (("CAM1", pred_cam1, truth_cam1), ("CAM2", pred_cam2, truth_cam2)):
            pairs = _auto_map_keys(sorted(pred_joints.keys()), sorted(truth_joints.keys()))
            errs = _compute_pa_mpjpe(pred_joints, truth_joints, pairs)
            if not errs:
                frame_stat[f"{cam_name}_all"] = np.nan
                frame_stat[f"{cam_name}_arm_leg"] = np.nan
                frame_stat[f"{cam_name}_reliable"] = np.nan
                continue
            all_mean = float(np.mean(list(errs.values())))
            al_vals = [v for k, v in errs.items() if _normalize_key(k) in ARM_LEG_KEYS]
            rel_vals = [v for k, v in errs.items() if _normalize_key(k) in RELIABLE_KEYS]
            frame_stat[f"{cam_name}_all"] = all_mean
            frame_stat[f"{cam_name}_arm_leg"] = float(np.mean(al_vals)) if al_vals else np.nan
            frame_stat[f"{cam_name}_reliable"] = float(np.mean(rel_vals)) if rel_vals else np.nan
        rows.append(frame_stat)
    return rows


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _write_method_csv(output_root: Path, method_name: str, rows: list[dict]) -> None:
    csv_path = output_root / f"pa_mpjpe_hbh_{method_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        cf.write("sep=,\n")
        writer = csv.writer(cf)
        writer.writerow([
            "Frame",
            "CAM1_Toan_than", "CAM1_Tay_Chan_8key", "CAM1_Uy_tin_Vai_Hong",
            "CAM2_Toan_than", "CAM2_Tay_Chan_8key", "CAM2_Uy_tin_Vai_Hong",
        ])
        for r in rows:
            writer.writerow([
                r["frame"],
                f"{r['CAM1_all']:.2f}" if not np.isnan(r.get("CAM1_all", np.nan)) else "",
                f"{r['CAM1_arm_leg']:.2f}" if not np.isnan(r.get("CAM1_arm_leg", np.nan)) else "",
                f"{r['CAM1_reliable']:.2f}" if not np.isnan(r.get("CAM1_reliable", np.nan)) else "",
                f"{r['CAM2_all']:.2f}" if not np.isnan(r.get("CAM2_all", np.nan)) else "",
                f"{r['CAM2_arm_leg']:.2f}" if not np.isnan(r.get("CAM2_arm_leg", np.nan)) else "",
                f"{r['CAM2_reliable']:.2f}" if not np.isnan(r.get("CAM2_reliable", np.nan)) else "",
            ])

        writer.writerow([
            "AVERAGE",
            f"{_safe_mean([r.get('CAM1_all', np.nan) for r in rows]):.2f}",
            f"{_safe_mean([r.get('CAM1_arm_leg', np.nan) for r in rows]):.2f}",
            f"{_safe_mean([r.get('CAM1_reliable', np.nan) for r in rows]):.2f}",
            f"{_safe_mean([r.get('CAM2_all', np.nan) for r in rows]):.2f}",
            f"{_safe_mean([r.get('CAM2_arm_leg', np.nan) for r in rows]):.2f}",
            f"{_safe_mean([r.get('CAM2_reliable', np.nan) for r in rows]):.2f}",
        ])


def run_hbh_evaluation(config: dict) -> None:
    paths = config.get("paths", {})
    eval_cfg = config.get("evaluation", {})

    output_root = Path(paths.get("hbh_output_dir", "output/hbh_results"))
    truth_dir = Path(eval_cfg.get("ground_truth_dir", "input/gtruth_results"))
    testcase_name = eval_cfg.get("testcase_name", "testcase1")

    if not truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {truth_dir}")

    method_dirs = sorted([p for p in output_root.glob("method_*") if p.is_dir()])
    if not method_dirs:
        print(f"[HBH-Eval] No method directories found under {output_root}")
        return

    for old in output_root.glob("pa_mpjpe_hbh_*.csv"):
        old.unlink()

    for method_dir in method_dirs:
        method_name = method_dir.name.replace("method_", "")
        rows = _evaluate_method_dir(method_dir, truth_dir, testcase_name)
        _write_method_csv(output_root, method_name, rows)

    print(f"[HBH-Eval] Done | methods={len(method_dirs)} | output={output_root}")
