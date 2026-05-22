# -*- coding: utf-8 -*-
"""
Protocol 1 — Universal PA-MPJPE Evaluation (Procrustes Aligned)
Hỗ trợ nhiều định dạng, tự động map key, chạy không tương tác cho 2 camera.
Đọc ground truth từ một thư mục chứa các file JSON tổng hợp có cấu trúc:
{
  "testcase_name": {
    "camera1": { keypoint dict },
    "camera2": { keypoint dict }
  }
}
Mỗi file tương ứng một frame.
"""

import csv
import json
import sys
import re
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes

# ─────────────────────────────────────────────
# CONSTANTS & MAPPING
# ─────────────────────────────────────────────

# 8 key cánh tay + cánh chân (shoulders, elbows, hips, knees)
ARM_LEG_KEYS = {
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
}

# Key uy tín (2 vai + 2 hông)
RELIABLE_KEYS = {
    "left_shoulder", "right_shoulder",
    "left_hip",      "right_hip",
}

# Datamap chuẩn dùng để tự động map
STANDARD_KEYS = [
    "spine3", "spine4", "spine2", "spine", "pelvis", "neck", "head", "head_top",
    "left_clavicle", "left_shoulder", "left_elbow", "left_wrist", "left_hand",
    "right_clavicle", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "left_hip", "left_knee", "left_ankle", "left_foot", "left_toe",
    "right_hip", "right_knee", "right_ankle", "right_foot", "right_toe",
]

# Từ điển đồng nghĩa để map OpenPose format sang Standard
ALIASES = {
    "RShoulder": "right_shoulder", "LShoulder": "left_shoulder",
    "RElbow": "right_elbow",       "LElbow": "left_elbow",
    "RWrist": "right_wrist",       "LWrist": "left_wrist",
    "RHip": "right_hip",           "LHip": "left_hip",
    "RKnee": "right_knee",         "LKnee": "left_knee",
    "RAnkle": "right_ankle",       "LAnkle": "left_ankle",
    "Neck": "neck",                "MidHip": "pelvis",
    "RBigToe": "right_toe",        "LBigToe": "left_toe",
    "RHeel": "right_foot",         "LHeel": "left_foot",
    "Nose": "head", "REye": "head", "LEye": "head"
}

# ─────────────────────────────────────────────
# HELPERS — I/O & PARSING
# ─────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    val = input(f"{prompt}{hint}: ").strip()
    return val if val else default

def extract_frame_id(path: Path) -> int:
    """Trích xuất số từ tên file (vd: pose_data_12.json -> 12, 000000.json -> 0)"""
    nums = re.findall(r'\d+', path.stem)
    return int(nums[-1]) if nums else -1

def find_keypoints_path(data, path_so_far=""):
    """
    Tìm đường dẫn (dạng a.b.c) đến dict chứa keypoints 3D.
    Ưu tiên các path chứa 'annot3' hoặc 'keypoints'.
    """
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
    
    recurse(data, path_so_far)
    
    for cand in candidates:
        if 'annot3' in cand or 'keypoints' in cand:
            return cand
    return candidates[0] if candidates else None

def load_points_auto(path: Path, default_path="annotations.annot3.keypoints") -> dict:
    """Tải keypoints từ file JSON (dự đoán), tự động phát hiện đường dẫn."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    try:
        node = data
        for key in default_path.split("."):
            node = node[key]
        if isinstance(node, dict):
            sample_val = next(iter(node.values()))
            if isinstance(sample_val, (list, tuple)) and len(sample_val) == 3:
                return {k: np.array(v, dtype=float) for k, v in node.items()}
    except (KeyError, TypeError, StopIteration):
        pass
    
    found_path = find_keypoints_path(data)
    if found_path:
        node = data
        for key in found_path.split("."):
            node = node[key]
        return {k: np.array(v, dtype=float) for k, v in node.items()}
    
    raise ValueError(f"Không tìm thấy keypoints 3D trong file {path}")

def load_truth_with_cameras(file_path: Path, testcase_name: str) -> tuple[dict, dict]:
    """
    Đọc file ground truth tổng hợp, trả về (kp_cam1, kp_cam2)
    Cấu trúc: { testcase_name: { "camera1": {...}, "camera2": {...} } }
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    testcase_data = data.get(testcase_name)
    if testcase_data is None:
        if len(data) == 1:
            testcase_data = next(iter(data.values()))
        else:
            raise KeyError(f"Không tìm thấy testcase '{testcase_name}' trong file {file_path}")
    kp_cam1 = testcase_data.get("camera1")
    kp_cam2 = testcase_data.get("camera2")
    if kp_cam1 is None or kp_cam2 is None:
        raise ValueError(f"File {file_path} thiếu key 'camera1' hoặc 'camera2'")
    return {k: np.array(v, dtype=float) for k, v in kp_cam1.items()}, {k: np.array(v, dtype=float) for k, v in kp_cam2.items()}

# ─────────────────────────────────────────────
# AUTO MAPPING
# ─────────────────────────────────────────────

def normalize_key(k: str) -> str:
    return ALIASES.get(k, k)

def auto_map_keys(pred_keys: list[str], truth_keys: list[str], logs: list[str]) -> list[tuple[str, str]]:
    std_index = {k: i for i, k in enumerate(STANDARD_KEYS)}
    truth_norm_to_raw = {normalize_key(tk): tk for tk in truth_keys}
    
    mapping = []
    lines = ["", "── LOG MAPPING KEY3D " + "─" * 52]
    lines.append(f"  {'Pred key (Gốc)':18s} {'Truth key (Gốc)':18s} {'Map qua':15s} Trạng thái")
    lines.append("  " + "─" * 68)

    for pk in pred_keys:
        pk_norm = normalize_key(pk)
        if pk_norm in truth_norm_to_raw:
            tk_raw = truth_norm_to_raw[pk_norm]
            mapping.append((pk, tk_raw))
            method = "exact/alias" if pk != tk_raw else "exact"
            lines.append(f"  {pk:18s} {tk_raw:18s} {method:15s} ✓")
        elif pk_norm in std_index:
            fi = std_index[pk_norm]
            matched_tk_raw = next(
                (tk for tk in truth_keys if normalize_key(tk) in std_index and std_index[normalize_key(tk)] == fi),
                None
            )
            if matched_tk_raw:
                mapping.append((pk, matched_tk_raw))
                lines.append(f"  {pk:18s} {matched_tk_raw:18s} {'std_pos':15s} ✓")
            else:
                lines.append(f"  {pk:18s} {'—':18s} {'—':15s} ✗ không map được")
                logs.append(f"[WARN] Không map được truth cho: {pk}")
        else:
            lines.append(f"  {pk:18s} {'—':18s} {'—':15s} ✗ ngoài từ điển chuẩn")
            logs.append(f"[WARN] Ngoài STANDARD_KEYS & ALIASES: {pk}")

    lines.append("─" * 72)
    block = "\n".join(lines)
    print(block)
    logs.append(block)
    return mapping

# ─────────────────────────────────────────────
# THUẬT TOÁN TÍNH PA-MPJPE (Procrustes Analysis)
# ─────────────────────────────────────────────

def compute_pa_mpjpe(pred_joints: dict, truth_joints: dict, joint_pairs: list) -> dict:
    valid_pairs = [(pk, tk) for pk, tk in joint_pairs if pk in pred_joints and tk in truth_joints]
    if not valid_pairs:
        return {}

    pred_matrix = np.array([pred_joints[pk] for pk, _ in valid_pairs])
    truth_matrix = np.array([truth_joints[tk] for _, tk in valid_pairs])

    truth_mean = np.mean(truth_matrix, axis=0)
    truth_centered = truth_matrix - truth_mean
    norm_truth = np.linalg.norm(truth_centered) 
    
    mtx1, mtx2, disparity = procrustes(truth_matrix, pred_matrix)

    mtx1_real = mtx1 * norm_truth
    mtx2_real = mtx2 * norm_truth
    distances = np.linalg.norm(mtx1_real - mtx2_real, axis=1) * 1000.0

    return {pk: float(dist) for (pk, _), dist in zip(valid_pairs, distances)}

# ─────────────────────────────────────────────
# EVALUATION CHO CẢ HAI CAMERA TỪ THƯ MỤC TRUTH TỔNG HỢP
# ─────────────────────────────────────────────

def evaluate_cameras_from_truth_folder(pred_dir: Path, truth_dir: Path, testcase_name: str, logs: list[str]) -> dict:
    """
    Đánh giá cho cả hai camera dựa trên thư mục truth chứa các file tổng hợp.
    Trả về dict với keys: "CAM1", "CAM2", mỗi value là result dict.
    """
    pred_files = sorted(
        [p for p in pred_dir.glob("*.json") if extract_frame_id(p) != -1],
        key=extract_frame_id
    )
    if not pred_files:
        print(f"[ERR] Không có file JSON hợp lệ trong {pred_dir}")
        return {}

    truth_files = sorted(
        [p for p in truth_dir.glob("*.json") if extract_frame_id(p) != -1],
        key=extract_frame_id
    )
    if not truth_files:
        print(f"[ERR] Không có file JSON hợp lệ trong {truth_dir}")
        return {}

    n_frames = min(len(pred_files), len(truth_files))
    if len(pred_files) != len(truth_files):
        logs.append(f"[WARN] Số lượng file không khớp (pred={len(pred_files)}, truth={len(truth_files)}). Chỉ xử lý {n_frames} frame đầu.")
        print(f"  [!] Số lượng file không khớp, chỉ xử lý {n_frames} frame.")

    # Cập nhật structure để lưu thêm dữ liệu từng frame
    results = {
        "CAM1": {"frame_data": [], "mpjpe_all": [], "mpjpe_arm_leg": [], "mpjpe_reliable": [], "frames": 0},
        "CAM2": {"frame_data": [], "mpjpe_all": [], "mpjpe_arm_leg": [], "mpjpe_reliable": [], "frames": 0}
    }
    joint_pairs_cache = {}

    for i in range(n_frames):
        pred_path = pred_files[i]
        truth_path = truth_files[i]
        frame_id = extract_frame_id(pred_path)

        try:
            pred_joints = load_points_auto(pred_path)
            kp_cam1, kp_cam2 = load_truth_with_cameras(truth_path, testcase_name)
        except Exception as e:
            logs.append(f"[ERR] Frame {i}: {e} → bỏ qua")
            continue

        for cam, truth_joints in [("CAM1", kp_cam1), ("CAM2", kp_cam2)]:
            cache_key = (cam, tuple(sorted(pred_joints.keys())), tuple(sorted(truth_joints.keys())))
            if cache_key not in joint_pairs_cache:
                pred_keys = sorted(pred_joints.keys())
                truth_keys = sorted(truth_joints.keys())
                joint_pairs = auto_map_keys(pred_keys, truth_keys, logs)
                if not joint_pairs:
                    logs.append(f"[WARN] {cam}: Không thể map keys, bỏ qua frame {i}")
                    continue
                joint_pairs_cache[cache_key] = joint_pairs
            else:
                joint_pairs = joint_pairs_cache[cache_key]

            errors = compute_pa_mpjpe(pred_joints, truth_joints, joint_pairs)
            if not errors:
                continue

            frame_all = np.mean(list(errors.values()))
            al_vals = [v for k, v in errors.items() if normalize_key(k) in ARM_LEG_KEYS]
            rel_vals = [v for k, v in errors.items() if normalize_key(k) in RELIABLE_KEYS]
            
            al_mean = np.mean(al_vals) if al_vals else np.nan
            rel_mean = np.mean(rel_vals) if rel_vals else np.nan

            results[cam]["mpjpe_all"].append(frame_all)
            if al_vals: results[cam]["mpjpe_arm_leg"].append(al_mean)
            if rel_vals: results[cam]["mpjpe_reliable"].append(rel_mean)
            
            # Lưu lại data của frame hiện tại
            results[cam]["frame_data"].append({
                "frame": frame_id,
                "all": frame_all,
                "arm_leg": al_mean,
                "reliable": rel_mean
            })
            
            results[cam]["frames"] += 1

    final = {}
    for cam in ["CAM1", "CAM2"]:
        if results[cam]["mpjpe_all"]:
            final[cam] = {
                "cam": cam,
                "mpjpe_all": np.mean(results[cam]["mpjpe_all"]),
                "mpjpe_arm_leg": np.mean(results[cam]["mpjpe_arm_leg"]) if results[cam]["mpjpe_arm_leg"] else np.nan,
                "mpjpe_reliable": np.mean(results[cam]["mpjpe_reliable"]) if results[cam]["mpjpe_reliable"] else np.nan,
                "frames": results[cam]["frames"],
                "frame_data": results[cam]["frame_data"] # Trả về list frame_data cho CSV
            }
    return final

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logs: list[str] = []

    print("\n" + "═" * 75)
    print("  PROTOCOL 1 — UNIVERSAL PA-MPJPE EVALUATION (PROCRUSTES)")
    print("═" * 75)

    pred_dir = Path(_ask("Thư mục dữ liệu Dự đoán (chứa JSON mỗi frame)", "pred_jsons"))
    if not pred_dir.exists():
        print(f"[ERR] Không tìm thấy: {pred_dir}")
        sys.exit(1)

    truth_dir = Path(_ask("Thư mục Ground Truth tổng hợp (chứa file JSON mỗi frame có cấu trúc test case + 2 camera)", "truth_combined"))
    if not truth_dir.exists():
        print(f"[ERR] Không tìm thấy: {truth_dir}")
        sys.exit(1)

    testcase_name = _ask("Tên test case (trong file ground truth)", "testcase1")
    if not testcase_name:
        testcase_name = "testcase1"

    results = evaluate_cameras_from_truth_folder(pred_dir, truth_dir, testcase_name, logs)

    print("\n" + "═" * 75)
    print("  KẾT QUẢ PA-MPJPE (mm)")
    print("═" * 75)
    print(f"{'Camera':<8} {'Toàn thân':>12} {'Tay+Chân (8 key)':>18} {'Uy tín (Vai+Hông)':>20} {'Frames':>8}")
    print("─" * 70)

    for cam in ["CAM1", "CAM2"]:
        if cam in results:
            r = results[cam]
            print(f"{r['cam']:<8} {r['mpjpe_all']:>12.2f} {r['mpjpe_arm_leg']:>18.2f} {r['mpjpe_reliable']:>20.2f} {r['frames']:>8}")
        else:
            print(f"{cam:<8} {'N/A':>12} {'N/A':>18} {'N/A':>20} {0:>8}")

    print("\n" + "═" * 75)
    print("  XUẤT FILE CSV")
    print("═" * 75)

    # ── Ghi CSV cho từng camera ──
    for cam in ["CAM1", "CAM2"]:
        if cam in results:
            r = results[cam]
            csv_path = Path(f"pa_mpjpe_{cam.lower()}.csv")
            
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                cf.write("sep=,\n")  # Chuẩn giúp Excel tự chia cột
                writer = csv.writer(cf)
                
                # Header
                writer.writerow(["Frame", "Toan_than", "Tay_Chan_8key", "Uy_tin_Vai_Hong"])
                
                # Rows chi tiết từng frame
                for fdata in r["frame_data"]:
                    writer.writerow([
                        fdata["frame"],
                        f"{fdata['all']:.2f}",
                        f"{fdata['arm_leg']:.2f}" if not np.isnan(fdata['arm_leg']) else "",
                        f"{fdata['reliable']:.2f}" if not np.isnan(fdata['reliable']) else ""
                    ])
                
                # Row tổng trung bình
                writer.writerow([
                    "AVERAGE",
                    f"{r['mpjpe_all']:.2f}",
                    f"{r['mpjpe_arm_leg']:.2f}" if not np.isnan(r['mpjpe_arm_leg']) else "",
                    f"{r['mpjpe_reliable']:.2f}" if not np.isnan(r['mpjpe_reliable']) else ""
                ])
                
            print(f"  [OK] Đã lưu kết quả chi tiết của {cam} vào {csv_path.resolve()}")

    if logs:
        log_path = Path("evaluation_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        print(f"\n[INFO] Chi tiết mapping và cảnh báo đã được lưu vào {log_path}")

if __name__ == "__main__":
    main()