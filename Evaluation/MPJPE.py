# -*- coding: utf-8 -*-
"""
Protocol 1 — Universal MPJPE Evaluation (Root Aligned)
Phiên bản: Ghi log tổng kết theo từng keypoint và toàn thân (Bỏ bảng per-frame).
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

# ─────────────────────────────────────────────
# CONSTANTS & MAPPING
# ─────────────────────────────────────────────

# 12 key cánh tay + cánh chân
ARM_LEG_KEYS = {
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_hand",     "right_hand",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
}

# Key uy tín (2 vai + 2 hông)
RELIABLE_KEYS = {
    "left_shoulder", "right_shoulder",
    "left_hip",      "right_hip",
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

def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("  [!] Vui lòng nhập số nguyên.")

def extract_frame_id(path: Path) -> int:
    nums = re.findall(r'\d+', path.stem)
    return int(nums[-1]) if nums else -1

def _find_keypoint_dicts(node, path: str = "") -> list[tuple[str, dict]]:
    results = []
    if isinstance(node, dict):
        vals = list(node.values())
        if vals and all(
            isinstance(v, (list, tuple)) and len(v) == 3
            and all(isinstance(x, (int, float)) for x in v)
            for v in vals
        ):
            results.append((path, {k: np.array(v, dtype=float) for k, v in node.items()}))
        else:
            for k, v in node.items():
                child = f"{path}.{k}" if path else k
                results.extend(_find_keypoint_dicts(v, child))
    return results

def resolve_annot_path(sample_file: Path, label: str) -> str:
    with open(sample_file, encoding="utf-8") as f:
        data = json.load(f)
    candidates = _find_keypoint_dicts(data)
    
    if not candidates:
        print(f"[ERR] Không tìm thấy tọa độ 3D hợp lệ trong {sample_file.name}")
        sys.exit(1)
        
    if len(candidates) == 1:
        print(f"  [{label}] Tự động nhận diện cấu trúc: '{candidates[0][0]}'")
        return candidates[0][0]

    print(f"\n  [{label}] Tìm thấy nhiều nhóm tọa độ 3D trong file, vui lòng chọn:")
    for i, (p, d) in enumerate(candidates):
        preview = list(d.keys())[:3]
        print(f"    [{i}] Path: {p if p else '<root>'}  (Ví dụ: {preview}...)")
        
    while True:
        raw = _ask("  Chọn index", "0")
        try:
            chosen = candidates[int(raw)][0]
            print(f"  Đã chọn: {chosen if chosen else '<root>'}")
            return chosen
        except (ValueError, IndexError):
            print("  [!] Index không hợp lệ.")

def load_points_by_path(path: Path, annot_path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    node = data
    if annot_path:
        for key in annot_path.split("."):
            node = node[key]
    return {k: np.array(v, dtype=float) for k, v in node.items()}

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
    print("\n".join(lines))
    return mapping

# ─────────────────────────────────────────────
# THUẬT TOÁN TÍNH MPJPE (Root Aligned)
# ─────────────────────────────────────────────

def get_root_coord(joints: dict) -> np.ndarray:
    """Tìm tọa độ gốc (Pelvis/MidHip). Nếu không có, tính trung bình 2 hông."""
    for k, v in joints.items():
        if normalize_key(k) == "pelvis":
            return v
            
    l_hip = next((v for k, v in joints.items() if normalize_key(k) == "left_hip"), None)
    r_hip = next((v for k, v in joints.items() if normalize_key(k) == "right_hip"), None)
    
    if l_hip is not None and r_hip is not None:
        return (l_hip + r_hip) / 2.0
        
    return None

def compute_mpjpe(pred_joints: dict, truth_joints: dict, joint_pairs: list) -> dict:
    """Căn chỉnh trọng tâm về Pelvis, sau đó tính Euclidean distance."""
    pred_root = get_root_coord(pred_joints)
    truth_root = get_root_coord(truth_joints)

    if pred_root is None or truth_root is None:
        raise ValueError("Thiếu dữ liệu Hông (Pelvis / LHip+RHip) để làm gốc tọa độ.")

    pred_aligned = {k: v - pred_root for k, v in pred_joints.items()}
    truth_aligned = {k: v - truth_root for k, v in truth_joints.items()}

    errors = {}
    for pk, tk in joint_pairs:
        if pk in pred_aligned and tk in truth_aligned:
            dist = np.linalg.norm(pred_aligned[pk] - truth_aligned[tk])
            errors[pk] = float(dist) * 1000.0

    return errors

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logs: list[str] = []

    print("\n" + "═" * 75)
    print("  PROTOCOL 1 — UNIVERSAL MPJPE EVALUATION (ROOT ALIGNED)")
    print("═" * 75)

    # ── 1. Dữ liệu Dự đoán ─────────────────────────────────────
    pred_dir = Path(_ask("Thư mục dữ liệu Dự đoán (fused/pose/keypoints)", "pred_jsons"))
    if not pred_dir.exists():
        print(f"[ERR] Không tìm thấy: {pred_dir}"); sys.exit(1)

    pred_files = sorted([p for p in pred_dir.glob("*.json") if extract_frame_id(p) != -1], key=extract_frame_id)
    if not pred_files:
        print(f"[ERR] Không có file JSON hợp lệ trong {pred_dir}"); sys.exit(1)

    pred_annot_path = resolve_annot_path(pred_files[0], "DỮ LIỆU DỰ ĐOÁN")
    pred_sample = load_points_by_path(pred_files[0], pred_annot_path)
    pred_keys = sorted(pred_sample.keys())

    # ── 2. Dữ liệu Thực tế ─────────────────────────────────────
    print("\n" + "─" * 40)
    truth_dir = Path(_ask("Thư mục dữ liệu Thực tế (Ground Truth)", "truth_jsons"))
    if not truth_dir.exists():
        print(f"[ERR] Không tìm thấy: {truth_dir}"); sys.exit(1)

    truth_files = sorted([p for p in truth_dir.glob("*.json") if extract_frame_id(p) != -1], key=extract_frame_id)
    if not truth_files:
        print(f"[ERR] Không có file JSON hợp lệ trong {truth_dir}"); sys.exit(1)

    truth_start = _ask_int("truth_start (Index frame đầu tiên của truth)", extract_frame_id(truth_files[0]))
    truth_annot_path = resolve_annot_path(truth_files[0], "GROUND TRUTH")
    truth_sample = load_points_by_path(truth_files[0], truth_annot_path)
    truth_keys = sorted(truth_sample.keys())

    # ── 3. Map Key ─────────────────────────────────────────────
    joint_pairs = auto_map_keys(pred_keys, truth_keys, logs)
    if not joint_pairs:
        print("[ERR] Không có cặp key nào được map."); sys.exit(1)

    # ── 4. Xử lý Lỗi (Gom nhóm dữ liệu, không in bảng per-frame) ──
    print("\n  Đang tính toán MPJPE. Vui lòng chờ...")
    
    missing_frames = []
    
    # Từ điển lưu list các lỗi của từng khớp qua các frame
    keypoint_errors = defaultdict(list)

    for pred_path in pred_files:
        pred_id = extract_frame_id(pred_path)
        truth_id = truth_start + (pred_id - extract_frame_id(pred_files[0]))
        
        truth_path_matches = [p for p in truth_files if extract_frame_id(p) == truth_id]
        
        if not truth_path_matches:
            missing_frames.append(pred_id)
            logs.append(f"[WARN] Pred {pred_id}: Không tìm thấy truth tương ứng → bỏ qua")
            continue

        try:
            pred_joints = load_points_by_path(pred_path, pred_annot_path)
            truth_joints = load_points_by_path(truth_path_matches[0], truth_annot_path)
            errors = compute_mpjpe(pred_joints, truth_joints, joint_pairs)
        except Exception as e:
            logs.append(f"[ERR] Frame {pred_id}: {e} → bỏ qua")
            continue

        if not errors: continue

        # Ghi nhận lỗi vào dictionary theo từng khớp
        for pk, err in errors.items():
            keypoint_errors[pk].append(err)

    if not keypoint_errors:
        print("\n[ERR] Không có frame nào được tính."); return

    # ── 5. Tính toán kết quả tổng hợp ──────────────────────────
    # Lấy trung bình cho từng khớp
    avg_per_keypoint = {pk: float(np.mean(err_list)) for pk, err_list in keypoint_errors.items()}
    
    # Tính các chỉ số tổng (từ trung bình của các khớp)
    all_keys_mean = float(np.mean(list(avg_per_keypoint.values())))
    
    al_vals = [v for k, v in avg_per_keypoint.items() if normalize_key(k) in ARM_LEG_KEYS]
    arm_leg_mean = float(np.mean(al_vals)) if al_vals else float("nan")
    
    rel_vals = [v for k, v in avg_per_keypoint.items() if normalize_key(k) in RELIABLE_KEYS]
    reliable_mean = float(np.mean(rel_vals)) if rel_vals else float("nan")

    total_calculated_frames = len(next(iter(keypoint_errors.values())))

    # ── 6. In Report ───────────────────────────────────────────
    print("\n" + "═" * 75)
    print("  CHI TIẾT MPJPE TRUNG BÌNH THEO TỪNG KHỚP (KEYPOINT)")
    print("═" * 75)
    
    # In danh sách khớp giảm dần theo độ lớn của lỗi (để dễ thấy khớp nào sai nhiều nhất)
    sorted_keys = sorted(avg_per_keypoint.items(), key=lambda x: x[1], reverse=True)
    for pk, err in sorted_keys:
        print(f"  > {pk:25s}: {err:>8.2f} mm")

    print("\n" + "═" * 75)
    print(f"  TỔNG KẾT (Trên {total_calculated_frames} frames hợp lệ)")
    print("═" * 75)
    if missing_frames:
        print(f"  [!] Bỏ qua {len(missing_frames)} frames bị thiếu Ground Truth.")
        
    print(f"  1. MPJPE Toàn thân (Tất cả) : {all_keys_mean:.2f} mm")
    print(f"  2. MPJPE Tay+Chân (12k)     : {arm_leg_mean:.2f} mm" if not np.isnan(arm_leg_mean) else "  2. MPJPE Tay+Chân : N/A")
    print(f"  3. MPJPE Uy tín (Vai+Hông)  : {reliable_mean:.2f} mm" if not np.isnan(reliable_mean) else "  3. MPJPE Uy tín   : N/A")
    print()

if __name__ == "__main__":
    main()