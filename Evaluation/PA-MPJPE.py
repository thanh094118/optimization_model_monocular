# -*- coding: utf-8 -*-
"""
Protocol 1 — Universal PA-MPJPE Evaluation (Procrustes Aligned)
Hỗ trợ nhiều định dạng: fused_data, pose_data, openpose_25...
"""

import json
import sys
import re
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes

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
    "Nose": "head", "REye": "head", "LEye": "head" # Approx nếu cần
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
    """Trích xuất số từ tên file (vd: pose_data_12.json -> 12, 000000.json -> 0)"""
    nums = re.findall(r'\d+', path.stem)
    return int(nums[-1]) if nums else -1

def _find_keypoint_dicts(node, path: str = "") -> list[tuple[str, dict]]:
    results = []
    if isinstance(node, dict):
        vals = list(node.values())
        # Nếu dict chứa toàn list/tuple 3 số -> đây là keypoint dict
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
    """Tự động dò tìm cấu trúc JSON để tìm nơi chứa tọa độ 3D"""
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
    if annot_path:  # Nếu path rỗng thì data chính là node
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
    
    # Tạo từ điển map từ tên chuẩn hóa sang tên gốc
    truth_norm_to_raw = {normalize_key(tk): tk for tk in truth_keys}
    
    mapping = []
    lines = ["", "── LOG MAPPING KEY3D " + "─" * 52]
    lines.append(f"  {'Pred key (Gốc)':18s} {'Truth key (Gốc)':18s} {'Map qua':15s} Trạng thái")
    lines.append("  " + "─" * 68)

    for pk in pred_keys:
        pk_norm = normalize_key(pk)
        
        # 1. Khớp chính xác (sau khi chuẩn hóa tên)
        if pk_norm in truth_norm_to_raw:
            tk_raw = truth_norm_to_raw[pk_norm]
            mapping.append((pk, tk_raw))
            method = "exact/alias" if pk != tk_raw else "exact"
            lines.append(f"  {pk:18s} {tk_raw:18s} {method:15s} ✓")
            
        # 2. Khớp qua vị trí trong STANDARD_KEYS
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

    # Centering và tính tỉ lệ thật
    truth_mean = np.mean(truth_matrix, axis=0)
    truth_centered = truth_matrix - truth_mean
    norm_truth = np.linalg.norm(truth_centered) 
    
    # Procrustes Alignment
    mtx1, mtx2, disparity = procrustes(truth_matrix, pred_matrix)

    # Đưa về kích thước mét và tính lỗi mm
    mtx1_real = mtx1 * norm_truth
    mtx2_real = mtx2 * norm_truth
    distances = np.linalg.norm(mtx1_real - mtx2_real, axis=1) * 1000.0

    return {pk: float(dist) for (pk, _), dist in zip(valid_pairs, distances)}

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logs: list[str] = []

    print("\n" + "═" * 75)
    print("  PROTOCOL 1 — UNIVERSAL PA-MPJPE EVALUATION (PROCRUSTES)")
    print("═" * 75)

    # ── 1. Dữ liệu Dự đoán (Predicted) ─────────────────────────
    pred_dir = Path(_ask("Thư mục dữ liệu Dự đoán (fused/pose/keypoints)", "pred_jsons"))
    if not pred_dir.exists():
        print(f"[ERR] Không tìm thấy: {pred_dir}"); sys.exit(1)

    pred_files = sorted(
        [p for p in pred_dir.glob("*.json") if extract_frame_id(p) != -1],
        key=extract_frame_id
    )
    if not pred_files:
        print(f"[ERR] Không có file JSON hợp lệ trong {pred_dir}"); sys.exit(1)

    pred_annot_path = resolve_annot_path(pred_files[0], "DỮ LIỆU DỰ ĐOÁN")
    pred_sample = load_points_by_path(pred_files[0], pred_annot_path)
    pred_keys = sorted(pred_sample.keys())
    print(f"  > Số lượng file: {len(pred_files)}")

    # ── 2. Dữ liệu Thực tế (Ground Truth) ──────────────────────
    print("\n" + "─" * 40)
    truth_dir = Path(_ask("Thư mục dữ liệu Thực tế (Ground Truth)", "truth_jsons"))
    if not truth_dir.exists():
        print(f"[ERR] Không tìm thấy: {truth_dir}"); sys.exit(1)

    truth_files = sorted(
        [p for p in truth_dir.glob("*.json") if extract_frame_id(p) != -1],
        key=extract_frame_id
    )
    if not truth_files:
        print(f"[ERR] Không có file JSON hợp lệ trong {truth_dir}"); sys.exit(1)

    truth_start = _ask_int("truth_start (Index frame đầu tiên của truth)", extract_frame_id(truth_files[0]))
    truth_annot_path = resolve_annot_path(truth_files[0], "GROUND TRUTH")
    truth_sample = load_points_by_path(truth_files[0], truth_annot_path)
    truth_keys = sorted(truth_sample.keys())
    print(f"  > Số lượng file: {len(truth_files)}")

    # ── Kiểm tra và Map key ────────────────────────────────────
    joint_pairs = auto_map_keys(pred_keys, truth_keys, logs)
    if not joint_pairs:
        print("[ERR] Không có cặp key nào được map."); sys.exit(1)

    mapped_pk = [pk for pk, _ in joint_pairs]
    extra_raw = _ask("\n  Key tính PA-MPJPE thêm (cách bởi dấu phẩy, Enter bỏ qua)", "")
    extra_keys = [k.strip() for k in extra_raw.split(",") if k.strip() and k.strip() in mapped_pk]

    # ── Xử lý Output ───────────────────────────────────────────
    header = f"\n{'Frame':>6}  {'Toàn thân':>12}  {'Tay+Chân(12k)':>15}  {'Uy tín(Vai+Hông)':>18}"
    if extra_keys: header += f"  {'Extra':>12}"
    print(header)
    print("─" * (57 + (14 if extra_keys else 0)))

    all_frame_mpjpe = []
    all_frame_arm_leg = []
    all_frame_reliable = []
    all_frame_extra = []
    missing_frames = []

    for pred_path in pred_files:
        pred_id = extract_frame_id(pred_path)
        truth_id = truth_start + (pred_id - extract_frame_id(pred_files[0]))
        
        # Tìm file truth có số thứ tự tương ứng
        truth_path_matches = [p for p in truth_files if extract_frame_id(p) == truth_id]
        
        if not truth_path_matches:
            missing_frames.append(pred_id)
            logs.append(f"[WARN] Pred {pred_id}: Không tìm thấy truth tương ứng → bỏ qua")
            continue

        try:
            pred_joints = load_points_by_path(pred_path, pred_annot_path)
            truth_joints = load_points_by_path(truth_path_matches[0], truth_annot_path)
            errors = compute_pa_mpjpe(pred_joints, truth_joints, joint_pairs)
        except Exception as e:
            logs.append(f"[ERR] Frame {pred_id}: {e} → bỏ qua")
            continue

        if not errors: continue

        # Tính toán theo nhóm (chú ý: phải kiểm tra tên gốc đã normalize)
        frame_all = float(np.mean(list(errors.values())))
        
        al_vals = [v for k, v in errors.items() if normalize_key(k) in ARM_LEG_KEYS]
        frame_arm_leg = float(np.mean(al_vals)) if al_vals else float("nan")
        
        rel_vals = [v for k, v in errors.items() if normalize_key(k) in RELIABLE_KEYS]
        frame_reliable = float(np.mean(rel_vals)) if rel_vals else float("nan")
        
        ex_vals = [errors[k] for k in extra_keys if k in errors]
        frame_extra = float(np.mean(ex_vals)) if ex_vals else float("nan")

        all_frame_mpjpe.append(frame_all)
        all_frame_arm_leg.append(frame_arm_leg)
        all_frame_reliable.append(frame_reliable)
        all_frame_extra.append(frame_extra)

        al_str = f"{frame_arm_leg:.2f}" if not np.isnan(frame_arm_leg) else "N/A"
        rel_str = f"{frame_reliable:.2f}" if not np.isnan(frame_reliable) else "N/A"
        ex_str = f"{frame_extra:.2f}" if not np.isnan(frame_extra) else "N/A"
        
        line = f"{pred_id:>6}  {frame_all:>12.2f}  {al_str:>15}  {rel_str:>18}"
        if extra_keys: line += f"  {ex_str:>12}"
        print(line)

    # ── Tổng kết ─────────────────────────────────────────────
    if not all_frame_mpjpe:
        print("\n[ERR] Không có frame nào được tính."); return

    val_al = [x for x in all_frame_arm_leg if not np.isnan(x)]
    val_rel = [x for x in all_frame_reliable if not np.isnan(x)]
    val_ex = [x for x in all_frame_extra if not np.isnan(x)]

    print("\n" + "═" * 75)
    print(f"  Frames đã tính : {len(all_frame_mpjpe)}" + (f"  |  Bỏ qua: {missing_frames}" if missing_frames else ""))
    print()
    print(f"  PA-MPJPE Toàn thân        : {np.mean(all_frame_mpjpe):.2f} mm")
    print(f"  PA-MPJPE Tay+Chân (12k)   : {np.mean(val_al):.2f} mm" if val_al else "  PA-MPJPE Tay+Chân : N/A")
    print(f"  PA-MPJPE Uy tín (Vai+Hông): {np.mean(val_rel):.2f} mm" if val_rel else "  PA-MPJPE Uy tín   : N/A")
    if extra_keys and val_ex:
        print(f"  PA-MPJPE Extra [{','.join(extra_keys)}] : {np.mean(val_ex):.2f} mm")
    print()

if __name__ == "__main__":
    main()
