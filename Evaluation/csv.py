# -*- coding: utf-8 -*-
"""
Protocol 1 — Standard MPJPE Evaluation
========================================
So sánh fused_jsons/fused_data_{i}.json (i=1,2,...)
với truth/frame_{:05d}.json (id bắt đầu từ truth_start).

Cách dùng:
    python eval_mpjpe.py --cam camera1 --truth_start 9
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────
# KEY MAPPING: fused → truth
# ─────────────────────────────────────────────
FUSED_TO_TRUTH = {
    "left_hand"      : "left_wrist",
    "right_hand"     : "right_wrist",
    "left_shoulder"  : "left_shoulder",
    "right_shoulder" : "right_shoulder",
    "left_elbow"     : "left_elbow",
    "right_elbow"    : "right_elbow",
    "left_hip"       : "left_hip",
    "right_hip"      : "right_hip",
    "left_knee"      : "left_knee",
    "right_knee"     : "right_knee",
    "left_ankle"     : "left_ankle",
    "right_ankle"    : "right_ankle",
    "left_foot"      : "left_foot",
    "right_foot"     : "right_foot",
    "neck"           : "neck",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_fused(path: Path, cam: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cam_data = data.get("optimized", {}).get(cam, {})
    return {k: np.array(v, dtype=float) for k, v in cam_data.items()}


def load_truth(path: Path, annot_key: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    kps = data["annotations"][annot_key]["keypoints"]
    return {k: np.array(v, dtype=float) for k, v in kps.items()}


def compute_pelvis_fused(joints: dict) -> np.ndarray:
    return (joints["left_hip"] + joints["right_hip"]) / 2.0


def root_align(joints: dict, root: np.ndarray) -> dict:
    return {k: v - root for k, v in joints.items()}


def mpjpe_per_joint(fused_aligned: dict, truth_aligned: dict,
                    joint_pairs: list) -> dict:
    errors = {}
    for fk, tk in joint_pairs:
        if fk not in fused_aligned or tk not in truth_aligned:
            continue
        err_m = float(np.linalg.norm(fused_aligned[fk] - truth_aligned[tk]))
        errors[fk] = err_m * 1000.0  # m → mm
    return errors


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Protocol 1 MPJPE Evaluation")
    parser.add_argument("--fused_dir",   default="fused_jsons", type=str)
    parser.add_argument("--truth_dir",   default="truth",       type=str)
    parser.add_argument("--cam",         default="camera1",     choices=["camera1", "camera2"])
    parser.add_argument("--truth_start", default=1,             type=int)
    parser.add_argument("--annot_key",   default="annot3",      choices=["annot3", "univ_annot3"])
    args = parser.parse_args()

    fused_dir = Path(args.fused_dir)
    truth_dir = Path(args.truth_dir)

    if not fused_dir.exists():
        print(f"[ERR] Không tìm thấy fused_dir: {fused_dir}"); sys.exit(1)
    if not truth_dir.exists():
        print(f"[ERR] Không tìm thấy truth_dir: {truth_dir}"); sys.exit(1)

    fused_files = sorted(
        fused_dir.glob("fused_data_*.json"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    if not fused_files:
        print(f"[ERR] Không có fused_data_*.json trong {fused_dir}"); sys.exit(1)

    joint_pairs = list(FUSED_TO_TRUTH.items())

    # ── Header console ───────────────────────────────────────
    print(f"\n{'Protocol 1 — Standard MPJPE':^72}")
    print(f"  cam={args.cam}  annot={args.annot_key}  "
          f"fused={fused_dir}  truth={truth_dir}  truth_start={args.truth_start}")
    print()
    print(f"{'Frame':>6}  {'MPJPE(mm)':>10}  {'Top-3 worst joints (mm)':}")
    print("─" * 72)

    all_frame_mpjpe  = []
    all_joint_errors = {fk: [] for fk, _ in joint_pairs}
    missing_frames   = []
    # csv_rows: list of (frame_id, mpjpe, all_sorted)
    # all_sorted: [(joint_name, error_mm), ...] giảm dần
    csv_rows = []

    for fused_path in fused_files:
        fused_id   = int(fused_path.stem.split("_")[-1])
        truth_id   = args.truth_start + (fused_id - 1)
        truth_path = truth_dir / f"frame_{truth_id:05d}.json"

        if not truth_path.exists():
            missing_frames.append(fused_id)
            print(f"  [WARN] frame {fused_id}: truth {truth_path.name} không tồn tại → bỏ qua")
            continue

        try:
            fused_joints = load_fused(fused_path, args.cam)
            truth_joints = load_truth(truth_path, args.annot_key)
        except Exception as e:
            print(f"  [ERR] frame {fused_id}: {e} → bỏ qua")
            continue

        if "left_hip" not in fused_joints or "right_hip" not in fused_joints:
            print(f"  [WARN] frame {fused_id}: thiếu hip joints → bỏ qua")
            continue
        if "pelvis" not in truth_joints:
            print(f"  [WARN] frame {fused_id}: truth thiếu 'pelvis' → bỏ qua")
            continue

        fused_aligned = root_align(fused_joints, compute_pelvis_fused(fused_joints))
        truth_aligned = root_align(truth_joints, truth_joints["pelvis"])

        errors = mpjpe_per_joint(fused_aligned, truth_aligned, joint_pairs)
        if not errors:
            continue

        frame_mpjpe = float(np.mean(list(errors.values())))
        all_frame_mpjpe.append(frame_mpjpe)

        for fk in errors:
            all_joint_errors[fk].append(errors[fk])

        # Sắp xếp tất cả joint theo error giảm dần
        all_sorted = sorted(errors.items(), key=lambda x: x[1], reverse=True)

        # Console: chỉ in top 3
        top3_str = "  |  ".join(f"{jn}: {val:.1f}" for jn, val in all_sorted[:3])
        print(f"{fused_id:>6}  {frame_mpjpe:>10.2f}  {top3_str}")

        csv_rows.append((fused_id, frame_mpjpe, all_sorted))

    # ── Ghi CSV ───────────────────────────────────────────────
    if csv_rows:
        csv_path = Path(f"mpjpe_{args.cam}.csv")

        # Thứ tự cột: joint của frame đầu tiên sorted giảm dần
        # (tên joint cố định, chỉ thứ tự trong frame đầu làm chuẩn)
        joint_col_names = [jn for jn, _ in csv_rows[0][2]]

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            cf.write("sep=,\n")  # Excel tự nhận dấu phẩy
            writer = csv.writer(cf)
            writer.writerow(["frame", "MPJPE_mm"] + joint_col_names)
            for frame_id, mpjpe, all_sorted in csv_rows:
                val_map = {jn: val for jn, val in all_sorted}
                row = [frame_id, f"{mpjpe:.2f}"] + [
                    f"{val_map[jn]:.2f}" if jn in val_map else ""
                    for jn in joint_col_names
                ]
                writer.writerow(row)

        print(f"\n  CSV đã lưu: {csv_path.resolve()}")

    # ── Tổng kết ──────────────────────────────────────────────
    if not all_frame_mpjpe:
        print("\n[ERR] Không có frame nào được tính.")
        return

    print("\n" + "═" * 72)
    print(f"  Frames đã tính : {len(all_frame_mpjpe)}"
          + (f"  |  Bỏ qua: {missing_frames}" if missing_frames else ""))
    print(f"  MPJPE mean     : {np.mean(all_frame_mpjpe):.2f} mm")
    print(f"  MPJPE median   : {np.median(all_frame_mpjpe):.2f} mm")
    print(f"  MPJPE min/max  : {np.min(all_frame_mpjpe):.2f} / {np.max(all_frame_mpjpe):.2f} mm")

    print(f"\n  {'Joint':<20} {'Mean':>8}  {'Max':>8}  {'Min':>8}  (mm)")
    print("  " + "─" * 48)

    joint_means = {
        fk: (np.mean(v), np.max(v), np.min(v))
        for fk, v in all_joint_errors.items() if v
    }
    for fk, (mean_e, max_e, min_e) in sorted(joint_means.items(),
                                               key=lambda x: x[1][0], reverse=True):
        print(f"  {fk:<20} {mean_e:>8.2f}  {max_e:>8.2f}  {min_e:>8.2f}")
    print()


if __name__ == "__main__":
    main()