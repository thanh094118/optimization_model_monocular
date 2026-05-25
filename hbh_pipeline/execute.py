#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
execute.py

Thiết kế lại từ notebook Colab gốc nhưng GIỮ logic phần execute.

Nhiệm vụ:
1. Scan dataset và tìm multi-camera pairs.
2. Load YOLO pose model.
3. Extract frame từ 2 camera.
4. Detect 2D COCO keypoints.
5. Convert COCO 17 -> H36M 17.
6. Chạy ĐỦ 5 triangulation methods như Colab gốc.
7. Lưu:
   - all_methods: kết quả 3D của cả 5 method
   - recon_3d: kết quả primary = Anatomical (SOTA)

Không tính MPJPE/PA-MPJPE ở đây vì phần đó được chuyển sang evaluation.py.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
from tqdm import tqdm
from ultralytics import YOLO

COCO_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]

H36M_JOINT_NAMES = [
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
    "Spine", "Thorax", "Neck", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
]

COCO_TO_H36M = {
    0: [11, 12], 1: [12], 2: [14], 3: [16],
    4: [11], 5: [13], 6: [15],
    7: [5, 6, 11, 12], 8: [5, 6], 9: [0], 10: [0],
    11: [5], 12: [7], 13: [9], 14: [6], 15: [8], 16: [10],
}

H36M_BONES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

H36M_BONE_LENGTHS = {
    (0, 1): 132, (1, 2): 442, (2, 3): 430,
    (0, 4): 132, (4, 5): 442, (5, 6): 430,
    (0, 7): 233, (7, 8): 257, (8, 9): 121, (9, 10): 115,
    (8, 11): 151, (11, 12): 278, (12, 13): 251,
    (8, 14): 151, (14, 15): 278, (15, 16): 251,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_dataset(data_dir):
    results = {}
    for split in ["train_set", "valid_set", "test_set"]:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
        results[split] = {}
        for subject in sorted(os.listdir(split_dir)):
            subject_dir = split_dir / subject
            if not subject_dir.is_dir():
                continue
            mp4s = [f for f in os.listdir(subject_dir) if f.endswith(".mp4")]
            motions = defaultdict(list)
            for filename in mp4s:
                base = filename.replace(".mp4", "")
                parts = base.rsplit("_cam_", 1)
                if len(parts) == 2:
                    motions[parts[0]].append({
                        "cam_id": parts[1],
                        "base": base,
                        "path": str(subject_dir / base),
                    })
            results[split][subject] = dict(motions)
    return results


def find_multicam_pairs(dataset_info):
    pairs = []
    for split, subjects in dataset_info.items():
        for subject, motions in subjects.items():
            for motion_name, cams in motions.items():
                if len(cams) >= 2:
                    pairs.append({
                        "split": split,
                        "subject": subject,
                        "motion": motion_name,
                        "cameras": cams,
                        "n_cams": len(cams),
                    })
    return pairs


def build_projection_matrix(cam):
    fx = cam["affine_intrinsics_matrix"][0][0]
    fy = cam["affine_intrinsics_matrix"][1][1]
    cx = cam["affine_intrinsics_matrix"][0][2]
    cy = cam["affine_intrinsics_matrix"][1][2]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    R = np.array(cam["extrinsic_matrix"], dtype=float)
    R[1:, :] *= -1
    t = np.array(cam["xyz"], dtype=float)
    Rt = np.hstack([R, (-R @ t).reshape(3, 1)])
    return K @ Rt


def detect_2d_pose(model, frame, conf_thresh=0.3):
    results = model(frame, verbose=False)
    if len(results) == 0 or results[0].keypoints is None:
        return None, None
    keypoints = results[0].keypoints
    if keypoints.xy is None or len(keypoints.xy) == 0:
        return None, None
    if keypoints.conf is not None and len(keypoints.conf) > 0:
        best_idx = keypoints.conf.mean(dim=1).argmax().item()
    else:
        best_idx = 0
    xy = keypoints.xy[best_idx].cpu().numpy()
    conf = keypoints.conf[best_idx].cpu().numpy() if keypoints.conf is not None else np.ones(17)
    return xy, conf


def extract_frame(video_path, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def coco_to_h36m(coco_kps):
    h36m = np.zeros((17, coco_kps.shape[1]), dtype=float)
    for h_idx, c_idxs in COCO_TO_H36M.items():
        h36m[h_idx] = np.mean(coco_kps[c_idxs], axis=0)
    return h36m


def coco_conf_to_h36m(conf):
    h36m_conf = np.zeros(17, dtype=float)
    for h_idx, c_idxs in COCO_TO_H36M.items():
        h36m_conf[h_idx] = np.mean(conf[c_idxs])
    return h36m_conf


def _dlt_single(P1, P2, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1],
    ])
    _, _, vt = svd(A)
    X = vt[-1]
    return X[:3] / X[3]


def _reproject(P, pt3d):
    h = P @ np.append(pt3d, 1.0)
    return h[:2] / h[2]


def triangulate_dlt(P1, P2, pts1, pts2):
    return np.array([_dlt_single(P1, P2, pts1[i], pts2[i]) for i in range(len(pts1))])


def triangulate_conf_algebraic(P1, P2, pts1, pts2, conf1, conf2):
    n = len(pts1)
    pts_3d = np.zeros((n, 3), dtype=float)
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        w1 = np.clip(conf1[i], 0.01, 1.0) ** 2
        w2 = np.clip(conf2[i], 0.01, 1.0) ** 2
        A = np.array([
            w1 * (x1 * P1[2] - P1[0]),
            w1 * (y1 * P1[2] - P1[1]),
            w2 * (x2 * P2[2] - P2[0]),
            w2 * (y2 * P2[2] - P2[1]),
        ])
        _, _, vt = svd(A)
        X = vt[-1]
        pts_3d[i] = X[:3] / X[3]
    return pts_3d


def triangulate_ransac(P1, P2, pts1, pts2, conf1, conf2, reproj_thresh=15.0):
    n = len(pts1)
    pts_3d = np.zeros((n, 3), dtype=float)
    for i in range(n):
        pt3d = _dlt_single(P1, P2, pts1[i], pts2[i])
        err1 = np.linalg.norm(_reproject(P1, pt3d) - pts1[i])
        err2 = np.linalg.norm(_reproject(P2, pt3d) - pts2[i])
        if err1 > reproj_thresh and err2 < reproj_thresh:
            w1, w2 = 0.1, 1.0
        elif err2 > reproj_thresh and err1 < reproj_thresh:
            w1, w2 = 1.0, 0.1
        elif err1 > reproj_thresh and err2 > reproj_thresh:
            w1, w2 = conf1[i] ** 2, conf2[i] ** 2
        else:
            w1, w2 = conf1[i], conf2[i]
        A = np.array([
            w1 * (pts1[i, 0] * P1[2] - P1[0]),
            w1 * (pts1[i, 1] * P1[2] - P1[1]),
            w2 * (pts2[i, 0] * P2[2] - P2[0]),
            w2 * (pts2[i, 1] * P2[2] - P2[1]),
        ])
        _, _, vt = svd(A)
        X = vt[-1]
        pts_3d[i] = X[:3] / X[3]
    return pts_3d


def triangulate_iterative(P1, P2, pts1, pts2, conf1, conf2, n_iter=50):
    n = len(pts1)
    pts_3d = np.zeros((n, 3), dtype=float)
    for i in range(n):
        x0 = _dlt_single(P1, P2, pts1[i], pts2[i])
        w1 = np.clip(conf1[i], 0.1, 1.0)
        w2 = np.clip(conf2[i], 0.1, 1.0)
        def residuals(X):
            r1 = _reproject(P1, X) - pts1[i]
            r2 = _reproject(P2, X) - pts2[i]
            return np.concatenate([w1 * r1, w2 * r2])
        result = least_squares(residuals, x0, method="lm", max_nfev=n_iter)
        pts_3d[i] = result.x
    return pts_3d


def triangulate_anatomical(P1, P2, pts1, pts2, conf1, conf2, bone_weight=0.01, n_iter=80):
    init_3d = triangulate_iterative(P1, P2, pts1, pts2, conf1, conf2, n_iter=30)
    ref_bones = {}
    for j1, j2 in H36M_BONES:
        ref_bones[(j1, j2)] = H36M_BONE_LENGTHS.get((j1, j2), np.linalg.norm(init_3d[j1] - init_3d[j2]))
    x0 = init_3d.flatten()
    def full_residuals(X):
        pts = X.reshape(17, 3)
        res = []
        for i in range(17):
            w1 = np.clip(conf1[i], 0.1, 1.0)
            w2 = np.clip(conf2[i], 0.1, 1.0)
            r1 = _reproject(P1, pts[i]) - pts1[i]
            r2 = _reproject(P2, pts[i]) - pts2[i]
            res.extend([w1 * r1[0], w1 * r1[1], w2 * r2[0], w2 * r2[1]])
        for j1, j2 in H36M_BONES:
            bone_len = np.linalg.norm(pts[j1] - pts[j2])
            target_len = ref_bones[(j1, j2)]
            if target_len > 0:
                res.append(bone_weight * (bone_len - target_len))
        return np.array(res)
    result = least_squares(full_residuals, x0, method="trf", max_nfev=n_iter)
    return result.x.reshape(17, 3)


TRIANGULATION_METHODS = {
    "DLT (baseline)": lambda P1, P2, p1, p2, c1, c2: triangulate_dlt(P1, P2, p1, p2),
    "Conf-Algebraic": triangulate_conf_algebraic,
    "RANSAC-DLT": triangulate_ransac,
    "Iterative Refine": triangulate_iterative,
    "Anatomical (SOTA)": triangulate_anatomical,
}


def run_sota_execute(multicam_pairs, cam_params, pose_model, max_pairs=15, frames_per_pair=5):
    results = []
    for pair_info in tqdm(multicam_pairs[:max_pairs], desc="Processing pairs"):
        cam_a = pair_info["cameras"][0]
        cam_b = pair_info["cameras"][1]
        vid_a = cam_a["path"] + ".mp4"
        vid_b = cam_b["path"] + ".mp4"
        json_a = cam_a["path"] + ".json"
        json_b = cam_b["path"] + ".json"
        npy_a = cam_a["path"] + ".npy"

        # Giữ logic Colab: pair phải có video/json/npy vì pipeline gốc dùng GT cùng vòng lặp.
        if not all(os.path.exists(f) for f in [vid_a, vid_b, json_a, json_b, npy_a]):
            continue

        meta_a = load_json(json_a)
        meta_b = load_json(json_b)
        cp_a = cam_params.get(meta_a["cam"])
        cp_b = cam_params.get(meta_b["cam"])
        if cp_a is None or cp_b is None:
            continue
        P1 = build_projection_matrix(cp_a)
        P2 = build_projection_matrix(cp_b)

        raw_data = np.load(npy_a, allow_pickle=True)
        n_frames = min(get_video_length(vid_a), get_video_length(vid_b), len(raw_data))
        if n_frames < 1:
            continue
        frame_indices = np.linspace(0, n_frames - 1, frames_per_pair, dtype=int)

        for frame_idx in frame_indices:
            frame_a = extract_frame(vid_a, int(frame_idx))
            frame_b = extract_frame(vid_b, int(frame_idx))
            if frame_a is None or frame_b is None:
                continue
            kps2d_a_coco, conf_a = detect_2d_pose(pose_model, frame_a)
            kps2d_b_coco, conf_b = detect_2d_pose(pose_model, frame_b)
            if kps2d_a_coco is None or kps2d_b_coco is None:
                continue
            if len(kps2d_a_coco) != 17 or len(kps2d_b_coco) != 17:
                continue
            valid = (conf_a > 0.3) & (conf_b > 0.3)
            if valid.sum() < 8:
                continue

            kps2d_a_h36m = coco_to_h36m(kps2d_a_coco)
            kps2d_b_h36m = coco_to_h36m(kps2d_b_coco)
            conf_a_h36m = coco_conf_to_h36m(conf_a)
            conf_b_h36m = coco_conf_to_h36m(conf_b)

            method_results = {}
            for method_name, tri_fn in TRIANGULATION_METHODS.items():
                recon = tri_fn(P1, P2, kps2d_a_h36m, kps2d_b_h36m, conf_a_h36m, conf_b_h36m)
                method_results[method_name] = {"recon_3d": recon}

            best_method = "Anatomical (SOTA)"
            best = method_results[best_method]
            results.append({
                "motion": pair_info["motion"],
                "subject": pair_info["subject"],
                "split": pair_info["split"],
                "cam_a": cam_a["cam_id"],
                "cam_b": cam_b["cam_id"],
                "video_a": vid_a,
                "video_b": vid_b,
                "json_a": json_a,
                "json_b": json_b,
                "npy_a": npy_a,
                "frame": int(frame_idx),
                "recon_3d": best["recon_3d"],
                "kps2d_a": kps2d_a_coco,
                "kps2d_b": kps2d_b_coco,
                "conf_a": conf_a,
                "conf_b": conf_b,
                "all_methods": method_results,
            })
    return results


def save_execute_outputs(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "recon_results.pkl"
    npz_path = out_dir / "recon_joints3d.npz"
    summary_path = out_dir / "recon_summary.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    joints3d = np.stack([r["recon_3d"] for r in results], axis=0) if results else np.empty((0, 17, 3), dtype=float)
    np.savez_compressed(npz_path, joints3d=joints3d, joint_names=np.array(H36M_JOINT_NAMES))
    summary = {
        "num_results": len(results),
        "primary_method": "Anatomical (SOTA)",
        "methods": list(TRIANGULATION_METHODS.keys()),
        "output_shape": list(joints3d.shape),
        "pkl": str(pkl_path),
        "npz": str(npz_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n[EXECUTE] Done.")
    print(f"  Results PKL: {pkl_path}")
    print(f"  Primary joints NPZ: {npz_path}")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Primary joints shape: {joints3d.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/data_athe/unzipped")
    parser.add_argument("--model", default="yolo26x-pose.pt")
    parser.add_argument("--max-pairs", type=int, default=15)
    parser.add_argument("--frames-per-pair", type=int, default=5)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    data_dir = base_dir / "data" / "data"
    cam_param_path = base_dir / "cam_param.json"
    out_dir = base_dir / "outputs" / "joints3d"
    print(f"[EXECUTE] DATA_DIR: {data_dir}")
    print(f"[EXECUTE] CAM_PARAM: {cam_param_path}")

    cam_params = load_json(cam_param_path)
    print(f"[EXECUTE] Loading YOLO model: {args.model}")
    pose_model = YOLO(args.model)

    dataset_info = scan_dataset(data_dir)
    multicam_pairs = find_multicam_pairs(dataset_info)
    print(f"[EXECUTE] Total multi-camera pairs: {len(multicam_pairs)}")
    for p in multicam_pairs[:5]:
        cids = [c["cam_id"] for c in p["cameras"]]
        print(f"  {p['split']}/{p['subject']}/{p['motion']} -> cams: {cids}")

    results = run_sota_execute(multicam_pairs, cam_params, pose_model, max_pairs=args.max_pairs, frames_per_pair=args.frames_per_pair)
    save_execute_outputs(results, out_dir)


if __name__ == "__main__":
    main()
