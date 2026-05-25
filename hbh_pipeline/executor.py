from __future__ import annotations

import copy
from pathlib import Path

import cv2
import pickle
import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
from ultralytics import YOLO

from json_io import read_json, write_json
from hbh_pipeline.config import DEFAULT_OUTPUT_DIR
from hbh_pipeline.logs import log_disabled, log_done, log_start, log_summary
from hbh_pipeline.evaluation import run_hbh_evaluation
from hbh_pipeline.visualization import run_hbh_visualization
from keypoints_map import get_smpl_joint_map

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


def _clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method_dir in output_dir.glob("method_*"):
        if not method_dir.is_dir():
            continue
        for old in method_dir.glob("hbh_data_*.json"):
            old.unlink()
    for old in output_dir.glob("recon_*.pkl"):
        old.unlink()


def _load_camera_params(profile_path: Path) -> dict:
    data = read_json(profile_path)
    tvec = data.get("tvec", data.get("xyz"))
    if tvec is None:
        raise KeyError(f"Missing 'tvec' in {profile_path} (legacy fallback 'xyz' also missing)")
    return {
        "affine_intrinsics_matrix": data["intrinsics_cam"],
        "extrinsic_matrix": data["extrinsic_cam"],
        "tvec": tvec,
    }


def _build_projection_matrix(cam: dict, rotation_mode: str = "raw") -> np.ndarray:
    fx = cam["affine_intrinsics_matrix"][0][0]
    fy = cam["affine_intrinsics_matrix"][1][1]
    cx = cam["affine_intrinsics_matrix"][0][2]
    cy = cam["affine_intrinsics_matrix"][1][2]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    R = np.array(cam["extrinsic_matrix"], dtype=float)
    t = np.array(cam["tvec"], dtype=float)

    mode = str(rotation_mode).strip().lower()
    if mode == "raw":
        pass
    elif mode == "flip_rows_yz":
        R[1:, :] *= -1.0
    elif mode == "flip_cols_yz":
        R[:, 1:] *= -1.0
    else:
        raise ValueError("hbh.rotation_mode must be one of: raw, flip_rows_yz, flip_cols_yz")

    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def detect_2d_pose(model, frame):
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
    A = np.array([x1 * P1[2] - P1[0], y1 * P1[2] - P1[1], x2 * P2[2] - P2[0], y2 * P2[2] - P2[1]])
    _, _, vt = svd(A)
    X = vt[-1]
    return X[:3] / X[3]


def _reproject(P, pt3d):
    h = P @ np.append(pt3d, 1.0)
    return h[:2] / h[2]


def triangulate_dlt(P1, P2, pts1, pts2, *_):
    return np.array([_dlt_single(P1, P2, pts1[i], pts2[i]) for i in range(len(pts1))])


def triangulate_conf_algebraic(P1, P2, pts1, pts2, conf1, conf2):
    n = len(pts1)
    pts_3d = np.zeros((n, 3), dtype=float)
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        w1 = np.clip(conf1[i], 0.01, 1.0) ** 2
        w2 = np.clip(conf2[i], 0.01, 1.0) ** 2
        A = np.array([w1 * (x1 * P1[2] - P1[0]), w1 * (y1 * P1[2] - P1[1]), w2 * (x2 * P2[2] - P2[0]), w2 * (y2 * P2[2] - P2[1])])
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
        A = np.array([w1 * (pts1[i, 0] * P1[2] - P1[0]), w1 * (pts1[i, 1] * P1[2] - P1[1]), w2 * (pts2[i, 0] * P2[2] - P2[0]), w2 * (pts2[i, 1] * P2[2] - P2[1])])
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
    "DLT (baseline)": triangulate_dlt,
    "Conf-Algebraic": triangulate_conf_algebraic,
    "RANSAC-DLT": triangulate_ransac,
    "Iterative Refine": triangulate_iterative,
    "Anatomical (SOTA)": triangulate_anatomical,
}


H36M_TO_SYSTEM = {
    "Pelvis": "pelvis",
    "R_Hip": "right_hip",
    "R_Knee": "right_knee",
    "R_Ankle": "right_ankle",
    "L_Hip": "left_hip",
    "L_Knee": "left_knee",
    "L_Ankle": "left_ankle",
    "Spine": "spine1",
    "Thorax": "spine3",
    "Neck": "neck",
    "Head": "head",
    "L_Shoulder": "left_shoulder",
    "L_Elbow": "left_elbow",
    "L_Wrist": "left_hand",
    "R_Shoulder": "right_shoulder",
    "R_Elbow": "right_elbow",
    "R_Wrist": "right_hand",
}


def run_hbh(config: dict) -> None:
    hbh_cfg = config.get("hbh", {})
    if not hbh_cfg.get("enabled", False):
        log_disabled()
        return

    paths = config.get("paths", {})
    runtime_cfg = config.get("runtime", {})

    video1 = Path(paths["camera1_video"])
    video2 = Path(paths["camera2_video"])
    if not video1.exists() or not video2.exists():
        raise FileNotFoundError(f"HBH video input missing: {video1} | {video2}")

    calib_out = Path(config.get("preprocess", {}).get("calibration", {}).get("output_dir", "output/preprocess_results"))
    cam1_profile = calib_out / "data_cam1.json"
    cam2_profile = calib_out / "data_cam2.json"
    if not cam1_profile.exists() or not cam2_profile.exists():
        raise FileNotFoundError(f"HBH camera profile missing: {cam1_profile} | {cam2_profile}")

    # Ground truth input is mapped consistently for later evaluation hooks.
    gt_dir = Path(config.get("evaluation", {}).get("ground_truth_dir", "input/gtruth_results"))
    if not gt_dir.exists():
        print(f"[HBH] WARNING: ground truth dir not found: {gt_dir}")

    output_dir = Path(paths.get("hbh_output_dir", DEFAULT_OUTPUT_DIR))
    if runtime_cfg.get("clean_output", True):
        _clean_output(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    log_start(str(video1), str(video2), str(output_dir))

    model_name = hbh_cfg.get("model", "yolo26x-pose.pt")
    pose_model = YOLO(model_name)
    rotation_mode = str(hbh_cfg.get("rotation_mode", "raw"))

    P1 = _build_projection_matrix(_load_camera_params(cam1_profile), rotation_mode=rotation_mode)
    P2 = _build_projection_matrix(_load_camera_params(cam2_profile), rotation_mode=rotation_mode)

    cap1 = cv2.VideoCapture(str(video1))
    cap2 = cv2.VideoCapture(str(video2))
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    n2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(n1, n2)

    max_frames = hbh_cfg.get("max_frames")
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    conf_threshold = float(hbh_cfg.get("conf_threshold", 0.3))
    min_valid_joints = int(hbh_cfg.get("min_valid_joints", 8))

    results = []
    primary_method = hbh_cfg.get("primary_method", "Anatomical (SOTA)")
    run_all_methods = str(primary_method).strip().lower() == "all"
    if not run_all_methods and primary_method not in TRIANGULATION_METHODS:
        raise ValueError(f"Invalid hbh.primary_method={primary_method!r}")
    selected_methods = list(TRIANGULATION_METHODS.keys()) if run_all_methods else [primary_method]

    smpl_joint_names = set(get_smpl_joint_map().keys())
    if not smpl_joint_names:
        raise ValueError("SMPL joint map is empty")

    for frame_idx in range(frame_count):
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read()
        if not ok1 or not ok2:
            break

        kps1_coco, conf1 = detect_2d_pose(pose_model, frame1)
        kps2_coco, conf2 = detect_2d_pose(pose_model, frame2)
        if kps1_coco is None or kps2_coco is None:
            continue
        if len(kps1_coco) != 17 or len(kps2_coco) != 17:
            continue
        valid = (conf1 > conf_threshold) & (conf2 > conf_threshold)
        if int(valid.sum()) < min_valid_joints:
            continue

        kps1_h36m = coco_to_h36m(kps1_coco)
        kps2_h36m = coco_to_h36m(kps2_coco)
        conf1_h36m = coco_conf_to_h36m(conf1)
        conf2_h36m = coco_conf_to_h36m(conf2)

        methods = {}
        for method_name in selected_methods:
            tri_fn = TRIANGULATION_METHODS[method_name]
            recon = tri_fn(P1, P2, kps1_h36m, kps2_h36m, conf1_h36m, conf2_h36m)
            methods[method_name] = {"recon_3d": recon.tolist()}

        method_joint_maps = {}
        for method_name, method_data in methods.items():
            method_joint_map = {}
            for joint_name_h36m, xyz in zip(
                [
                    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
                    "Spine", "Thorax", "Neck", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
                    "R_Shoulder", "R_Elbow", "R_Wrist",
                ],
                method_data["recon_3d"],
            ):
                system_name = H36M_TO_SYSTEM[joint_name_h36m]
                if system_name in smpl_joint_names:
                    method_joint_map[system_name] = xyz
            method_joint_maps[method_name] = method_joint_map

        out_name = f"hbh_data_{frame_idx + 1}.json"
        for method_name in selected_methods:
            method_dir = output_dir / f"method_{method_name}"
            method_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                method_dir / out_name,
                {
                    "keypoints3d": {
                        "camera1": method_joint_maps[method_name],
                        "camera2": copy.deepcopy(method_joint_maps[method_name]),
                    },
                    "metadata": {
                        "source": "hbh_pipeline",
                        "camera_ids": ["cam1", "cam2"],
                        "primary_method": primary_method,
                        "key3d_method": method_name,
                        "frame_index": frame_idx + 1,
                    },
                },
            )

        results.append({
            "frame": frame_idx + 1,
            "camera_ids": ["cam1", "cam2"],
            "selected_methods": selected_methods,
            "all_methods": methods,
        })

    cap1.release()
    cap2.release()

    with (output_dir / "recon_results.pkl").open("wb") as f:
        pickle.dump(results, f)
    log_summary(
        num_results=len(results),
        primary_method=primary_method,
        selected_methods=selected_methods,
        output_dir=str(output_dir),
        video1=str(video1),
        video2=str(video2),
        cam_profiles=[str(cam1_profile), str(cam2_profile)],
        ground_truth_dir=str(gt_dir),
    )

    log_done(str(output_dir), len(results))
    run_hbh_evaluation(config)
    run_hbh_visualization(config)
