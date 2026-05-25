from __future__ import annotations

import copy
from pathlib import Path

import cv2
import pickle

from json_io import read_json, write_json
from hbh_pipeline.config import DEFAULT_OUTPUT_DIR
from hbh_pipeline.execute import (
    YOLO,
    TRIANGULATION_METHODS,
    build_projection_matrix,
    coco_conf_to_h36m,
    coco_to_h36m,
    detect_2d_pose,
)
from hbh_pipeline.logs import log_disabled, log_done, log_start, log_summary
from hbh_pipeline.evaluation import run_hbh_evaluation
from hbh_pipeline.visualization import run_hbh_visualization
from keypoints_map import get_smpl_joint_map


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
    return {
        "affine_intrinsics_matrix": data["intrinsics_cam"],
        "extrinsic_matrix": data["extrinsic_cam"],
        "xyz": data["xyz"],
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

    P1 = build_projection_matrix(_load_camera_params(cam1_profile))
    P2 = build_projection_matrix(_load_camera_params(cam2_profile))

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
