import numpy as np
from pathlib import Path
import torch
import datetime
from typing import Optional

from json_io import read_json, write_json
from optimization_pipeline.pipeline_monolithic import (
    SmplBody25Model, run_phase2_init_translation, smooth_poses,
    to_tensor_dict, optimize_phase5, optimize_phase6
)
from keypoints_map import load_keypoints3d_map


def _load_fusion_confidence_for_frame(fused_metadata_dir: Path, frame_idx: int, cam_id: str) -> dict:
    metadata_path = fused_metadata_dir / f"fused_data_{frame_idx}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing fused metadata file for frame {frame_idx}: {metadata_path}")

    metadata = read_json(metadata_path)
    joint_confidence = metadata.get("joint_confidence")
    if not isinstance(joint_confidence, dict):
        raise ValueError(f"Missing joint_confidence in {metadata_path}")

    camera_confidence = joint_confidence.get(cam_id)
    if not isinstance(camera_confidence, dict):
        raise ValueError(f"Missing joint_confidence for {cam_id} in {metadata_path}")

    return camera_confidence


def _resolve_2d_frame_key(keypoints2d_payload: dict, source_frame_idx: int) -> Optional[str]:
    keypoints_by_frame = keypoints2d_payload.get("keypoints", {})
    if not isinstance(keypoints_by_frame, dict):
        return None

    direct_key = str(int(source_frame_idx))
    if direct_key in keypoints_by_frame:
        return direct_key

    frame_ids = keypoints2d_payload.get("frame_ids", [])
    if isinstance(frame_ids, list) and 0 <= int(source_frame_idx) < len(frame_ids):
        positional_key = str(int(frame_ids[int(source_frame_idx)]))
        if positional_key in keypoints_by_frame:
            return positional_key

    return None

def execute_optimization(config):
    opt_config = config.get("optimization", {})
    if not opt_config.get("enabled", True):
        return

    preprocess_out_dir = Path(config["preprocess"]["output_dir"])
    pose_out_dir = Path(config["paths"]["pose_output_dir"])
    fused_out_dir = Path(config["paths"]["fused_output_dir"])
    learnable_out_dir = Path(config["paths"]["learnable_output_dir"])
    opt_out_dir = Path(config["paths"]["optimized_output_dir"])
    
    keypoints3d_dir = opt_out_dir / "keypoints3d"
    metadata_dir = opt_out_dir / "metadata"
    keypoints3d_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    cameras_list = opt_config.get("cameras", ["camera1", "camera2"])
    
    device_name = opt_config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    
    smpl_model_path = config["paths"]["smpl_model"]
    j_regressor_path = config["paths"]["j_regressor_3d"]
    model = SmplBody25Model(
        smpl_model_path=smpl_model_path,
        j_regressor_path=j_regressor_path,
        device=device,
        model_cfg=opt_config["model"],
    )
    
    learnable_meta_dir = learnable_out_dir / "metadata"
    learnable_files = sorted(learnable_meta_dir.glob("learnable_frame_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    if not learnable_files:
        print("[Optimization] No learnable frames found.")
        return
    learnable_frame_ids = [int(p.stem.split("_")[-1]) for p in learnable_files]

    n_frames = len(learnable_files)
    print(f"[Optimization] Processing {n_frames} frames.")

    camera_results = {}
    camera_metadata = {}
    
    map_data = load_keypoints3d_map(config["paths"]["keypoints3d_map"])
    canonical_names = [kp["name"] for kp in map_data["keypoints"]]
    name_to_reg_idx = {kp["name"]: kp["regressor_index"] for kp in map_data["keypoints"]}
    priority1_names = list(map_data["priority1"])
    priority1_indices = [name_to_reg_idx[name] for name in priority1_names]
    n_regressor_joints = max(name_to_reg_idx.values()) + 1
    fused_metadata_dir = fused_out_dir / "metadata"
    phase2_cfg = dict(opt_config["phase2"])
    phase2_cfg["joint_indices"] = priority1_indices

    for cam_id in cameras_list:
        print(f"[Optimization] Processing {cam_id}...")
        cam_num = cam_id.replace("camera", "")
        data_cam_file = preprocess_out_dir / f"data_cam{cam_num}.json"
        
        if not data_cam_file.exists():
            raise FileNotFoundError(f"Missing required preprocess profile: {data_cam_file}")
            
        data_cam = read_json(data_cam_file)
        
        Th_list = []
        Rh_list = []
        poses_list = []
        keypoints2d_list = []
        
        cam_2d_key = f"2D_camera_cam{cam_num}"
        if cam_2d_key in data_cam:
            keypoints2d_payload = data_cam[cam_2d_key]
            keypoints2d_all = keypoints2d_payload["keypoints"]
        else:
            raise ValueError(f"Missing required {cam_2d_key} in {data_cam_file}")

        intrinsics_source = data_cam.get("intrinsics_source", "")
        if intrinsics_source == "intri_cam":
            K_array = np.array(data_cam["intrinsics_cam"], dtype=np.float32)
        else:
            K_array = np.array(data_cam["intrinsics_estimation"], dtype=np.float32)
            
        K_arrays = np.repeat(K_array[None, ...], n_frames, axis=0)

        for i, l_file in enumerate(learnable_files):
            frame_id = learnable_frame_ids[i]
            l_data = read_json(l_file)
            
            # The learnable output places data under "learnable_smplify" -> "cameraX"
            learnable_base = l_data.get("learnable_smplify", {})
            
            if cam_id not in learnable_base:
                raise ValueError(f"Missing {cam_id} in {l_file} under 'learnable_smplify'")
            else:
                cam_data = learnable_base[cam_id]
                if "pose_pred" not in cam_data or len(cam_data["pose_pred"]) != 72:
                     raise ValueError(f"Invalid 'pose_pred' for {cam_id} in {l_file}")
                else:
                    pose_pred = cam_data["pose_pred"]
            
            Rh_list.append(pose_pred[:3])
            poses_list.append(pose_pred[3:])
            
            trans_cam = data_cam.get("trans_cam", np.zeros(3))
            if isinstance(trans_cam, list) and len(trans_cam) == n_frames:
                Th_list.append(trans_cam[i])
            elif isinstance(trans_cam, list) and len(trans_cam) > 0 and isinstance(trans_cam[0], list):
                Th_list.append(trans_cam[i])
            else:
                Th_list.append(trans_cam)
                
                
            # Read source frame index from pose metadata
            pose_meta_file = pose_out_dir / "metadata" / f"pose_data_{frame_id}.json"
            if not pose_meta_file.exists():
                raise FileNotFoundError(f"Missing pose metadata file for frame {frame_id}: {pose_meta_file}")
            
            pose_meta = read_json(pose_meta_file)
            source_frame_indices = pose_meta.get("metadata", {}).get("source_frame_indices", {})
            if cam_id not in source_frame_indices:
                raise ValueError(f"Missing source_frame_indices for {cam_id} in {pose_meta_file}")
            
            source_frame_idx = source_frame_indices[cam_id]
            fusion_confidence = _load_fusion_confidence_for_frame(
                fused_metadata_dir=fused_metadata_dir,
                frame_idx=frame_id,
                cam_id=cam_id,
            )
                
            kp2d_frame = np.zeros((n_regressor_joints, 3), dtype=np.float32)
            frame_str = _resolve_2d_frame_key(keypoints2d_payload, source_frame_idx)
            if frame_str is not None:
                frame_keypoints = keypoints2d_all[frame_str]
                for j_name in canonical_names:
                    if j_name not in frame_keypoints:
                        raise ValueError(f"Missing 2D keypoint {j_name} for frame {frame_str} in {data_cam_file}")
                    if j_name not in fusion_confidence:
                        raise ValueError(
                            f"Missing fusion joint_confidence for {cam_id}.{j_name} in fused_data_{frame_id}.json"
                        )
                    reg_idx = name_to_reg_idx[j_name]
                    kp2d_frame[reg_idx, :2] = np.asarray(frame_keypoints[j_name][:2], dtype=np.float32)
                    kp2d_frame[reg_idx, 2] = float(fusion_confidence[j_name])
            else:
                raise ValueError(
                    f"Frame {source_frame_idx} not found in 2D keypoints for {cam_id}. "
                    "Expected either a direct tracking frame_id match or a valid positional index."
                )
            keypoints2d_list.append(kp2d_frame)

        Th_array = np.array(Th_list, dtype=np.float32)
        Rh_array = np.array(Rh_list, dtype=np.float32)
        poses_array = np.array(poses_list, dtype=np.float32)
        keypoints2d_array = np.array(keypoints2d_list, dtype=np.float32)
        shapes_array = np.array(data_cam["betas"], dtype=np.float32)
        
        if shapes_array.ndim == 1:
            shapes_array = shapes_array[None, :]

        params = {
            "Th": Th_array,
            "Rh": Rh_array,
            "poses": poses_array,
            "shapes": shapes_array
        }
        
        cameras = {
            "K": K_arrays,
            "R": np.repeat(np.eye(3, dtype=np.float32)[None, ...], n_frames, axis=0),
            "T": np.zeros((n_frames, 3), dtype=np.float32)
        }
        
        print(f"[{cam_id}] Running Phase 2...")
        params = run_phase2_init_translation(
            params=params,
            keypoints=keypoints2d_array,
            cameras=cameras,
            model=model,
            phase2_cfg=phase2_cfg,
        )
        
        print(f"[{cam_id}] Running Phase 3...")
        params["poses"] = smooth_poses(params["poses"], window_size=int(opt_config["phase3"]["window_size"]))
        
        print(f"[{cam_id}] Running Phase 5...")
        params_t, cameras_t, keypoints_t = to_tensor_dict(params, cameras, keypoints2d_array, device=device)
        phase5_summary = optimize_phase5(
            params=params_t,
            cameras=cameras_t,
            keypoints=keypoints_t,
            model=model,
            phase5_cfg=opt_config["phase5"],
        )
        
        print(f"[{cam_id}] Running Phase 6...")
        phase6_summaries = optimize_phase6(
            params=params_t,
            cameras=cameras_t,
            keypoints=keypoints_t,
            model=model,
            phase6_cfg=opt_config["phase6"],
        )
        
        final_params = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in params_t.items()}
        final_keypoints_3d = model.keypoints(final_params)
        
        camera_results[cam_id] = final_keypoints_3d.tolist()
        camera_metadata[cam_id] = {
            "params": {
                "Th": final_params["Th"].tolist(),
                "Rh": final_params["Rh"].tolist(),
                "poses": final_params["poses"].tolist(),
                "shapes": final_params["shapes"].tolist(),
            },
            "summaries": {
                "phase5": phase5_summary,
                "phase6": phase6_summaries,
            },
            "source_files": {
                "data_cam": str(data_cam_file),
                "intrinsics_source": intrinsics_source,
                "fusion_metadata_dir": str(fused_metadata_dir),
            }
        }

    # Write frame outputs
    print("[Optimization] Writing output files...")
    for i in range(n_frames):
        frame_idx = learnable_frame_ids[i]
        keypoints_out = {}
        meta_out = {}
        
        for cam_id in camera_results:
            keypoints_out[cam_id] = {}
            meta_out[cam_id] = {}
            
            for custom_name in canonical_names:
                idx = name_to_reg_idx[custom_name]
                keypoints_out[cam_id][custom_name] = [
                    round(float(camera_results[cam_id][i][idx][0]), 5),
                    round(float(camera_results[cam_id][i][idx][1]), 5),
                    round(float(camera_results[cam_id][i][idx][2]), 5),
                ]
                
            meta_out[cam_id]["Rh"] = camera_metadata[cam_id]["params"]["Rh"][i]
            meta_out[cam_id]["Th"] = camera_metadata[cam_id]["params"]["Th"][i]
            meta_out[cam_id]["poses"] = camera_metadata[cam_id]["params"]["poses"][i]
            
            if len(camera_metadata[cam_id]["params"]["shapes"]) > i:
                meta_out[cam_id]["shapes"] = camera_metadata[cam_id]["params"]["shapes"][i]
            else:
                meta_out[cam_id]["shapes"] = camera_metadata[cam_id]["params"]["shapes"][0]
                
            meta_out[cam_id]["phase5_summary"] = camera_metadata[cam_id]["summaries"]["phase5"]
            meta_out[cam_id]["phase6_summary"] = camera_metadata[cam_id]["summaries"]["phase6"]
            meta_out[cam_id]["source_files"] = camera_metadata[cam_id]["source_files"]

        write_json(keypoints3d_dir / f"optimized_data_{frame_idx}.json", keypoints_out)
        write_json(metadata_dir / f"optimized_data_{frame_idx}.json", meta_out)

    # Write global summary metadata
    summary_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "frame_count": n_frames,
        "cameras_processed": list(camera_results.keys()),
        "config_snapshot": {
            "device": device_name,
            "phase2": phase2_cfg,
            "phase3": opt_config["phase3"],
        },
            "model_paths": {
                "smpl_model": smpl_model_path,
                "j_regressor_3d": j_regressor_path,
            }
        }
    write_json(opt_out_dir / "metadata.json", summary_data)
    print(f"[Optimization] Done. Saved {n_frames} frames to {opt_out_dir}")
