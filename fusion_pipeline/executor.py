from pathlib import Path
import re
import copy
import joblib
from json_io import read_json, write_json
from fusion_pipeline.config import OUTPUT_SUBDIRS
from fusion_pipeline.detection import load_torso_mask, make_raw_judgement_fallback
from fusion_pipeline.pipeline import run_phase3_pipeline
from fusion_pipeline.logs import log_disabled, log_done, log_occlusion


def _frame_index(path: Path) -> int:
    match = re.search(r"\d+", path.name)
    if not match:
        raise ValueError(f"Cannot extract frame index from {path.name}")
    return int(match.group())


def _clean_output(output_dir: Path, pattern: str = "*.json", create_split_dirs: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob(pattern):
        old_json.unlink()
    if create_split_dirs:
        for subdir in OUTPUT_SUBDIRS:
            target_dir = output_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            for old_json in target_dir.glob(pattern):
                old_json.unlink()


def _extract_person_payload(wham_data):
    if isinstance(wham_data, dict):
        if 0 in wham_data:
            return wham_data[0]
        if "0" in wham_data:
            return wham_data["0"]
        for value in wham_data.values():
            if isinstance(value, dict):
                return value
    if isinstance(wham_data, list):
        for value in wham_data:
            if isinstance(value, dict):
                return value
    return None


def _load_verts_if_available(paths: dict, occlusion_enabled: bool):
    if not occlusion_enabled:
        return False, None, None, 0, 0

    wham_path_1 = Path(paths["cam1_pkl"])
    wham_path_2 = Path(paths["cam2_pkl"])
    if not (wham_path_1.exists() and wham_path_2.exists()):
        print("[Fusion] WARNING: WHAM PKL files not found. Occlusion disabled.")
        return False, None, None, 0, 0

    person_1 = _extract_person_payload(joblib.load(wham_path_1))
    person_2 = _extract_person_payload(joblib.load(wham_path_2))

    if person_1 is None or person_2 is None or "verts_cam" not in person_1 or "verts_cam" not in person_2:
        print("[Fusion] WARNING: verts_cam not found. Occlusion disabled.")
        return False, None, None, 0, 0

    verts_cam1 = person_1["verts_cam"]
    verts_cam2 = person_2["verts_cam"]
    print(f"[Fusion] WHAM cam1 verts: {verts_cam1.shape[0]} frames")
    print(f"[Fusion] WHAM cam2 verts: {verts_cam2.shape[0]} frames")
    return True, verts_cam1, verts_cam2, verts_cam1.shape[0], verts_cam2.shape[0]


def _load_pose_frame(path: Path, metadata_dir: Path):
    data = read_json(path)
    if "camera1" in data and "camera2" in data:
        metadata_path = metadata_dir / path.name
        if metadata_path.exists():
            metadata_data = read_json(metadata_path)
            data.update(metadata_data)
        return data
    return data


def _load_2d_profile(config: dict, cam_id: str):
    preprocess_dir = Path(config.get("preprocess", {}).get("output_dir", "output/preprocess_results"))
    profile_path = preprocess_dir / f"data_{cam_id}.json"
    if not profile_path.exists():
        print(f"[Fusion] 2D confidence not found: {profile_path}")
        return None
    profile = read_json(profile_path)
    payload = profile.get(f"2D_camera_{cam_id}")
    if not isinstance(payload, dict):
        print(f"[Fusion] 2D confidence missing in {profile_path.name}")
        return None
    keypoints = payload.get("keypoints")
    if not isinstance(keypoints, dict):
        print(f"[Fusion] 2D confidence keypoints missing in {profile_path.name}")
        return None
    return keypoints


def _load_2d_profiles(config: dict) -> dict:
    return {
        "camera1": _load_2d_profile(config, "cam1"),
        "camera2": _load_2d_profile(config, "cam2"),
    }


def _frame_confidence_from_profile(profile, source_idx, frame_idx: int):
    if not profile:
        return None
    candidates = []
    if source_idx is not None:
        candidates.append(int(source_idx))
    candidates.extend([frame_idx - 1, frame_idx])
    for candidate in candidates:
        data = profile.get(str(candidate))
        if isinstance(data, dict):
            return data
    return None


def _confidence2d_for_frame(data: dict, frame_idx: int, profiles: dict) -> dict:
    source_indices = data.get("metadata", {}).get("source_frame_indices", {})
    return {
        "camera1": _frame_confidence_from_profile(profiles.get("camera1"), source_indices.get("camera1"), frame_idx),
        "camera2": _frame_confidence_from_profile(profiles.get("camera2"), source_indices.get("camera2"), frame_idx),
    }


def run_fusion(config: dict) -> None:
    paths = config["paths"]
    runtime_cfg = config.get("runtime", {})
    fusion_cfg = config.get("fusion", {})

    if not fusion_cfg.get("enabled", True):
        log_disabled()
        return

    input_dir = Path(paths["pose_output_dir"])
    output_dir = Path(paths["fused_output_dir"])

    debug_cfg = fusion_cfg.get("debug", {})
    debug1_dir = (output_dir / "debug1") if debug_cfg.get("save_debug1", True) else None
    debug2_dir = (output_dir / "debug2") if debug_cfg.get("save_debug2", False) else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Pose JSON directory not found: {input_dir}")

    if runtime_cfg.get("clean_output", True):
        _clean_output(output_dir, "fused_data_*.json", create_split_dirs=True)
        if debug1_dir is not None:
            _clean_output(debug1_dir, "fused_data_*.json")
        if debug2_dir is not None:
            _clean_output(debug2_dir, "fused_data_*.json")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if debug1_dir is not None:
            debug1_dir.mkdir(parents=True, exist_ok=True)
        if debug2_dir is not None:
            debug2_dir.mkdir(parents=True, exist_ok=True)

    occlusion_cfg = fusion_cfg.get("occlusion", {})
    occlusion_enabled = occlusion_cfg.get("enabled", True)
    if occlusion_enabled:
        load_torso_mask(paths.get("segmentation"))

    wham_loaded, verts_cam1, verts_cam2, n_frames_1, n_frames_2 = _load_verts_if_available(paths, occlusion_enabled)
    confidence2d_profiles = _load_2d_profiles(config)

    keypoints_dir = input_dir / "keypoints3d"
    metadata_dir = input_dir / "metadata"
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Pose keypoints directory not found: {keypoints_dir}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Pose metadata directory not found: {metadata_dir}")
    file_paths = sorted(keypoints_dir.glob("pose_data_*.json"), key=_frame_index)
    print(f"[Fusion] Found {len(file_paths)} pose JSON files")

    ransac_cfg = fusion_cfg.get("ransac", {})
    opt_cfg = fusion_cfg.get("optimization", {})

    prev_result = None
    for path in file_paths:
        frame_idx = _frame_index(path)
        out_name = f"fused_data_{frame_idx}.json"
        data = _load_pose_frame(path, metadata_dir=metadata_dir)

        if wham_loaded:
            wham_frame = frame_idx - 1
            if 0 <= wham_frame < n_frames_1 and wham_frame < n_frames_2:
                verts_input = {"camera1": verts_cam1[wham_frame], "camera2": verts_cam2[wham_frame]}
            else:
                print(f"[Fusion] Frame {frame_idx}: WHAM frame out of range. Occlusion skipped.")
                verts_input = None
        else:
            verts_input = None

        try:
            prev_opt = prev_result["optimized"] if prev_result and "optimized" in prev_result else None
            confidence2d_by_cam = _confidence2d_for_frame(data, frame_idx, confidence2d_profiles)
            result = run_phase3_pipeline(
                data,
                verts_by_cam=verts_input,
                occlusion_tau=occlusion_cfg.get("tau", 0.01),
                regularization=opt_cfg.get("regularization", True),
                regularization_lambda=opt_cfg.get("regularization_lambda", 1.0),
                temporal_lambda=opt_cfg.get("temporal_lambda", 2.0),
                max_iter=opt_cfg.get("max_iter", 1000),
                ransac_threshold=ransac_cfg.get("threshold", 0.05),
                ransac_max_combos=ransac_cfg.get("max_combos", 500),
                frame_idx=frame_idx,
                prev_optimized_data=prev_opt,
                debug1_dir=debug1_dir,
                debug2_dir=debug2_dir,
                confidence2d_by_cam=confidence2d_by_cam,
            )
            occluded_keys = sorted(
                {
                    name
                    for name in set(result.get("vis1", {})) | set(result.get("vis2", {}))
                    if (not result.get("vis1", {}).get(name, True)) or (not result.get("vis2", {}).get(name, True))
                }
            )
            log_occlusion(frame_idx=frame_idx, occluded_keys=occluded_keys)
            prev_result = result
        except Exception as e:
            print(f"[Fusion] Frame {frame_idx}: FAILED ({e}) -> fallback")
            result = copy.deepcopy(prev_result) if prev_result is not None else make_raw_judgement_fallback(data, frame_idx, e)

        fused_keypoints = {
            "camera1": result.get("optimized", {}).get("camera1", {}),
            "camera2": result.get("optimized", {}).get("camera2", {}),
        }
        fused_metadata = {k: v for k, v in result.items() if k not in ("camera1", "camera2", "optimized")}
        write_json(output_dir / "keypoints3d" / out_name, fused_keypoints)
        write_json(output_dir / "metadata" / out_name, fused_metadata)

    log_done(str(output_dir))
    if debug1_dir is not None:
        print(f"[Fusion] Debug1: {debug1_dir}")
    if debug2_dir is not None:
        print(f"[Fusion] Debug2: {debug2_dir}")
