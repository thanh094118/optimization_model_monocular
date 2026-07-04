from pathlib import Path
import re
import copy
import joblib
import numpy as np
from json_io import read_json, write_json
from keypoints_map import load_keypoints3d_map
from fusion_pipeline.config import OUTPUT_SUBDIRS
from fusion_pipeline.correction import (
    apply_confidence_corrections,
    apply_rotation_mismatch_corrections,
    estimate_bidirectional_similarity,
)
from fusion_pipeline.detector import (
    as_xyz,
    compute_visibility_from_mesh_vertices,
    detect_cross_view_errors,
    get_orientation_flag,
    load_torso_mask,
    make_raw_judgement_fallback,
)
from fusion_pipeline.optimization import calculate_stats, optimize_f_points
from preprocess_pipeline.calib import resolve_selected_intrinsics
from config_loader import resolve_inputs


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


def _load_occlusion_intrinsics(config: dict) -> dict:
    return {
        "camera1": np.asarray(resolve_selected_intrinsics(config, "cam1"), dtype=float),
        "camera2": np.asarray(resolve_selected_intrinsics(config, "cam2"), dtype=float),
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


def run_phase3_pipeline(
    data_in,
    map_path,
    verts_by_cam=None,
    intrinsics_by_cam=None,
    occlusion_tau=0.05,
    regularization=False,
    regularization_lambda=1.0,
    temporal_lambda=2.0,
    max_iter=1000,
    ransac_threshold=0.05,
    ransac_max_combos=500,
    frame_idx=None,
    prev_optimized_data=None,
    confidence2d_by_cam=None,
):
    cam1 = {k: as_xyz(v) for k, v in data_in["camera1"].items()}
    cam2 = {k: as_xyz(v) for k, v in data_in["camera2"].items()}

    map_data = load_keypoints3d_map(map_path)
    expected_names = [kp["name"] for kp in map_data["keypoints"]]
    
    if set(cam1.keys()) != set(expected_names) or set(cam2.keys()) != set(expected_names):
        raise ValueError("Input does not have exactly the 21 expected keys for both cameras")
        
    names = expected_names
    cam1 = {k: cam1[k] for k in names}
    cam2 = {k: cam2[k] for k in names}

    if verts_by_cam is not None:
        if intrinsics_by_cam is None:
            raise ValueError("intrinsics_by_cam is required when verts_by_cam is provided")
        vis1 = compute_visibility_from_mesh_vertices(cam1, verts_by_cam["camera1"], intrinsics_by_cam["camera1"], occlusion_tau)
        vis2 = compute_visibility_from_mesh_vertices(cam2, verts_by_cam["camera2"], intrinsics_by_cam["camera2"], occlusion_tau)
    else:
        vis1 = {n: True for n in names}
        vis2 = {n: True for n in names}

    confidence2d_by_cam = confidence2d_by_cam or {}
    detected = detect_cross_view_errors(
        cam1,
        cam2,
        names,
        vis1,
        vis2,
        confidence2d1=confidence2d_by_cam.get("camera1"),
        confidence2d2=confidence2d_by_cam.get("camera2"),
    )
    m_set = detected["M"]
    k1_set = detected["K1"]
    k2_set = detected["K2"]
    l_list = detected["L"]
    all_weights = detected["weights"]
    H1_all = detected["H1"]
    H2_all = detected["H2"]

    t12, t21, a_list = estimate_bidirectional_similarity(
        cam1,
        cam2,
        l_list,
        threshold=ransac_threshold,
        max_combos=ransac_max_combos,
    )

    cam1_corr, cam2_corr = apply_confidence_corrections(cam1, cam2, k1_set, k2_set, t12, t21)
    cam1_corr, cam2_corr = apply_rotation_mismatch_corrections(
        cam1_corr,
        cam2_corr,
        cam1,
        cam2,
        m_set,
        k1_set,
        k2_set,
        H1_all,
        H2_all,
        t12,
        t21,
    )

    a_new = sorted(set(a_list) | k1_set | k2_set)
    f_list = [n for n in names if n not in set(a_new)]
    before_stats = calculate_stats(cam1_corr, cam2_corr, names, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)
    optimized_data, _ = optimize_f_points(
        {"camera1": cam1_corr, "camera2": cam2_corr},
        a_new,
        f_list,
        conf1=H1_all,
        conf2=H2_all,
        vis1=vis1,
        vis2=vis2,
        regularization=regularization,
        regularization_lambda=regularization_lambda,
        prev_data=prev_optimized_data,
        temporal_lambda=temporal_lambda,
        max_iter=max_iter,
    )
    after_stats = calculate_stats(optimized_data["camera1"], optimized_data["camera2"], names, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)

    flags1_after = get_orientation_flag(optimized_data["camera1"])
    flags2_after = get_orientation_flag(optimized_data["camera2"])
    m_after = {n for n in names if (flags1_after.get(n, 0) == 1 and flags2_after.get(n, 0) == -1) or (flags1_after.get(n, 0) == -1 and flags2_after.get(n, 0) == 1)}

    return {
        "M": sorted(m_set),
        "M_after": sorted(m_after),
        "M_resolved": len(m_after) == 0 and len(m_set) > 0,
        "K1": sorted(k1_set),
        "K2": sorted(k2_set),
        "A_new": a_new,
        "F": f_list,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "optimized": {"camera1": {k: list(v) for k, v in optimized_data["camera1"].items()}, "camera2": {k: list(v) for k, v in optimized_data["camera2"].items()}},
        "joint_confidence": {"camera1": H1_all, "camera2": H2_all},
        "vis1": {k: bool(v) for k, v in vis1.items()},
        "vis2": {k: bool(v) for k, v in vis2.items()},
    }


def run_fusion(config: dict) -> None:
    paths = config["paths"]
    inputs = resolve_inputs(config)
    runtime_cfg = config.get("runtime", {})
    fusion_cfg = config.get("fusion", {})

    if not fusion_cfg.get("enabled", True):
        print("[Fusion] Disabled by config: fusion.enabled=false")
        return

    input_dir = Path(paths["pose_output_dir"])
    output_dir = Path(paths["fused_output_dir"])

    if not input_dir.exists():
        raise FileNotFoundError(f"Pose JSON directory not found: {input_dir}")

    if runtime_cfg.get("clean_output", True):
        _clean_output(output_dir, "fused_data_*.json", create_split_dirs=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    occlusion_cfg = fusion_cfg.get("occlusion", {})
    occlusion_enabled = occlusion_cfg.get("enabled", True)
    if occlusion_enabled:
        load_torso_mask(paths.get("segmentation"))

    wham_loaded, verts_cam1, verts_cam2, n_frames_1, n_frames_2 = _load_verts_if_available(inputs, occlusion_enabled)
    confidence2d_profiles = _load_2d_profiles(config)
    occlusion_intrinsics = _load_occlusion_intrinsics(config) if occlusion_enabled and wham_loaded else None

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
                map_path=paths["keypoints3d_map"],
                verts_by_cam=verts_input,
                intrinsics_by_cam=occlusion_intrinsics,
                occlusion_tau=occlusion_cfg.get("tau", 0.01),
                regularization=opt_cfg.get("regularization", True),
                regularization_lambda=opt_cfg.get("regularization_lambda", 1.0),
                temporal_lambda=opt_cfg.get("temporal_lambda", 2.0),
                max_iter=opt_cfg.get("max_iter", 1000),
                ransac_threshold=ransac_cfg.get("threshold", 0.05),
                ransac_max_combos=ransac_cfg.get("max_combos", 500),
                frame_idx=frame_idx,
                prev_optimized_data=prev_opt,
                confidence2d_by_cam=confidence2d_by_cam,
            )
            occluded_keys = sorted(
                {
                    name
                    for name in set(result.get("vis1", {})) | set(result.get("vis2", {}))
                    if (not result.get("vis1", {}).get(name, True)) or (not result.get("vis2", {}).get(name, True))
                }
            )
            if occluded_keys:
                print(f"[Fusion] Frame {frame_idx}: Occlusion: {', '.join(occluded_keys)}")
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

    print(f"[Fusion] Done. Output: {output_dir}")
