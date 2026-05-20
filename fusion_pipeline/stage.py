from pathlib import Path
import re
import copy
import joblib
from json_io import read_json, write_json
from fusion_pipeline.geometry import load_torso_mask
from fusion_pipeline.core import run_phase3_pipeline, make_raw_judgement_fallback


def _frame_index(path: Path) -> int:
    match = re.search(r"\d+", path.name)
    if not match:
        raise ValueError(f"Cannot extract frame index from {path.name}")
    return int(match.group())


def _clean_output(output_dir: Path, pattern: str = "*.json") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob(pattern):
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


def run_fusion(config: dict) -> None:
    paths = config["paths"]
    runtime_cfg = config.get("runtime", {})
    fusion_cfg = config.get("fusion", {})

    if not fusion_cfg.get("enabled", True):
        print("[Fusion] Disabled by config: fusion.enabled=false")
        return

    input_dir = Path(paths["pose_output_dir"])
    output_dir = Path(paths["fused_output_dir"])

    debug_cfg = fusion_cfg.get("debug", {})
    debug1_dir = Path(paths.get("debug1_dir", "output/debug1")) if debug_cfg.get("save_debug1", True) else None
    debug2_dir = Path(paths.get("debug2_dir", "output/debug2")) if debug_cfg.get("save_debug2", False) else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Pose JSON directory not found: {input_dir}")

    if runtime_cfg.get("clean_output", True):
        _clean_output(output_dir, "fused_data_*.json")
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

    file_paths = sorted(input_dir.glob("pose_data_*.json"), key=_frame_index)
    print(f"[Fusion] Found {len(file_paths)} pose JSON files")

    ransac_cfg = fusion_cfg.get("ransac", {})
    opt_cfg = fusion_cfg.get("optimization", {})

    prev_result = None
    for path in file_paths:
        frame_idx = _frame_index(path)
        out_path = output_dir / f"fused_data_{frame_idx}.json"
        data = read_json(path)

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
            )
            print(f"[Fusion] Frame {frame_idx}: OK | A_new={len(result['A_new'])} F={len(result['F'])} | Mean={result['after_stats'][2]:.5f}m")
            prev_result = result
        except Exception as e:
            print(f"[Fusion] Frame {frame_idx}: FAILED ({e}) -> fallback")
            result = copy.deepcopy(prev_result) if prev_result is not None else make_raw_judgement_fallback(data, frame_idx, e)

        write_json(out_path, result)

    print(f"[Fusion] Done. Output: {output_dir}")
    if debug1_dir is not None:
        print(f"[Fusion] Debug1: {debug1_dir}")
    if debug2_dir is not None:
        print(f"[Fusion] Debug2: {debug2_dir}")
