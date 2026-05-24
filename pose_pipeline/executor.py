import copy
from pathlib import Path
import joblib
import numpy as np

from pose_pipeline.smpl_runner import create_smpl_model, get_3d_joints_for_frame
from pose_pipeline.config import POSE_OUTPUT_SUBDIRS
from pose_pipeline.logs import log_disabled, log_done, log_using_offset
from json_io import write_json


def _extract_person_payload(data):
    if isinstance(data, dict) or "defaultdict" in str(type(data)):
        if 0 in data:
            return data[0]
        if "0" in data:
            return data["0"]
        for value in data.values():
            if isinstance(value, dict):
                return value
    if isinstance(data, list):
        for value in data:
            if isinstance(value, dict):
                return value
    return None


def load_pkl_data(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("PKL file not found: {}".format(file_path))
    person_data = _extract_person_payload(joblib.load(file_path))
    if person_data is None:
        raise ValueError("Cannot read person payload from {}".format(file_path))
    for key in ("pose", "trans", "betas"):
        if key not in person_data:
            raise KeyError("Missing key {!r} in {}. Available keys: {}".format(key, file_path, list(person_data.keys())))
    return person_data


def slice_person_frames(person_data, start, frame_count):
    start = int(start)
    frame_count = int(frame_count)
    total_frames = int(np.asarray(person_data["pose"]).shape[0])
    end = min(total_frames, start + frame_count)
    out = copy.deepcopy(dict(person_data))
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_frames:
            out[key] = value[start:end].copy()
    return out


def _clean_pose_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob("pose_data_*.json"):
        old_json.unlink()
    for subdir in POSE_OUTPUT_SUBDIRS:
        target_dir = output_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for old_json in target_dir.glob("pose_data_*.json"):
            old_json.unlink()


def run_pose_export(config: dict) -> None:
    paths = config["paths"]
    runtime_cfg = config.get("runtime", {})
    pose_cfg = config.get("pose_export", {})

    if not pose_cfg.get("enabled", True):
        log_disabled()
        return

    cam1_file = Path(paths["cam1_pkl"])
    cam2_file = Path(paths["cam2_pkl"])
    smpl_model_path = Path(paths["smpl_model"])
    output_dir = Path(paths["pose_output_dir"])

    if runtime_cfg.get("clean_output", True):
        _clean_pose_output(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("[Pose] Loading SMPL model...")
    model = create_smpl_model(smpl_model_path)

    print(f"[Pose] Loading camera 1: {cam1_file}")
    cam1_data = load_pkl_data(cam1_file)
    print(f"[Pose] Loading camera 2: {cam2_file}")
    cam2_data = load_pkl_data(cam2_file)

    offset = int(config.get("runtime", {}).get("selected_offset", 0))
    cam1_start = max(0, -offset)
    cam2_start = max(0, offset)
    min_frames = max(0, min(len(cam1_data["pose"]) - cam1_start, len(cam2_data["pose"]) - cam2_start))
    sync_result = {
        "offset": offset,
        "left_start": cam1_start,
        "right_start": cam2_start,
        "frame_count": min_frames,
        "method": config.get("runtime", {}).get("offset_method", "unknown"),
    }

    print(f"[Pose] Cam1 frames: {len(cam1_data['pose'])}")
    print(f"[Pose] Cam2 frames: {len(cam2_data['pose'])}")
    log_using_offset(offset=offset, method=sync_result["method"])
    print(f"[Pose] Exporting {min_frames} pose JSON files...")

    cam1_export_data = slice_person_frames(cam1_data, cam1_start, min_frames)
    cam2_export_data = slice_person_frames(cam2_data, cam2_start, min_frames)

    for i in range(min_frames):
        keypoints3d_data = {
            "camera1": get_3d_joints_for_frame(model, cam1_export_data, i),
            "camera2": get_3d_joints_for_frame(model, cam2_export_data, i),
        }
        metadata_data = {
            "metadata": {
                "camera_sync": sync_result,
                "source_frame_indices": {
                    "camera1": cam1_start + i,
                    "camera2": cam2_start + i,
                },
            }
        }
        keypoints_path = output_dir / "keypoints3d" / f"pose_data_{i + 1}.json"
        metadata_path = output_dir / "metadata" / f"pose_data_{i + 1}.json"
        write_json(keypoints_path, keypoints3d_data)
        write_json(metadata_path, metadata_data)
        if (i + 1) % 50 == 0 or (i + 1) == min_frames:
            print(f"[Pose] Saved pose_data_{i + 1}.json ({i + 1}/{min_frames})")

    log_done(str(output_dir))
