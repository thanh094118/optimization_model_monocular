from pathlib import Path
from pose_pipeline.io_utils.pkl_loader import load_pkl_data
from pose_pipeline.sync import slice_person_frames
from pose_pipeline.smpl_runner import create_smpl_model, get_3d_joints_for_frame
from json_io import write_json


def _clean_pose_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob("pose_data_*.json"):
        old_json.unlink()
    for subdir in ("keypoints3d", "metadata"):
        target_dir = output_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for old_json in target_dir.glob("pose_data_*.json"):
            old_json.unlink()


def run_pose_export(config: dict) -> None:
    paths = config["paths"]
    runtime_cfg = config.get("runtime", {})
    pose_cfg = config.get("pose_export", {})

    if not pose_cfg.get("enabled", True):
        print("[Pose] Disabled by config: pose_export.enabled=false")
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
    print(f"[Pose] Using offset={offset} ({sync_result['method']})")
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

    print(f"[Pose] Done. Output: {output_dir}")
