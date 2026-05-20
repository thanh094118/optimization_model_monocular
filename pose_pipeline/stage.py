from pathlib import Path
from pose_pipeline.io_utils.pkl_loader import load_pkl_data
from pose_pipeline.sync import estimate_camera_frame_offset, slice_person_frames
from pose_pipeline.smpl_runner import create_smpl_model, get_3d_joints_for_frame
from json_io import write_json


def _clean_pose_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob("pose_data_*.json"):
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

    sync_result = estimate_camera_frame_offset(
        cam1_data,
        cam2_data,
        max_offset=pose_cfg.get("max_sync_offset", 40),
        min_overlap=pose_cfg.get("min_overlap", 8),
        min_improvement_ratio=pose_cfg.get("min_improvement_ratio", 0.05),
    )
    min_frames = int(sync_result["frame_count"])
    cam1_start = int(sync_result["left_start"])
    cam2_start = int(sync_result["right_start"])

    print(f"[Pose] Cam1 frames: {len(cam1_data['pose'])}")
    print(f"[Pose] Cam2 frames: {len(cam2_data['pose'])}")
    print(f"[Pose] Sync: offset={sync_result['offset']}, cam1_start={cam1_start}, cam2_start={cam2_start}")
    print(f"[Pose] Exporting {min_frames} pose JSON files...")

    cam1_export_data = slice_person_frames(cam1_data, cam1_start, min_frames)
    cam2_export_data = slice_person_frames(cam2_data, cam2_start, min_frames)

    for i in range(min_frames):
        frame_data = {
            "camera1": get_3d_joints_for_frame(model, cam1_export_data, i),
            "camera2": get_3d_joints_for_frame(model, cam2_export_data, i),
            "metadata": {
                "camera_sync": sync_result,
                "source_frame_indices": {
                    "camera1": cam1_start + i,
                    "camera2": cam2_start + i,
                },
            },
        }
        output_path = output_dir / f"pose_data_{i + 1}.json"
        write_json(output_path, frame_data)
        if (i + 1) % 50 == 0 or (i + 1) == min_frames:
            print(f"[Pose] Saved {output_path.name} ({i + 1}/{min_frames})")

    print(f"[Pose] Done. Output: {output_dir}")
