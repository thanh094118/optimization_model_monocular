import re

def fix_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "executor.py" in filepath and "learnable_pipeline" in filepath:
        content = content.replace(
            'map_data = load_keypoints3d_map()',
            'map_data = load_keypoints3d_map(config["paths"]["keypoints3d_map"])'
        )
        content = content.replace(
            'learnable_cfg.setdefault("j_regressor_body25", paths.get("j_regressor_body25"))',
            'learnable_cfg.setdefault("j_regressor_body25", "models/J_regressor_body25.npy")'
        )
    elif "executor.py" in filepath and "fusion_pipeline" in filepath:
        content = content.replace(
            'def _fuse_frame(data,',
            'def _fuse_frame(data, map_path,'
        )
        content = content.replace(
            'map_data = load_keypoints3d_map()',
            'map_data = load_keypoints3d_map(map_path)'
        )
        content = content.replace(
            '_fuse_frame(data=copy.deepcopy(fusion_data),',
            '_fuse_frame(data=copy.deepcopy(fusion_data), map_path=config["paths"]["keypoints3d_map"],'
        )
    elif "executor.py" in filepath and "optimization_pipeline" in filepath:
        content = content.replace(
            'map_data = load_keypoints3d_map()',
            'map_data = load_keypoints3d_map(config["paths"]["keypoints3d_map"])'
        )
    elif "smpl_runner.py" in filepath and "pose_pipeline" in filepath:
        # We need to pass config or map_path to smpl_runner. 
        # But maybe we just hardcode the config reading or pass it.
        pass
    elif "offset_paper.py" in filepath and "preprocess_pipeline" in filepath:
        content = content.replace(
            'def compute_offset(cam1_path, cam2_path, out_dir):',
            'def compute_offset(cam1_path, cam2_path, out_dir, map_path):'
        )
        content = content.replace(
            'map_data = load_keypoints3d_map()',
            'map_data = load_keypoints3d_map(map_path)'
        )
    elif "executor.py" in filepath and "refinement_pipeline" in filepath:
        content = content.replace(
            'map_data = load_keypoints3d_map()',
            'map_data = load_keypoints3d_map(config["paths"]["keypoints3d_map"])'
        )
    elif "extract_2d.py" in filepath and "preprocess_pipeline" in filepath:
        content = content.replace(
            'def _load_keypoints2d_map(map_path: Path | None = None) -> dict[str, object]:\n    return load_keypoints2d_map()',
            'def _load_keypoints2d_map(map_path: Path | None = None) -> dict[str, object]:\n    return load_keypoints2d_map(str(map_path) if map_path else "configs/keypoints2D_map.yml")'
        )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

for f in [
    "learnable_pipeline/executor.py",
    "fusion_pipeline/executor.py",
    "optimization_pipeline/executor.py",
    "refinement_pipeline/executor.py",
    "preprocess_pipeline/offset_paper.py",
    "preprocess_pipeline/extract_2d.py"
]:
    fix_file(f)
