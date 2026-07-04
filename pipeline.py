from pathlib import Path

from pose_pipeline import run_pose_export
from fusion_pipeline import run_fusion
from learnable_pipeline import run_learnable_smplify
from visualization_pipeline import run_visualization
from evaluation_pipeline import run_evaluation
from preprocess_pipeline.stage import run_preprocess


def _evaluation_input_dirs(config):
    paths = config.get("paths", {})
    return [
        Path(paths["pose_output_dir"]) / "keypoints3d",
        Path(paths["fused_output_dir"]) / "keypoints3d",
        Path(paths["learnable_output_dir"]) / "keypoints3d",
    ]


def _require_evaluation_inputs(config):
    missing = []
    for module_dir in _evaluation_input_dirs(config):
        if not module_dir.exists() or not any(module_dir.glob("*.json")):
            missing.append(str(module_dir))
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Evaluation stage requires existing pose/fusion/learnable outputs. "
            f"Missing or empty directories: {joined}"
        )


def _run_full_pipeline(config, include_visualization: bool) -> None:
    selected_offset, _ = run_preprocess(config)
    print(f"[Pipeline] Running pose with offset={selected_offset}")
    run_pose_export(config)
    print(f"[Pipeline] Running fusion with offset={selected_offset}")
    run_fusion(config)
    print(f"[Pipeline] Running learnable with offset={selected_offset}")
    run_learnable_smplify(config)
    print(f"[Pipeline] Running evaluation with offset={selected_offset}")
    run_evaluation(config)
    if include_visualization:
        print(f"[Pipeline] Running visualization with offset={selected_offset}")
        run_visualization(config)


def run_pipeline(config, stage_override=None):
    if stage_override == "evaluation":
        _require_evaluation_inputs(config)
        print("[Pipeline] Running evaluation only")
        run_evaluation(config)
        return

    if stage_override is None:
        _run_full_pipeline(config, include_visualization=False)
        return

    if stage_override == "visualization":
        _run_full_pipeline(config, include_visualization=True)
        return

    raise ValueError(f"Unsupported runtime.stage: {stage_override}")
