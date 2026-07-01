from pathlib import Path

from pose_pipeline.stage import run_pose_export
from fusion_pipeline.stage import run_fusion
from learnable_pipeline.stage import run_learnable_smplify
from optimization_pipeline.stage import run_optimization
from visualization_pipeline.stage import run_visualization
from evaluation_pipeline.stage import run_evaluation
from refinement_pipeline.stage import run_refinement_optimization
from preprocess_pipeline.stage import run_preprocess


def _evaluation_inputs_ready(config):
    paths = config.get("paths", {})
    module_dirs = [
        Path(paths["pose_output_dir"]) / "keypoints3d",
        Path(paths["fused_output_dir"]) / "keypoints3d",
        Path(paths["learnable_output_dir"]) / "keypoints3d",
        Path(paths["optimized_output_dir"]) / "keypoints3d",
    ]
    for module_dir in module_dirs:
        if not module_dir.exists():
            return False
        if not any(module_dir.glob("*.json")):
            return False
    return True


def run_pipeline(config):
    stage = config.get("runtime", {}).get("stage", "all")

    if stage == "evaluation":
        if _evaluation_inputs_ready(config):
            print("[Pipeline] Evaluation inputs already available. Skipping preprocess/pose/fusion/learnable/optimization.")
        else:
            print("[Pipeline] Missing evaluation inputs. Running prerequisite stages without extract-frame work.")
            run_preprocess(config, extract_frames=False)
            selected_offset = config.get("runtime", {}).get("selected_offset")

            print(f"[Pipeline] Running pose with offset={selected_offset}")
            run_pose_export(config)

            print(f"[Pipeline] Running fusion with offset={selected_offset}")
            run_fusion(config)

            print(f"[Pipeline] Running learnable with offset={selected_offset}")
            run_learnable_smplify(config)

            print(f"[Pipeline] Running optimization with offset={selected_offset}")
            run_optimization(config)

        print("[Pipeline] Running evaluation only")
        run_evaluation(config)
        return

    selected_offset, _ = run_preprocess(config)

    if stage in ("all", "all_vis", "pose"):
        print(f"[Pipeline] Running pose with offset={selected_offset}")
        run_pose_export(config)

    if stage in ("all", "all_vis", "fusion"):
        print(f"[Pipeline] Running fusion with offset={selected_offset}")
        run_fusion(config)

    if stage in ("all", "all_vis", "learnable"):
        print(f"[Pipeline] Running learnable with offset={selected_offset}")
        run_learnable_smplify(config)

    if stage in ("all", "all_vis", "optimization"):
        print(f"[Pipeline] Running optimization with offset={selected_offset}")
        run_optimization(config)

    if stage in ("all", "all_vis", "evaluation"):
        print(f"[Pipeline] Running evaluation with offset={selected_offset}")
        run_evaluation(config)

    if stage in ("all", "all_vis", "visualization"):
        print(f"[Pipeline] Running visualization with offset={selected_offset}")
        run_visualization(config)

    if stage == "refinement":
        print(f"[Pipeline] Running refinement with offset={selected_offset}")
        run_refinement_optimization(config)
