from pose_pipeline.stage import run_pose_export
from fusion_pipeline.stage import run_fusion
from learnable_pipeline.stage import run_learnable_smplify
from visualization_pipeline.stage import run_visualization
from evaluation_pipeline.stage import run_evaluation
from refinement_pipeline.stage import run_refinement_optimization
from preprocess_pipeline.stage import run_preprocess


def run_pipeline(config):
    stage = config.get("runtime", {}).get("stage", "all")

    selected_offset, _ = run_preprocess(config)

    if stage in ("all", "pose"):
        print(f"[Pipeline] Running pose with offset={selected_offset}")
        run_pose_export(config)

    if stage in ("all", "fusion"):
        print(f"[Pipeline] Running fusion with offset={selected_offset}")
        run_fusion(config)

    if stage in ("all", "learnable"):
        print(f"[Pipeline] Running learnable with offset={selected_offset}")
        run_learnable_smplify(config)

    if stage in ("all", "evaluation"):
        print(f"[Pipeline] Running evaluation with offset={selected_offset}")
        run_evaluation(config)

    if stage in ("all", "visualization"):
        print(f"[Pipeline] Running visualization with offset={selected_offset}")
        run_visualization(config)

    if stage == "refinement":
        print(f"[Pipeline] Running refinement with offset={selected_offset}")
        run_refinement_optimization(config)
