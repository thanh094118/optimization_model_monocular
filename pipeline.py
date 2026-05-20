from pose_pipeline.stage import run_pose_export
from fusion_pipeline.stage import run_fusion
from learnable_pipeline.stage import run_learnable_smplify
from visualization_pipeline.stage import run_visualization
from refirement_pipeline.stage import run_refirement_optimization


def run_pipeline(config):
    stage = config.get("runtime", {}).get("stage", "all")

    if stage in ("all", "pose"):
        run_pose_export(config)

    if stage in ("all", "fusion", "pose_fusion"):
        run_fusion(config)

    if stage in ("all", "learnable", "postprocess", "learnable_visualization"):
        run_learnable_smplify(config)

    if stage in ("all", "visualization", "postprocess", "learnable_visualization"):
        run_visualization(config)

    if stage == "refirement":
        run_refirement_optimization(config)