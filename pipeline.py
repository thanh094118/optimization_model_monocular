from pose_pipeline.stage import run_pose_export
from fusion_pipeline.stage import run_fusion
from learnable_pipeline.stage import run_learnable_smplify
from visualization_pipeline.stage import run_visualization
from evaluation_pipeline.stage import run_evaluation
from refinement_pipeline.stage import run_refinement_optimization
from preprocess_pipeline.extract_image import extract_images


def run_pipeline(config):
    stage = config.get("runtime", {}).get("stage", "all")

    # Always run preprocess first. Existing extracted folders are skipped by default.
    extract_images(
        input_folder="input",
        output_folder="output",
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        restart=False,
        debug=False,
    )

    if stage in ("all", "pose"):
        run_pose_export(config)

    if stage in ("all", "fusion"):
        run_fusion(config)

    if stage in ("all", "learnable"):
        run_learnable_smplify(config)

    if stage in ("all", "evaluation"):
        run_evaluation(config)

    if stage in ("all", "visualization"):
        run_visualization(config)

    if stage == "refinement":
        run_refinement_optimization(config)
