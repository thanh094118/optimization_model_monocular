from pose_pipeline.stage import run_pose_export
from fusion_pipeline.stage import run_fusion
from learnable_pipeline.stage import run_learnable_smplify
from visualization_pipeline.stage import run_visualization
from evaluation_pipeline.stage import run_evaluation
from refinement_pipeline.stage import run_refinement_optimization
from preprocess_pipeline.extract_image import extract_images, run_offset_estimation
from preprocess_pipeline.offset_selector import resolve_selected_offset


def run_pipeline(config):
    stage = config.get("runtime", {}).get("stage", "all")

    # Always run preprocess first. Existing extracted folders are skipped by default.
    extract_images(
        input_folder="input",
        output_folder="output/preprocess_results",
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        restart=False,
        debug=False,
    )
    run_offset_estimation(
        input_folder="input",
        output_folder="output/preprocess_results",
        smpl_model_path="models/SMPL_NEUTRAL.pkl",
    )

    selected_offset, selected_method, selected_path = resolve_selected_offset(config)
    config.setdefault("runtime", {})["selected_offset"] = selected_offset
    config["runtime"]["offset_method"] = selected_method
    print(f"[Pipeline] Selected offset method={selected_method}, offset={selected_offset}, file={selected_path}")

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
