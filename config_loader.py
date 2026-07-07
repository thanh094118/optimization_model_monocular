import yaml
from pathlib import Path


ALLOWED_STAGES = {
    "visualization",
    "evaluation",
}

INPUT_KEYS = (
    "cam1_pkl",
    "cam2_pkl",
    "camera1_video",
    "camera2_video",
    "ground_truth_dir",
)

def resolve_inputs(config):
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Missing config section: inputs")
    for key in INPUT_KEYS:
        if key not in inputs or not inputs[key]:
            raise ValueError("Missing config input: inputs.{}".format(key))
    return inputs


def resolve_preprocess_output_dir(config):
    paths = config.get("paths")
    if not isinstance(paths, dict) or not paths.get("preprocess_output_dir"):
        raise ValueError("Missing config path: paths.preprocess_output_dir")
    return paths["preprocess_output_dir"]


def load_config(config_path):
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError("Config file not found: {}".format(path))

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    validate_config(config)
    return config


def validate_config(config):
    if "runtime" not in config:
        raise ValueError("Missing config section: runtime")

    if "paths" not in config:
        raise ValueError("Missing config section: paths")

    runtime_cfg = config["runtime"]
    for key in ("stage", "clean_output"):
        if key not in runtime_cfg or runtime_cfg[key] is None:
            raise ValueError("Missing config runtime parameter: runtime.{}".format(key))

    stage = runtime_cfg["stage"]
    if stage not in ALLOWED_STAGES:
        raise ValueError(
            "Invalid runtime.stage={!r}. Allowed: {}".format(
                stage, sorted(ALLOWED_STAGES)
            )
        )

    paths = config["paths"]
    inputs = resolve_inputs(config)
    evaluation_cfg = config.get("evaluation")
    if not isinstance(evaluation_cfg, dict):
        raise ValueError("Missing config section: evaluation")
    metrics_cfg = evaluation_cfg.get("metrics")
    if not isinstance(metrics_cfg, dict):
        raise ValueError("Missing config section: evaluation.metrics")

    required_paths = [
        "smpl_model",
        "keypoints3d_map",
        "keypoints2d_map",
        "j_regressor_3d",
        "segmentation",
        "preprocess_output_dir",
        "pose_output_dir",
        "fused_output_dir",
        "learnable_output_dir",
        "visualization_output_dir",
        "evaluation_output_dir",
    ]

    for key in required_paths:
        if key not in paths:
            raise ValueError("Missing config path: paths.{}".format(key))

    for key in INPUT_KEYS:
        if key not in inputs or not inputs[key]:
            raise ValueError("Missing config input: inputs.{}".format(key))

    fusion_cfg = config.get("fusion", {})
    for key in ("enabled", "belief", "occlusion", "ransac", "optimization"):
        if key not in fusion_cfg:
            raise ValueError("Missing config section: fusion.{}".format(key))

    if "enabled" not in evaluation_cfg:
        raise ValueError("Missing config section: evaluation.enabled")

    learnable_cfg = config.get("learnable")
    if not isinstance(learnable_cfg, dict):
        raise ValueError("Missing config section: learnable")
    if "enabled" not in learnable_cfg:
        raise ValueError("Missing config section: learnable.enabled")
    if "checkpoint" not in learnable_cfg or not learnable_cfg["checkpoint"]:
        raise ValueError("Missing config learnable parameter: learnable.checkpoint")

    visualization_cfg = config.get("visualization")
    if not isinstance(visualization_cfg, dict):
        raise ValueError("Missing config section: visualization")
    if "enabled" not in visualization_cfg:
        raise ValueError("Missing config section: visualization.enabled")
    for key in ("target_fps", "dpi", "max_frames", "cameras"):
        if key not in visualization_cfg:
            raise ValueError("Missing config visualization parameter: visualization.{}".format(key))

    belief_cfg = fusion_cfg["belief"]
    for key in ("alpha", "beta"):
        if key not in belief_cfg or belief_cfg[key] is None:
            raise ValueError("Missing config fusion belief parameter: fusion.belief.{}".format(key))

    occlusion_cfg = fusion_cfg["occlusion"]
    for key in ("enabled", "tau"):
        if key not in occlusion_cfg or occlusion_cfg[key] is None:
            raise ValueError("Missing config fusion occlusion parameter: fusion.occlusion.{}".format(key))

    ransac_cfg = fusion_cfg["ransac"]
    for key in ("threshold", "max_combos"):
        if key not in ransac_cfg or ransac_cfg[key] is None:
            raise ValueError("Missing config fusion ransac parameter: fusion.ransac.{}".format(key))

    opt_cfg = fusion_cfg["optimization"]
    for key in ("regularization", "regularization_lambda", "temporal_lambda", "max_iter"):
        if key not in opt_cfg or opt_cfg[key] is None:
            raise ValueError("Missing config fusion optimization parameter: fusion.optimization.{}".format(key))

    for key in ("pa_mpjpe", "mpjpe", "pck"):
        if key not in metrics_cfg or metrics_cfg[key] is None:
            raise ValueError("Missing config evaluation metric flag: evaluation.metrics.{}".format(key))
