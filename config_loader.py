from pathlib import Path
import yaml


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

DEFAULT_EVALUATION_METRICS = {
    "pa_mpjpe": True,
    "mpjpe": False,
    "pck": False,
}


def resolve_inputs(config):
    inputs = dict(config.get("inputs") or {})
    paths = config.get("paths", {})
    for key in INPUT_KEYS:
        if key not in inputs and key in paths:
            inputs[key] = paths[key]
    return inputs


def load_config(config_path):
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError("Config file not found: {}".format(path))

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    evaluation_cfg = config.setdefault("evaluation", {})
    metrics_cfg = evaluation_cfg.setdefault("metrics", {})
    for key, value in DEFAULT_EVALUATION_METRICS.items():
        metrics_cfg.setdefault(key, value)

    validate_config(config)
    return config


def validate_config(config):
    if "runtime" not in config:
        raise ValueError("Missing config section: runtime")

    if "paths" not in config:
        raise ValueError("Missing config section: paths")

    stage = config.get("runtime", {}).get("stage", "visualization")
    if stage not in ALLOWED_STAGES:
        raise ValueError(
            "Invalid runtime.stage={!r}. Allowed: {}".format(
                stage, sorted(ALLOWED_STAGES)
            )
        )

    paths = config["paths"]
    inputs = resolve_inputs(config)

    required_paths = [
        "smpl_model",
        "keypoints3d_map",
        "keypoints2d_map",
        "mapping_3dto2d",
        "j_regressor_3d",
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
