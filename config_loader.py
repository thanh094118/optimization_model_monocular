from pathlib import Path
import yaml


ALLOWED_STAGES = {
    "all",
    "pose",
    "fusion",
    "learnable",
    "visualization",
    "pose_fusion",
    "postprocess",
    "learnable_visualization",
    "refinement",
}


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

    stage = config.get("runtime", {}).get("stage", "all")
    if stage not in ALLOWED_STAGES:
        raise ValueError(
            "Invalid runtime.stage={!r}. Allowed: {}".format(
                stage, sorted(ALLOWED_STAGES)
            )
        )

    paths = config["paths"]

    required_paths = [
        "cam1_pkl",
        "cam2_pkl",
        "smpl_model",
        "pose_output_dir",
        "fused_output_dir",
        "learnable_output_dir",
        "visualization_output_dir",
        "refinement_output_dir",
    ]

    for key in required_paths:
        if key not in paths:
            raise ValueError("Missing config path: paths.{}".format(key))

    if "refinement" in config:
        required_refinement = [
            "data_dir",
            "wham_file",
            "video",
            "intrinsics",
            "subject_params",
            "parameters_yaml",
        ]

        for key in required_refinement:
            if key not in config["refinement"]:
                raise ValueError("Missing config key: refinement.{}".format(key))
