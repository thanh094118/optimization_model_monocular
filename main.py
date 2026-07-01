import argparse

from compat import configure_stdout_encoding, patch_numpy_and_inspect
from config_loader import load_config
from pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Motion Pipeline")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument(
        "--stage",
        default=None,
        choices=[
            "all",
            "all_vis",
            "preprocess",
            "pose",
            "fusion",
            "learnable",
            "optimization",
            "visualization",
            "evaluation",
            "refinement",
        ],
        help="Override runtime.stage in YAML",
    )
    return parser.parse_args()


def main():
    configure_stdout_encoding()
    patch_numpy_and_inspect()

    args = parse_args()
    config = load_config(args.config)

    if args.stage:
        config["runtime"]["stage"] = args.stage

    run_pipeline(config)


if __name__ == "__main__":
    main()
