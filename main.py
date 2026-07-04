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
            "visualization",
            "evaluation",
        ],
        help="Override runtime.stage in YAML",
    )
    return parser.parse_args()


def main():
    configure_stdout_encoding()
    patch_numpy_and_inspect()

    args = parse_args()
    config = load_config(args.config)

    run_pipeline(config, stage_override=args.stage)


if __name__ == "__main__":
    main()
