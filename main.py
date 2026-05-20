import argparse
from compat import patch_numpy_and_inspect, configure_stdout_encoding
from config_loader import load_config
from pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Pose + Fusion + Learnable-SMPLify Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--stage",
        choices=["all", "pose", "fusion", "learnable", "pose_fusion"],
        default=None,
        help="Override runtime.stage in YAML",
    )
    return parser.parse_args()


def main():
    configure_stdout_encoding()
    patch_numpy_and_inspect()
    args = parse_args()
    config = load_config(args.config)
    if args.stage:
        config.setdefault("runtime", {})["stage"] = args.stage
    run_pipeline(config)


if __name__ == "__main__":
    main()
