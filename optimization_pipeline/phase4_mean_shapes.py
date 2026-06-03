#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml


PHASE_KEY = "phase4_mean_shapes"


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_base_dir(config_path, config):
    base_dir = config.get("root_dir", ".")
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = (Path(config_path).resolve().parent / base_path).resolve()
    return base_path


def resolve_path(base_dir, value):
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_data(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as file:
            return pickle.load(file)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    raise ValueError("Unsupported input format: {}".format(path))


def save_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(payload, file)


def extract_params(data):
    if isinstance(data, dict) and "params" in data:
        return data["params"]
    if isinstance(data, dict):
        return data
    raise ValueError("Params input must be a dict or contain a 'params' key")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: share mean SMPL shape")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("configs.yml")),
        help="Path to optimization config YAML",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    phase_config = config.get(PHASE_KEY)
    if not phase_config:
        raise KeyError("Missing config section: {}".format(PHASE_KEY))

    base_dir = resolve_base_dir(args.config, config)
    input_path = resolve_path(base_dir, phase_config["inputs"]["params_path"])
    output_path = resolve_path(base_dir, phase_config["output"]["params_path"])

    axis = int(phase_config["mean_shape"].get("axis", 0))
    keepdims = bool(phase_config["mean_shape"].get("keepdims", True))

    params_blob = load_data(input_path)
    params = extract_params(params_blob)

    shapes = np.asarray(params["shapes"], dtype=np.float32)
    params["shapes"] = shapes.mean(axis=axis, keepdims=keepdims).astype(np.float32)

    save_pickle(output_path, {"params": params})
    print("[{}] wrote {}".format(PHASE_KEY, output_path))


if __name__ == "__main__":
    main()
