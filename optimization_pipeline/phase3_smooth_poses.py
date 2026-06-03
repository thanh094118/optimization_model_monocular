#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml


PHASE_KEY = "phase3_smooth_poses"


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


def smooth_poses(poses, window_size):
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2:
        raise ValueError("params['poses'] must have shape [frames, pose_dim]")
    if poses.shape[0] == 0 or window_size <= 0:
        return poses.copy()

    padding_before = np.repeat(poses[:1], window_size, axis=0)
    padding_after = np.repeat(poses[-1:], window_size, axis=0)
    poses_full = np.vstack([padding_before, poses, padding_after])
    smoothed = poses.copy()
    n_frames = poses.shape[0]

    for width in range(1, window_size + 1):
        smoothed += poses_full[window_size - width:window_size - width + n_frames]
        smoothed += poses_full[window_size + width:window_size + width + n_frames]

    smoothed /= (2 * window_size + 1)
    return smoothed.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: smooth SMPL poses")
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
    window_size = int(phase_config["smooth"]["window_size"])

    params_blob = load_data(input_path)
    params = extract_params(params_blob)
    params["poses"] = smooth_poses(params["poses"], window_size=window_size)

    save_pickle(output_path, {"params": params})
    print("[{}] wrote {}".format(PHASE_KEY, output_path))


if __name__ == "__main__":
    main()
