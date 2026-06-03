#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml


PHASE_KEY = "phase2_init_translation"


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


def extract_keypoints(data):
    if isinstance(data, dict):
        if "keypoints" in data:
            data = data["keypoints"]
        elif "keypoints2d" in data:
            data = data["keypoints2d"]
    keypoints = np.asarray(data, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[-1] < 3:
        raise ValueError("2D keypoints must have shape [frames, joints, >=3]")
    return keypoints[..., :3]


def extract_cameras(data, n_frames):
    if isinstance(data, dict) and "cameras" in data:
        data = data["cameras"]
    if not isinstance(data, dict) or "K" not in data:
        raise ValueError("Camera input must be a dict containing key 'K'")
    k_array = np.asarray(data["K"], dtype=np.float32)
    if k_array.ndim == 2:
        k_array = np.repeat(k_array[None, ...], n_frames, axis=0)
    if k_array.ndim != 3 or k_array.shape[1:] != (3, 3):
        raise ValueError("Camera intrinsics K must have shape [frames, 3, 3] or [3, 3]")
    if k_array.shape[0] != n_frames:
        raise ValueError("Camera frames {} do not match keypoint frames {}".format(k_array.shape[0], n_frames))
    return {"K": k_array}


def ensure_frame_count(array, n_frames):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] == n_frames:
        return array
    if array.shape[0] == 1:
        return np.repeat(array, n_frames, axis=0)
    raise ValueError("Cannot broadcast array with shape {} to {} frames".format(array.shape, n_frames))


class SmplBody25Model:
    def __init__(self, smpl_model_path, j_regressor_path, device):
        try:
            import smplx
        except ImportError as exc:
            raise ImportError(
                "smplx is required at runtime for {}. Activate the project environment first.".format(PHASE_KEY)
            ) from exc

        self.device = device
        self.model = smplx.create(
            str(smpl_model_path),
            model_type="smpl",
            use_pca=False,
            flat_hand_mean=True,
            gender="neutral",
        ).to(device).eval()
        regressor = np.load(j_regressor_path)
        self.regressor = torch.tensor(regressor, dtype=torch.float32, device=device)

    def keypoints(self, params):
        rh = ensure_frame_count(params["Rh"], params["poses"].shape[0])
        th = ensure_frame_count(params["Th"], params["poses"].shape[0])
        poses = ensure_frame_count(params["poses"], params["poses"].shape[0])
        shapes = ensure_frame_count(params["shapes"], params["poses"].shape[0])

        rh_t = torch.tensor(rh, dtype=torch.float32, device=self.device)
        th_t = torch.tensor(th, dtype=torch.float32, device=self.device)
        poses_t = torch.tensor(poses, dtype=torch.float32, device=self.device)
        shapes_t = torch.tensor(shapes, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            output = self.model(
                betas=shapes_t,
                global_orient=rh_t,
                body_pose=poses_t,
                transl=th_t,
                return_verts=True,
            )
            joints = torch.einsum("jv,bvc->bjc", self.regressor, output.vertices)
        return joints.detach().cpu().numpy().astype(np.float32)


def solve_translation(points3d, keypoints2d, intrinsics):
    a_matrix = np.zeros((2 * points3d.shape[0], 3), dtype=np.float32)
    b_vector = np.zeros((2 * points3d.shape[0], 1), dtype=np.float32)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    for joint_idx in range(points3d.shape[0]):
        conf = float(keypoints2d[joint_idx, 2])
        if conf <= 0:
            continue
        a_matrix[2 * joint_idx, 0] = 1.0
        a_matrix[2 * joint_idx + 1, 1] = 1.0
        a_matrix[2 * joint_idx, 2] = -(keypoints2d[joint_idx, 0] - cx) / fx
        a_matrix[2 * joint_idx + 1, 2] = -(keypoints2d[joint_idx, 1] - cy) / fy
        b_vector[2 * joint_idx, 0] = points3d[joint_idx, 2] * (keypoints2d[joint_idx, 0] - cx) / fx - points3d[joint_idx, 0]
        b_vector[2 * joint_idx + 1, 0] = points3d[joint_idx, 2] * (keypoints2d[joint_idx, 1] - cy) / fy - points3d[joint_idx, 1]
        a_matrix[2 * joint_idx:2 * joint_idx + 2] *= conf
        b_vector[2 * joint_idx:2 * joint_idx + 2] *= conf

    normal_matrix = a_matrix.T @ a_matrix
    rhs = a_matrix.T @ b_vector
    return np.linalg.pinv(normal_matrix) @ rhs


def main():
    parser = argparse.ArgumentParser(description="Phase 2: initialize translation from 2D keypoints")
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
    input_params_path = resolve_path(base_dir, phase_config["inputs"]["params_path"])
    keypoints_path = resolve_path(base_dir, phase_config["inputs"]["keypoints_path"])
    cameras_path = resolve_path(base_dir, phase_config["inputs"]["cameras_path"])
    output_path = resolve_path(base_dir, phase_config["output"]["params_path"])

    device_name = phase_config["model"].get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    params_blob = load_data(input_params_path)
    params = extract_params(params_blob)
    keypoints = extract_keypoints(load_data(keypoints_path))
    cameras = extract_cameras(load_data(cameras_path), n_frames=keypoints.shape[0])

    params["Rh"] = ensure_frame_count(params["Rh"], keypoints.shape[0])
    params["Th"] = ensure_frame_count(params["Th"], keypoints.shape[0])
    params["poses"] = ensure_frame_count(params["poses"], keypoints.shape[0])
    params["shapes"] = ensure_frame_count(params["shapes"], keypoints.shape[0])

    model = SmplBody25Model(
        smpl_model_path=resolve_path(base_dir, phase_config["model"]["smpl_model_path"]),
        j_regressor_path=resolve_path(base_dir, phase_config["model"]["j_regressor_path"]),
        device=device,
    )

    num_keypoints = int(phase_config["model"].get("num_keypoints", 15))
    min_conf_sum = float(phase_config["thresholds"].get("min_conf_sum", num_keypoints / 2.0))

    params["Th"] = np.zeros_like(params["Th"], dtype=np.float32)
    joints3d = model.keypoints(params)

    for frame_idx in range(joints3d.shape[0]):
        keypoints_frame = keypoints[frame_idx, :num_keypoints]
        if float(keypoints_frame[:, 2].sum()) < min_conf_sum:
            if frame_idx > 0:
                params["Th"][frame_idx] = params["Th"][frame_idx - 1]
            continue
        translation = solve_translation(
            points3d=joints3d[frame_idx, :num_keypoints],
            keypoints2d=keypoints_frame,
            intrinsics=cameras["K"][frame_idx],
        )
        params["Th"][frame_idx] += translation[:, 0]

    save_pickle(output_path, {"params": params})
    print("[{}] wrote {}".format(PHASE_KEY, output_path))


if __name__ == "__main__":
    main()
