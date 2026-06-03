#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml


PHASE_KEY = "phase6_refine_poses"


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


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


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


def expand_numpy_frames(array, n_frames):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] == n_frames:
        return array
    if array.shape[0] == 1:
        return np.repeat(array, n_frames, axis=0)
    raise ValueError("Cannot broadcast array with shape {} to {} frames".format(array.shape, n_frames))


def expand_tensor_frames(tensor, n_frames):
    if tensor.shape[0] == n_frames:
        return tensor
    if tensor.shape[0] == 1:
        repeats = [n_frames] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeats)
    raise ValueError("Cannot broadcast tensor with shape {} to {} frames".format(tuple(tensor.shape), n_frames))


def extract_cameras(data, n_frames):
    if isinstance(data, dict) and "cameras" in data:
        data = data["cameras"]
    if not isinstance(data, dict) or "K" not in data:
        raise ValueError("Camera input must be a dict containing key 'K'")

    k_array = np.asarray(data["K"], dtype=np.float32)
    r_array = np.asarray(data.get("R", np.eye(3, dtype=np.float32)), dtype=np.float32)
    t_array = np.asarray(data.get("T", np.zeros((3,), dtype=np.float32)), dtype=np.float32)

    if k_array.ndim == 2:
        k_array = np.repeat(k_array[None, ...], n_frames, axis=0)
    if r_array.ndim == 2:
        r_array = np.repeat(r_array[None, ...], n_frames, axis=0)
    if t_array.ndim == 1:
        t_array = np.repeat(t_array[None, ...], n_frames, axis=0)

    return {
        "K": expand_numpy_frames(k_array, n_frames),
        "R": expand_numpy_frames(r_array, n_frames),
        "T": expand_numpy_frames(t_array, n_frames),
    }


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

    def __call__(self, params):
        n_frames = params["poses"].shape[0]
        rh = expand_tensor_frames(params["Rh"], n_frames)
        th = expand_tensor_frames(params["Th"], n_frames)
        poses = expand_tensor_frames(params["poses"], n_frames)
        shapes = expand_tensor_frames(params["shapes"], n_frames)

        output = self.model(
            betas=shapes,
            global_orient=rh,
            body_pose=poses,
            transl=th,
            return_verts=True,
        )
        joints = torch.einsum("jv,bvc->bjc", self.regressor, output.vertices)
        return {"keypoints": joints}


def project_points(points3d, cameras):
    rotated = torch.einsum("bij,bkj->bki", cameras["R"], points3d) + cameras["T"][:, None, :]
    projected = torch.einsum("bij,bkj->bki", cameras["K"], rotated)
    return projected[..., :2] / projected[..., 2:].clamp_min(1e-8)


def gmof_loss(estimate, target, confidence, rho):
    squared = torch.sum((estimate - target) ** 2, dim=-1)
    robust = squared / (squared + float(rho) ** 2)
    return torch.sum(robust * confidence) / (confidence.sum() + 1e-5)


def smooth_sequence_loss(value, window_weight, order):
    if value.shape[0] <= 1:
        return value.new_tensor(0.0)
    loss = value.new_tensor(0.0)
    for width, weight in enumerate(window_weight, start=1):
        if value.shape[0] <= width:
            continue
        velocity = value[width:] - value[:-width]
        if order == 2:
            if velocity.shape[0] <= 1:
                continue
            velocity = velocity[1:] - velocity[:-1]
        squared = torch.sum(velocity ** 2, dim=-1)
        loss = loss + float(weight) * squared.mean()
    return loss


def depth_smooth_loss(translations, cameras, window_weight, order):
    depth_values = torch.einsum("bi,bij->bj", translations, cameras["R"].transpose(1, 2))[..., 2:3]
    return smooth_sequence_loss(depth_values, window_weight=window_weight, order=order)


def to_tensor_dict(params, cameras, keypoints, device):
    params_t = {}
    n_frames = keypoints.shape[0]
    for key, value in params.items():
        params_t[key] = torch.tensor(expand_numpy_frames(value, n_frames), dtype=torch.float32, device=device)
    cameras_t = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in cameras.items()
    }
    keypoints_t = torch.tensor(keypoints, dtype=torch.float32, device=device)
    return params_t, cameras_t, keypoints_t


def load_gmm_prior(gmm_path, start, end, device):
    with Path(gmm_path).open("rb") as file:
        gmm = pickle.load(file, encoding="latin1")

    means = np.asarray(gmm["means"], dtype=np.float32)
    covars = np.asarray(gmm["covars"], dtype=np.float32)
    weights = np.asarray(gmm["weights"], dtype=np.float32)
    precisions = np.stack([np.linalg.inv(cov) for cov in covars]).astype(np.float32)

    sqrdets = np.array([np.sqrt(np.linalg.det(cov)) for cov in covars], dtype=np.float32)
    pose_dim = end - start
    const = (2 * np.pi) ** (pose_dim / 2.0)
    nll_weights = -np.log(weights / (const * (sqrdets / sqrdets.min())) + 1.0e-15)

    return {
        "means": torch.tensor(means[:, start:end], dtype=torch.float32, device=device),
        "precisions": torch.tensor(precisions[:, start:end, start:end], dtype=torch.float32, device=device),
        "nll_weights": torch.tensor(nll_weights[None, :], dtype=torch.float32, device=device),
    }


def gmm_prior_loss(poses, prior):
    diff = poses.unsqueeze(1) - prior["means"].unsqueeze(0)
    precision_product = torch.einsum("mij,bmj->bmi", prior["precisions"], diff)
    quadratic = (precision_product * diff).sum(dim=-1)
    likelihood = 0.5 * quadratic + prior["nll_weights"]
    return torch.min(likelihood, dim=1).values.mean()


def optimize_once(params, cameras, keypoints, model, phase_config, prior):
    optimizer_cfg = phase_config["optimizer"]
    repro_cfg = phase_config["reprojection"]
    smooth_cfg = phase_config["smooth"]
    init_cfg = phase_config["init_anchor"]

    window_weight = smooth_cfg["window_weight"]
    order = int(smooth_cfg.get("order", 2))
    init_pose_anchor = params["poses"].detach().clone()

    opt_tensors = [params["poses"], params["Rh"], params["Th"]]
    for tensor in opt_tensors:
        tensor.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        opt_tensors,
        lr=float(optimizer_cfg.get("lr", 1.0)),
        max_iter=int(optimizer_cfg.get("max_iter", 20)),
        line_search_fn="strong_wolfe",
        tolerance_grad=float(optimizer_cfg.get("tolerance_grad", 1.0e-7)),
        tolerance_change=float(optimizer_cfg.get("tolerance_change", 1.0e-7)),
    )

    max_outer_steps = int(optimizer_cfg.get("max_outer_steps", 1000))
    relative_tolerance = float(optimizer_cfg.get("relative_tolerance", 1.0e-7))
    last_loss = None

    def compute_components():
        outputs = model(params)
        pred_keypoints = outputs["keypoints"]
        n_joints = min(pred_keypoints.shape[1], keypoints.shape[1])
        pred_keypoints = pred_keypoints[:, :n_joints]
        target_keypoints = keypoints[:, :n_joints, :2]
        target_conf = keypoints[:, :n_joints, 2]

        pred_2d = project_points(pred_keypoints, cameras)
        repro_loss = gmof_loss(pred_2d, target_keypoints, target_conf, rho=repro_cfg["gm_rho"])
        th_linear = smooth_sequence_loss(params["Th"], window_weight=window_weight, order=order)
        th_depth = depth_smooth_loss(params["Th"], cameras, window_weight=window_weight, order=order)
        pose_smooth = smooth_sequence_loss(params["poses"], window_weight=window_weight, order=order)
        keypoint_smooth = smooth_sequence_loss(pred_keypoints, window_weight=window_weight, order=order)
        init_loss = torch.mean((params["poses"] - init_pose_anchor) ** 2)
        prior_loss = gmm_prior_loss(params["poses"], prior)

        total_loss = (
            float(repro_cfg["weight"]) * repro_loss
            + float(smooth_cfg["weight"]) * (
                float(smooth_cfg["th_linear_weight"]) * th_linear
                + float(smooth_cfg["th_depth_weight"]) * th_depth
                + float(smooth_cfg["pose_weight"]) * pose_smooth
                + float(smooth_cfg["keypoints_weight"]) * keypoint_smooth
            )
            + float(init_cfg["weight"]) * init_loss
            + float(phase_config["prior"]["weight"]) * prior_loss
        )
        return total_loss, {
            "reprojection": float(repro_loss.detach().cpu()),
            "smooth_th_linear": float(th_linear.detach().cpu()),
            "smooth_th_depth": float(th_depth.detach().cpu()),
            "smooth_pose": float(pose_smooth.detach().cpu()),
            "smooth_keypoints": float(keypoint_smooth.detach().cpu()),
            "init_anchor": float(init_loss.detach().cpu()),
            "prior": float(prior_loss.detach().cpu()),
        }

    def closure():
        optimizer.zero_grad()
        total_loss, _ = compute_components()
        total_loss.backward()
        return total_loss

    for _ in range(max_outer_steps):
        loss = optimizer.step(closure)
        if not torch.isfinite(loss):
            break
        current = float(loss.detach().cpu())
        if last_loss is not None:
            relative_change = (last_loss - current) / max(1.0e-5, abs(last_loss), abs(current))
            if relative_change <= relative_tolerance:
                break
        last_loss = current

    with torch.no_grad():
        final_loss, components = compute_components()

    return {
        "total_loss": float(final_loss.detach().cpu()),
        "loss_components": components,
        "optimized_keys": ["poses", "Rh", "Th"],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6: refine SMPL poses, root rotation, and translation")
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
    params_path = resolve_path(base_dir, phase_config["inputs"]["params_path"])
    keypoints_path = resolve_path(base_dir, phase_config["inputs"]["keypoints_path"])
    cameras_path = resolve_path(base_dir, phase_config["inputs"]["cameras_path"])
    output_path = resolve_path(base_dir, phase_config["output"]["params_path"])
    summary_path = resolve_path(base_dir, phase_config["output"]["summary_path"])

    device_name = phase_config["model"].get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    params = extract_params(load_data(params_path))
    keypoints = extract_keypoints(load_data(keypoints_path))
    cameras = extract_cameras(load_data(cameras_path), n_frames=keypoints.shape[0])
    params_t, cameras_t, keypoints_t = to_tensor_dict(params, cameras, keypoints, device=device)

    model = SmplBody25Model(
        smpl_model_path=resolve_path(base_dir, phase_config["model"]["smpl_model_path"]),
        j_regressor_path=resolve_path(base_dir, phase_config["model"]["j_regressor_path"]),
        device=device,
    )
    prior = load_gmm_prior(
        gmm_path=resolve_path(base_dir, phase_config["prior"]["gmm_path"]),
        start=int(phase_config["prior"]["start"]),
        end=int(phase_config["prior"]["end"]),
        device=device,
    )

    repeat = int(phase_config.get("repeat", 1))
    repeat_summaries = []
    for repeat_idx in range(repeat):
        summary = optimize_once(
            params=params_t,
            cameras=cameras_t,
            keypoints=keypoints_t,
            model=model,
            phase_config=phase_config,
            prior=prior,
        )
        summary["repeat_index"] = repeat_idx
        repeat_summaries.append(summary)

    params_out = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in params_t.items()}
    save_pickle(output_path, {"params": params_out})
    save_json(
        summary_path,
        {
            "phase": PHASE_KEY,
            "repeat": repeat,
            "summaries": repeat_summaries,
        },
    )
    print("[{}] wrote {}".format(PHASE_KEY, output_path))


if __name__ == "__main__":
    main()
