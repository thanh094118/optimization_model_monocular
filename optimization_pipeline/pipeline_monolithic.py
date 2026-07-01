#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml


PHASE_KEY = "pipeline"


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

    if k_array.ndim != 3 or k_array.shape[1:] != (3, 3):
        raise ValueError("Camera intrinsics K must have shape [frames, 3, 3] or [3, 3]")
    if k_array.shape[0] != n_frames:
        raise ValueError("Camera frames {} do not match keypoint frames {}".format(k_array.shape[0], n_frames))

    return {
        "K": k_array.astype(np.float32),
        "R": np.asarray(r_array, dtype=np.float32),
        "T": np.asarray(t_array, dtype=np.float32),
    }


def ensure_frame_count(array, n_frames):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] == n_frames:
        return array
    if array.shape[0] == 1:
        return np.repeat(array, n_frames, axis=0)
    raise ValueError("Cannot broadcast array with shape {} to {} frames".format(array.shape, n_frames))


def save_stage(path, params):
    save_pickle(path, {"params": params})


def ensure_frame_count_tensor(tensor, n_frames, device):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == n_frames:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(n_frames, *tensor.shape[1:])
    raise ValueError(f"Cannot broadcast tensor with shape {tensor.shape} to {n_frames} frames")

class SmplBody25Model:
    def __init__(self, smpl_model_path, j_regressor_path, device, model_cfg):
        try:
            import smplx
        except ImportError as exc:
            raise ImportError("smplx is required at runtime. Activate the project environment first.") from exc

        self.device = device
        self.model = smplx.create(
            str(smpl_model_path),
            model_type=str(model_cfg["model_type"]),
            use_pca=bool(model_cfg["use_pca"]),
            flat_hand_mean=bool(model_cfg["flat_hand_mean"]),
            gender=str(model_cfg["gender"]),
        ).to(device).eval()
        regressor = np.load(j_regressor_path)
        self.regressor = torch.tensor(regressor, dtype=torch.float32, device=device)

    def keypoints(self, params):
        # params can be numpy arrays or tensors. 
        # If they are numpy arrays, convert to tensors first.
        n_frames = params["poses"].shape[0]
        
        rh = params["Rh"]
        th = params["Th"]
        poses = params["poses"]
        shapes = params["shapes"]
        
        if not isinstance(rh, torch.Tensor):
            rh = torch.tensor(rh, dtype=torch.float32, device=self.device)
        if not isinstance(th, torch.Tensor):
            th = torch.tensor(th, dtype=torch.float32, device=self.device)
        if not isinstance(poses, torch.Tensor):
            poses = torch.tensor(poses, dtype=torch.float32, device=self.device)
        if not isinstance(shapes, torch.Tensor):
            shapes = torch.tensor(shapes, dtype=torch.float32, device=self.device)
            
        rh_t = ensure_frame_count_tensor(rh, n_frames, self.device)
        th_t = ensure_frame_count_tensor(th, n_frames, self.device)
        poses_t = ensure_frame_count_tensor(poses, n_frames, self.device)
        shapes_t = ensure_frame_count_tensor(shapes, n_frames, self.device)

        # We return the tensor here so gradients flow!
        output = self.model(
            betas=shapes_t,
            global_orient=rh_t,
            body_pose=poses_t,
            transl=th_t,
            return_verts=True,
        )
        joints = torch.einsum("jv,bvc->bjc", self.regressor, output.vertices)
        
        # If the inputs didn't require grad and we just want numpy (like before optimization),
        # we can just return the tensor, and the caller can detach/numpy it.
        # But wait! The old signature returned numpy array for `keypoints`. Let's keep it returning numpy if called directly as `keypoints` maybe?
        # Actually, let's just return numpy array here if inputs were numpy, else return tensor.
        if not isinstance(params["Rh"], torch.Tensor):
            return joints.detach().cpu().numpy().astype(np.float32)
        return joints

    def __call__(self, params):
        joints = self.keypoints(params)
        return {
            "keypoints": joints
        }


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


def run_phase2_init_translation(params, keypoints, cameras, model, phase2_cfg):
    params = params.copy()
    params["Th"] = np.zeros_like(params["Th"], dtype=np.float32)
    joints3d = model.keypoints(params)
    joint_indices = phase2_cfg.get("joint_indices")
    if joint_indices is None:
        num_keypoints = int(phase2_cfg["num_keypoints"])
        joint_indices = list(range(num_keypoints))
    else:
        joint_indices = [int(idx) for idx in joint_indices]
    min_conf_sum = float(phase2_cfg["min_conf_sum"])

    for frame_idx in range(joints3d.shape[0]):
        keypoints_frame = keypoints[frame_idx, joint_indices]
        if float(keypoints_frame[:, 2].sum()) < min_conf_sum:
            if frame_idx > 0:
                params["Th"][frame_idx] = params["Th"][frame_idx - 1]
            continue
        translation = solve_translation(
            points3d=joints3d[frame_idx, joint_indices],
            keypoints2d=keypoints_frame,
            intrinsics=cameras["K"][frame_idx],
        )
        params["Th"][frame_idx] += translation[:, 0]
    return params


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


def project_points(points3d, cameras):
    rotated = torch.einsum("bij,bkj->bki", cameras["R"], points3d) + cameras["T"][:, None, :]
    return rotated[..., :2] / rotated[..., 2:].clamp_min(1e-8)


def weighted_l2_loss(estimate, target, confidence):
    squared = torch.sum((estimate - target) ** 2, dim=-1)
    return torch.sum(squared * confidence) / (confidence.sum() + 1e-5)


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


def _slice_tensor_frame_range(tensor, start, end):
    if tensor.shape[0] == 1:
        return tensor
    return tensor[start:end]


def to_tensor_dict(params, cameras, keypoints, device):
    params_t = {}
    n_frames = keypoints.shape[0]
    for key, value in params.items():
        params_t[key] = torch.tensor(ensure_frame_count(value, n_frames), dtype=torch.float32, device=device)
    cameras_t = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in cameras.items()
    }
    keypoints_t = torch.tensor(keypoints, dtype=torch.float32, device=device)
    return params_t, cameras_t, keypoints_t


def load_gmm_prior(gmm_path, prior_cfg, device):
    with Path(gmm_path).open("rb") as file:
        gmm = pickle.load(file, encoding="latin1")

    means = np.asarray(gmm["means"], dtype=np.float32)
    covars = np.asarray(gmm["covars"], dtype=np.float32)
    weights = np.asarray(gmm["weights"], dtype=np.float32)
    precisions = np.stack([np.linalg.inv(cov) for cov in covars]).astype(np.float32)

    sqrdets = np.array([np.sqrt(np.maximum(np.linalg.det(cov), 1e-10)) for cov in covars], dtype=np.float32)
    start = int(prior_cfg["start"])
    end = int(prior_cfg["end"])
    pose_dim = end - start
    const = (2 * np.pi) ** (pose_dim / 2.0)
    
    min_sqrdet = np.maximum(sqrdets.min(), 1e-10)
    val = weights / (const * (sqrdets / min_sqrdet))
    nll_weights = -np.log(np.maximum(val, 1e-15))

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


def _optimize_phase5_window(params, cameras, keypoints, model, phase5_cfg):
    optimizer_cfg = phase5_cfg["optimizer"]
    repro_cfg = phase5_cfg["reprojection"]
    smooth_cfg = phase5_cfg["smooth"]

    joint_indices = repro_cfg["joint_indices"]
    window_weight = smooth_cfg["window_weight"]
    order = int(smooth_cfg["order"])

    opt_tensors = [params["Th"], params["Rh"]]
    for tensor in opt_tensors:
        tensor.requires_grad_(True)

    if str(optimizer_cfg["optim_type"]).lower() != "lbfgs":
        raise NotImplementedError("Only lbfgs optimizer is supported by this standalone pipeline")

    line_search_fn = optimizer_cfg.get("line_search_fn", "strong_wolfe")
    if not line_search_fn:
        line_search_fn = None

    optimizer = torch.optim.LBFGS(
        opt_tensors,
        lr=float(optimizer_cfg["lr"]),
        max_iter=int(optimizer_cfg["max_iter"]),
        line_search_fn=line_search_fn,
        tolerance_grad=float(optimizer_cfg["tolerance_grad"]),
        tolerance_change=float(optimizer_cfg["tolerance_change"]),
    )

    max_outer_steps = int(optimizer_cfg["max_outer_steps"])
    relative_tolerance = float(optimizer_cfg["relative_tolerance"])
    last_loss = None

    def compute_components():
        outputs = model(params)
        pred_keypoints = outputs["keypoints"][:, joint_indices, :]
        target_keypoints = keypoints[:, joint_indices, :2]
        target_conf = keypoints[:, joint_indices, 2]
        target_homo = torch.cat([target_keypoints, torch.ones_like(target_keypoints[..., :1])], dim=-1)
        invKtrans = torch.inverse(cameras["K"]).transpose(-1, -2)
        target_points = torch.matmul(target_homo, invKtrans)[..., :2]
        pred_2d = project_points(pred_keypoints, cameras)
        repro_loss = weighted_l2_loss(pred_2d, target_points, target_conf)
        linear_loss = smooth_sequence_loss(params["Th"], window_weight=window_weight, order=order)
        depth_loss = depth_smooth_loss(params["Th"], cameras, window_weight=window_weight, order=order)
        total_loss = (
            float(repro_cfg["weight"]) * repro_loss
            + float(smooth_cfg["weight"]) * (
                float(smooth_cfg["linear_weight"]) * linear_loss
                + float(smooth_cfg["depth_weight"]) * depth_loss
            )
        )
        return total_loss, {
            "reprojection": float(repro_loss.detach().cpu()),
            "smooth_linear": float(linear_loss.detach().cpu()),
            "smooth_depth": float(depth_loss.detach().cpu()),
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
        "optimized_keys": ["Th", "Rh"],
    }


def optimize_phase5(params, cameras, keypoints, model, phase5_cfg):
    chunk_size = int(phase5_cfg.get("chunk_size", 0) or 0)
    chunk_stride = int(phase5_cfg.get("chunk_stride", chunk_size or 0) or 0)
    n_frames = int(keypoints.shape[0])

    if chunk_size <= 0 or chunk_size >= n_frames:
        return _optimize_phase5_window(params, cameras, keypoints, model, phase5_cfg)

    if chunk_stride <= 0:
        chunk_stride = chunk_size

    chunk_summaries = []
    for chunk_start in range(0, n_frames, chunk_stride):
        chunk_end = min(n_frames, chunk_start + chunk_size)
        if chunk_end <= chunk_start:
            break

        params_chunk = {
            key: _slice_tensor_frame_range(value, chunk_start, chunk_end).detach().clone()
            for key, value in params.items()
        }
        cameras_chunk = {
            key: _slice_tensor_frame_range(value, chunk_start, chunk_end).detach().clone()
            for key, value in cameras.items()
        }
        keypoints_chunk = keypoints[chunk_start:chunk_end].detach().clone()

        summary = _optimize_phase5_window(params_chunk, cameras_chunk, keypoints_chunk, model, phase5_cfg)

        with torch.no_grad():
            params["Th"][chunk_start:chunk_end].copy_(params_chunk["Th"].detach())
            params["Rh"][chunk_start:chunk_end].copy_(params_chunk["Rh"].detach())

        chunk_summaries.append(
            {
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                **summary,
            }
        )

    return {
        "mode": "chunked",
        "chunk_size": chunk_size,
        "chunk_stride": chunk_stride,
        "chunk_count": len(chunk_summaries),
        "chunk_summaries": chunk_summaries,
        "optimized_keys": ["Th", "Rh"],
    }


def optimize_phase6(params, cameras, keypoints, model, phase6_cfg):
    optimizer_cfg = phase6_cfg["optimizer"]
    repro_cfg = phase6_cfg["reprojection"]
    smooth_cfg = phase6_cfg["smooth"]
    repeat = max(1, int(phase6_cfg.get("repeat", 1)))

    window_weight = smooth_cfg["window_weight"]
    order = int(smooth_cfg["order"])
    rh_weight = float(smooth_cfg.get("rh_weight", 1.0))
    repeat_summaries = []

    for repeat_idx in range(repeat):
        opt_tensors = [params["Rh"], params["Th"]]
        for tensor in opt_tensors:
            tensor.requires_grad_(True)

        if str(optimizer_cfg["optim_type"]).lower() != "lbfgs":
            raise NotImplementedError("Only lbfgs optimizer is supported by this standalone pipeline")

        line_search_fn = optimizer_cfg.get("line_search_fn", "strong_wolfe")
        if not line_search_fn:
            line_search_fn = None

        optimizer = torch.optim.LBFGS(
            opt_tensors,
            lr=float(optimizer_cfg["lr"]),
            max_iter=int(optimizer_cfg["max_iter"]),
            line_search_fn=line_search_fn,
            tolerance_grad=float(optimizer_cfg["tolerance_grad"]),
            tolerance_change=float(optimizer_cfg["tolerance_change"]),
        )

        max_outer_steps = int(optimizer_cfg["max_outer_steps"])
        relative_tolerance = float(optimizer_cfg["relative_tolerance"])
        last_loss = None

        def compute_components():
            outputs = model(params)
            pred_keypoints = outputs["keypoints"]
            target_keypoints = keypoints[:, :, :2]
            target_conf = keypoints[:, :, 2]
            target_homo = torch.cat([target_keypoints, torch.ones_like(target_keypoints[..., :1])], dim=-1)
            invKtrans = torch.inverse(cameras["K"]).transpose(-1, -2)
            target_points = torch.matmul(target_homo, invKtrans)[..., :2]
            pred_2d = project_points(pred_keypoints, cameras)
            repro_loss = gmof_loss(pred_2d, target_points, target_conf, rho=repro_cfg["gm_rho"])
            th_linear = smooth_sequence_loss(params["Th"], window_weight=window_weight, order=order)
            th_depth = depth_smooth_loss(params["Th"], cameras, window_weight=window_weight, order=order)
            rh_smooth = smooth_sequence_loss(params["Rh"], window_weight=window_weight, order=order)

            total_loss = (
                float(repro_cfg["weight"]) * repro_loss
                + float(smooth_cfg["weight"]) * (
                    float(smooth_cfg["th_linear_weight"]) * th_linear
                    + float(smooth_cfg["th_depth_weight"]) * th_depth
                    + rh_weight * rh_smooth
                )
            )
            return total_loss, {
                "reprojection": float(repro_loss.detach().cpu()),
                "smooth_th_linear": float(th_linear.detach().cpu()),
                "smooth_th_depth": float(th_depth.detach().cpu()),
                "smooth_rh": float(rh_smooth.detach().cpu()),
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
        repeat_summaries.append(
            {
                "repeat_index": repeat_idx,
                "total_loss": float(final_loss.detach().cpu()),
                "loss_components": components,
                "optimized_keys": ["Rh", "Th"],
            }
        )

    return repeat_summaries


def main():
    parser = argparse.ArgumentParser(description="Monolithic pipeline: phase 2 + 3 + 5 + 6")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("configs.yml")),
        help="Path to pipeline config YAML",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    phase_config = config.get(PHASE_KEY)
    if not phase_config:
        raise KeyError("Missing config section: {}".format(PHASE_KEY))

    base_dir = resolve_base_dir(args.config, config)
    inputs = phase_config["inputs"]
    runtime_cfg = phase_config["runtime"]
    output_cfg = phase_config["output"]
    model_cfg = phase_config["model"]
    phase2_cfg = phase_config["phase2"]
    phase3_cfg = phase_config["phase3"]
    phase5_cfg = phase_config["phase5"]
    phase6_cfg = phase_config["phase6"]

    params_path = resolve_path(base_dir, inputs["params_path"])
    keypoints_path = resolve_path(base_dir, inputs["keypoints_path"])
    cameras_path = resolve_path(base_dir, inputs["cameras_path"])
    smpl_model_path = resolve_path(base_dir, inputs["smpl_model_path"])
    j_regressor_path = resolve_path(base_dir, inputs["j_regressor_path"])
    output_dir = resolve_path(base_dir, output_cfg["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_path = output_dir / "phase2_params.pkl"
    stage3_path = output_dir / "phase3_params.pkl"
    stage5_path = output_dir / "phase5_params.pkl"
    final_params_path = output_dir / "pipeline_params.pkl"
    summary_path = output_dir / "pipeline_summary.json"

    device_name = runtime_cfg["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    params_blob = load_data(params_path)
    params = extract_params(params_blob)
    keypoints = extract_keypoints(load_data(keypoints_path))
    cameras = extract_cameras(load_data(cameras_path), n_frames=keypoints.shape[0])

    for required_key in ["Rh", "Th", "poses", "shapes"]:
        if required_key not in params:
            raise KeyError("Missing '{}' in params input".format(required_key))

    model = SmplBody25Model(
        smpl_model_path=smpl_model_path,
        j_regressor_path=j_regressor_path,
        device=device,
        model_cfg=model_cfg,
    )

    params2 = run_phase2_init_translation(
        params=params,
        keypoints=keypoints,
        cameras=cameras,
        model=model,
        phase2_cfg=phase2_cfg,
    )
    save_stage(stage2_path, params2)

    params3 = params2.copy()
    params3["poses"] = smooth_poses(params3["poses"], window_size=int(phase3_cfg["window_size"]))
    save_stage(stage3_path, params3)

    params_t, cameras_t, keypoints_t = to_tensor_dict(params3, cameras, keypoints, device=device)
    phase5_summary = optimize_phase5(
        params=params_t,
        cameras=cameras_t,
        keypoints=keypoints_t,
        model=model,
        phase5_cfg=phase5_cfg,
    )
    save_stage(stage5_path, {key: value.detach().cpu().numpy().astype(np.float32) for key, value in params_t.items()})

    phase6_summaries = optimize_phase6(
        params=params_t,
        cameras=cameras_t,
        keypoints=keypoints_t,
        model=model,
        phase6_cfg=phase6_cfg,
    )

    params_final = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in params_t.items()}
    save_stage(final_params_path, params_final)
    save_json(
        summary_path,
        {
            "phase": PHASE_KEY,
            "inputs": {
                "params_path": str(params_path),
                "keypoints_path": str(keypoints_path),
                "cameras_path": str(cameras_path),
                "smpl_model_path": str(smpl_model_path),
                "j_regressor_path": str(j_regressor_path),
            },
            "stages": {
                "phase2": {"output": str(stage2_path)},
                "phase3": {"output": str(stage3_path)},
                "phase5": {"output": str(stage5_path), "summary": phase5_summary},
                "phase6": {"output": str(final_params_path), "summaries": phase6_summaries},
            },
            "final_output": str(final_params_path),
        },
    )
    print("[{}] wrote {}".format(PHASE_KEY, final_params_path))


if __name__ == "__main__":
    main()
