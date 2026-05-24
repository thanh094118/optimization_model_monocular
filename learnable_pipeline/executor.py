# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import copy
import json
import re

import joblib
import numpy as np
import torch
import smplx

from pose_pipeline.smpl_runner import SMPL_JOINT_MAP
from learnable_pipeline.config import OUTPUT_SUBDIRS
from learnable_pipeline.logs import log_disabled, log_done, log_header
from json_io import write_json

NOTEBOOK_JOINT_NAMES = list(SMPL_JOINT_MAP.keys())
NOTEBOOK_JOINT_INDICES = [SMPL_JOINT_MAP[name] for name in NOTEBOOK_JOINT_NAMES]


def _frame_index(path: Path) -> int:
    match = re.search(r"\d+", path.name)
    if not match:
        raise ValueError(f"Cannot extract frame index from {path.name}")
    return int(match.group())


def _clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob("learnable_frame_*.json"):
        old_json.unlink()
    meta = output_dir / "metadata.json"
    if meta.exists():
        meta.unlink()
    for subdir in OUTPUT_SUBDIRS:
        target_dir = output_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for old_json in target_dir.glob("learnable_frame_*.json"):
            old_json.unlink()


def load_fused_results(input_dir: Path) -> list[dict]:
    keypoints_dir = input_dir / "keypoints3d"
    metadata_dir = input_dir / "metadata"
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Fused keypoints directory not found: {keypoints_dir}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Fused metadata directory not found: {metadata_dir}")
    file_paths = sorted(keypoints_dir.glob("fused_data_*.json"), key=_frame_index)
    results = []
    for path in file_paths:
        with path.open("r", encoding="utf-8") as f:
            frame_data = json.load(f)
        metadata_path = metadata_dir / path.name
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                frame_data.update(json.load(f))
        results.append(frame_data)
    print(f"[Learnable] Loaded {len(results)} fused frames from {input_dir}")
    return results


def _as_np_xyz(point) -> np.ndarray:
    arr = np.asarray(point, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape}")
    return arr


def _get_pose_from_frame(frame_data: dict, camera_name: str):
    if "optimized" in frame_data and camera_name in frame_data["optimized"]:
        return frame_data["optimized"][camera_name]
    if camera_name in frame_data:
        return frame_data[camera_name]
    return None


def _get_joint_confidence(frame_data: dict, camera_name: str) -> dict:
    return frame_data.get("joint_confidence", {}).get(camera_name, {})


def extract_target_arrays(results_list: list[dict], camera_name: str):
    targets = []
    weights = []
    for frame_data in results_list:
        pose = _get_pose_from_frame(frame_data, camera_name)
        if pose is None:
            raise KeyError(f"Frame missing {camera_name} pose")
        conf = _get_joint_confidence(frame_data, camera_name)
        frame_targets = []
        frame_weights = []
        for name in NOTEBOOK_JOINT_NAMES:
            if name not in pose:
                raise KeyError(f"Missing joint {name} in {camera_name}")
            frame_targets.append(_as_np_xyz(pose[name]))
            frame_weights.append(float(conf.get(name, 1.0)))
        targets.append(frame_targets)
        weights.append(frame_weights)
    return np.asarray(targets, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def _extract_person_payload(wham_data):
    if isinstance(wham_data, dict):
        if 0 in wham_data:
            return wham_data[0]
        if "0" in wham_data:
            return wham_data["0"]
        for value in wham_data.values():
            if isinstance(value, dict):
                return value
    if isinstance(wham_data, list):
        for value in wham_data:
            if isinstance(value, dict):
                return value
    return None


def _truncate_person(person: dict, frame_count: int) -> dict:
    out = {}
    for key, value in person.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] > frame_count:
            out[key] = value[:frame_count].copy()
        else:
            out[key] = value
    return out


def load_wham_data(cam1_pkl: Path, cam2_pkl: Path, frame_count: int) -> dict:
    if not cam1_pkl.exists() or not cam2_pkl.exists():
        raise FileNotFoundError(f"WHAM files not found: {cam1_pkl} or {cam2_pkl}")

    print(f"[Learnable] Loading WHAM camera 1: {cam1_pkl}")
    person_1 = _extract_person_payload(joblib.load(cam1_pkl))
    print(f"[Learnable] Loading WHAM camera 2: {cam2_pkl}")
    person_2 = _extract_person_payload(joblib.load(cam2_pkl))

    if person_1 is None or person_2 is None:
        raise ValueError("Cannot extract person payload from WHAM PKL files")
    for camera_name, person in (("camera1", person_1), ("camera2", person_2)):
        for key in ("pose", "trans"):
            if key not in person:
                raise KeyError(f"Missing key {key!r} in {camera_name} WHAM data")

    return {
        "camera1": _truncate_person(person_1, frame_count),
        "camera2": _truncate_person(person_2, frame_count),
    }


def create_smpl_model(smpl_model_path: Path, batch_size: int, device: torch.device):
    return smplx.create(
        str(smpl_model_path),
        model_type="smpl",
        batch_size=batch_size,
        use_pca=False,
        flat_hand_mean=True,
        gender="neutral",
    ).to(device).eval()


def betas_per_frame(person_data: dict, frame_count: int) -> np.ndarray:
    betas = np.asarray(person_data.get("betas", np.zeros((frame_count, 10))), dtype=np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]
    if betas.shape[0] == 1:
        betas = np.repeat(betas, frame_count, axis=0)
    elif betas.shape[0] < frame_count:
        betas = np.repeat(betas[:1], frame_count, axis=0)
    else:
        betas = betas[:frame_count]
    return betas[:, :10]


def forward_smpl_joints_torch(model, pose_t: torch.Tensor, trans_t: torch.Tensor, betas_t: torch.Tensor) -> torch.Tensor:
    output = model(
        betas=betas_t,
        global_orient=pose_t[:, :3],
        body_pose=pose_t[:, 3:72],
        transl=trans_t[:, :3],
    )
    return output.joints[:, NOTEBOOK_JOINT_INDICES, :]


def forward_smpl_joints_numpy(model, pose_np: np.ndarray, trans_np: np.ndarray, betas_np: np.ndarray, device: torch.device) -> np.ndarray:
    pose_t = torch.as_tensor(pose_np[:, :72], dtype=torch.float32, device=device)
    trans_t = torch.as_tensor(trans_np[:, :3], dtype=torch.float32, device=device)
    betas_t = torch.as_tensor(betas_np[:, :10], dtype=torch.float32, device=device)
    with torch.no_grad():
        joints = forward_smpl_joints_torch(model, pose_t, trans_t, betas_t)
    return joints.detach().cpu().numpy()


def fit_smpl_to_target(raw_person: dict, target_joints: np.ndarray, joint_weights: np.ndarray, model, cfg: dict, device: torch.device):
    frame_count = min(len(raw_person["pose"]), target_joints.shape[0])
    raw_pose_np = np.asarray(raw_person["pose"][:frame_count, :72], dtype=np.float32)
    raw_trans_np = np.asarray(raw_person["trans"][:frame_count, :3], dtype=np.float32)
    betas_np = betas_per_frame(raw_person, frame_count)

    target = torch.as_tensor(target_joints[:frame_count], dtype=torch.float32, device=device)
    weights = torch.as_tensor(joint_weights[:frame_count], dtype=torch.float32, device=device)
    raw_pose = torch.as_tensor(raw_pose_np, dtype=torch.float32, device=device)
    raw_trans = torch.as_tensor(raw_trans_np, dtype=torch.float32, device=device)
    betas = torch.as_tensor(betas_np, dtype=torch.float32, device=device)

    pose_param = raw_pose.clone().detach().requires_grad_(True)
    trans_param = raw_trans.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [pose_param, trans_param],
        lr=float(cfg.get("lr", 0.03)),
    )

    num_iters = int(cfg.get("num_iters", 80))
    pose_prior_weight = float(cfg.get("pose_prior_weight", 0.05))
    trans_prior_weight = float(cfg.get("trans_prior_weight", 0.05))

    last_loss = None
    for _ in range(num_iters):
        optimizer.zero_grad()
        pred = forward_smpl_joints_torch(model, pose_param, trans_param, betas)
        dist = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1) + 1e-12)
        data_loss = (dist * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
        pose_prior = torch.mean((pose_param - raw_pose) ** 2)
        trans_prior = torch.mean((trans_param - raw_trans) ** 2)
        total_loss = data_loss + pose_prior_weight * pose_prior + trans_prior_weight * trans_prior
        total_loss.backward()
        optimizer.step()
        last_loss = float(total_loss.detach().cpu())

    return {
        "pose": pose_param.detach().cpu().numpy(),
        "trans": trans_param.detach().cpu().numpy(),
        "betas": betas_np,
        "last_loss": last_loss,
    }


def pose_to_joint_dicts(pose_np: np.ndarray, trans_np: np.ndarray, betas_np: np.ndarray, model, device: torch.device) -> list[dict]:
    joints = forward_smpl_joints_numpy(model, pose_np, trans_np, betas_np, device)
    frames = []
    for frame in joints:
        frames.append({
            name: [round(float(x), 5) for x in frame[idx]]
            for idx, name in enumerate(NOTEBOOK_JOINT_NAMES)
        })
    return frames


def run_learnable_smplify(config: dict) -> None:
    paths = config["paths"]
    runtime_cfg = config.get("runtime", {})
    learnable_cfg = config.get("learnable_smplify") or config.get("learnable", {})

    if not learnable_cfg.get("enabled", True):
        log_disabled()
        return

    input_dir = Path(paths["fused_output_dir"])
    output_dir = Path(paths["learnable_output_dir"])
    smpl_model_path = Path(paths["smpl_model"])
    cam1_pkl = Path(paths["cam1_pkl"])
    cam2_pkl = Path(paths["cam2_pkl"])

    if not input_dir.exists():
        raise FileNotFoundError(f"Fused JSON directory not found: {input_dir}")
    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL model not found: {smpl_model_path}")

    if runtime_cfg.get("clean_output", True):
        _clean_output(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    device_name = learnable_cfg.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[Learnable] CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    log_header()

    judgement_results = load_fused_results(input_dir)
    frame_count = len(judgement_results)
    if frame_count == 0:
        raise ValueError(f"No fused_data_*.json files found in {input_dir / 'keypoints3d'}")

    wham_data = load_wham_data(cam1_pkl, cam2_pkl, frame_count)
    smpl_model = create_smpl_model(smpl_model_path, frame_count, device)

    output = copy.deepcopy(judgement_results)
    camera_losses = {}

    for camera_name in ("camera1", "camera2"):
        print(f"[Learnable] Processing {camera_name}")
        target_joints, joint_weights = extract_target_arrays(judgement_results, camera_name)
        raw_person = wham_data[camera_name]
        fitted = fit_smpl_to_target(raw_person, target_joints, joint_weights, smpl_model, learnable_cfg, device)
        final_frames = pose_to_joint_dicts(fitted["pose"], fitted["trans"], fitted["betas"], smpl_model, device)
        camera_losses[camera_name] = fitted.get("last_loss")

        for frame_idx, frame_joints in enumerate(final_frames):
            output[frame_idx].setdefault("final", {})[camera_name] = frame_joints
            output[frame_idx].setdefault("learnable_smplify", {})[camera_name] = frame_joints

        print(f"[Learnable] Done {camera_name}. last_loss={camera_losses[camera_name]}")

    metadata = {
        "input_dir": str(input_dir),
        "cam1_pkl": str(cam1_pkl),
        "cam2_pkl": str(cam2_pkl),
        "smpl_model": str(smpl_model_path),
        "frame_count": frame_count,
        "device": str(device),
        "camera_last_loss": camera_losses,
        "timestamp": datetime.now().isoformat(),
    }
    write_json(output_dir / "metadata.json", metadata)

    for i, frame_data in enumerate(output):
        out_name = f"learnable_frame_{i + 1}.json"
        keypoints_data = {
            "camera1": frame_data.get("final", {}).get("camera1", {}),
            "camera2": frame_data.get("final", {}).get("camera2", {}),
        }
        metadata_data = {k: v for k, v in frame_data.items() if k not in ("camera1", "camera2", "final", "learnable_smplify")}
        write_json(output_dir / "keypoints3d" / out_name, keypoints_data)
        write_json(output_dir / "metadata" / out_name, metadata_data)
        if (i + 1) % 50 == 0 or (i + 1) == frame_count:
            print(f"[Learnable] Saved {i + 1}/{frame_count} files")

    log_done(str(output_dir))
