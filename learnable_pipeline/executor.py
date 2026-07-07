from __future__ import annotations
from pathlib import Path
from datetime import datetime
import copy
import importlib
import json
import re
import shutil
import sys
from typing import Any
import joblib
import numpy as np
import torch
import yaml

try:
    from easydict import EasyDict as edict
except Exception:
    class edict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc
        def __setattr__(self, name, value):
            self[name] = value

from keypoints_map import load_keypoints3d_map
from learnable_pipeline.post_opt import post_optimize_smpl_sequence
from json_io import write_json
from config_loader import resolve_inputs

LEARNABLE_FILE_PREFIX = "learnable_frame_"
OUTPUT_SUBDIRS = ("keypoints3d", "metadata")

BODY25_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
    "left_elbow", "left_wrist", "mid_hip", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear",
    "left_ear", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
    "right_small_toe", "right_heel",
)
BODY25_INDEX = {name: idx for idx, name in enumerate(BODY25_NAMES)}

def _get_custom_to_body25(map_data):
    custom_to_body25 = {}
    for kp in map_data["keypoints"]:
        # "Map 19 key tương thích vào BODY25... Không gán palm-center vào wrist."
        if kp["name"] in ("left_hand", "right_hand"):
            continue
        body25_name = kp.get("body25_alias")
        if body25_name:
            custom_to_body25[kp["name"]] = body25_name
    return custom_to_body25

def _frame_index(path: Path) -> int:
    match = re.search(r"\d+", path.name)
    if not match:
        raise ValueError(f"Cannot extract frame index from {path.name}")
    return int(match.group())

def _clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = output_dir / "metadata.json"
    if meta.exists():
        meta.unlink()
    for old_json in output_dir.glob("learnable_frame_*.json"):
        old_json.unlink()
    for subdir in OUTPUT_SUBDIRS:
        target_dir = output_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for old_json in target_dir.glob("learnable_frame_*.json"):
            old_json.unlink()

def load_fused_results(input_dir: Path) -> list[dict]:
    keypoints_dir = input_dir / "keypoints3d"
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Fused keypoints directory not found: {keypoints_dir}")
    file_paths = sorted(keypoints_dir.glob("fused_data_*.json"), key=_frame_index)
    results = []
    for path in file_paths:
        with path.open("r", encoding="utf-8") as f:
            frame_data = json.load(f)
        frame_data["frame_id"] = _frame_index(path)
        results.append(frame_data)
    print(f"[Learnable] Loaded {len(results)} fused frames from {input_dir}")
    return results

def _as_np_xyz(point: Any) -> np.ndarray:
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

def extract_fusion_joint_dicts(results_list: list[dict], camera_name: str, target_names: list[str]) -> list[dict[str, np.ndarray]]:
    frames: list[dict[str, np.ndarray]] = []
    for frame_idx, frame_data in enumerate(results_list):
        frame_id = int(frame_data.get("frame_id", frame_idx + 1))
        pose = _get_pose_from_frame(frame_data, camera_name)
        if pose is None:
            raise KeyError(f"Frame {frame_id} missing {camera_name} pose")
        frame_joints = {}
        for name in target_names:
            if name not in pose:
                raise KeyError(f"Frame {frame_id} missing joint {name!r} in {camera_name}")
            frame_joints[name] = _as_np_xyz(pose[name])
        frames.append(frame_joints)
    return frames

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
    person_1 = _extract_person_payload(joblib.load(cam1_pkl))
    person_2 = _extract_person_payload(joblib.load(cam2_pkl))
    if person_1 is None or person_2 is None:
        raise ValueError("Cannot extract person payload from WHAM PKL files")
    return {
        "camera1": _truncate_person(person_1, frame_count),
        "camera2": _truncate_person(person_2, frame_count),
    }

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

def _to_edict(value):
    if isinstance(value, dict):
        return edict({k: _to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_edict(v) for v in value]
    return value

def _import_repo_symbols(repo_src: Path):
    repo_src = repo_src.resolve()
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    basic_module = importlib.import_module("module.backbone.basic_modules")
    transforms_module = importlib.import_module("common.transforms")

    def _patched_unit_gcn_forward(self, x):
        N, C, T, V = x.size()
        A = self.A.to(x.device)
        if self.mask_learning:
            A = A * self.mask
        for i, a in enumerate(A):
            xa = x.view(-1, V).mm(a).view(N, C, T, V)
            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y = y + self.conv_list[i](xa)
        if self.use_local_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(N, self.out_channels * V, T)
            y = self.bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn(y)
        y = self.relu(y)
        return y

    def _patched_rot6d_to_axis_angle(x):
        import torch.nn.functional as F
        batch_size = x.shape[0]
        x = x.view(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        rot_mat = torch.stack((b1, b2, b3), dim=-1)
        rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1), device=x.device, dtype=torch.float32)], dim=2)
        axis_angle = transforms_module.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)
        axis_angle[torch.isnan(axis_angle)] = 0.0
        return axis_angle

    basic_module.Unit_GCN.forward = _patched_unit_gcn_forward
    transforms_module.rot6d_to_axis_angle = _patched_rot6d_to_axis_angle

    net_module = importlib.import_module("module.net_body25")
    kp_module = importlib.import_module("common.keypoint_geo")
    return net_module.NetBody25, kp_module.normalize_kp

def _load_checkpoint_state(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if state is None and isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

def load_netbody25(learnable_cfg: dict, device: torch.device):
    repo_src = Path(learnable_cfg["repo_src"])
    config_path = Path(learnable_cfg["net_config"])
    checkpoint_path = Path(learnable_cfg["checkpoint"])
    smpl_family_dir = Path(learnable_cfg["smpl_family_dir"])

    regressor_path = smpl_family_dir / "smpl" / "J_regressor_body25.npy"
    if not regressor_path.exists():
        explicit_regressor = learnable_cfg.get("j_regressor_body25")
        if explicit_regressor:
            explicit_regressor_path = Path(explicit_regressor)
            regressor_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(explicit_regressor_path, regressor_path)

    smpl_dir = smpl_family_dir / "smpl"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    required_smpl = ("SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl")
    explicit_neutral = learnable_cfg.get("smpl_neutral_path")
    neutral_src = Path(explicit_neutral) if explicit_neutral else None

    for smpl_name in required_smpl:
        target_path = smpl_dir / smpl_name
        if target_path.exists():
            continue
        explicit_key = f"smpl_{smpl_name.split('_')[1].split('.')[0].lower()}_path"
        explicit_src = learnable_cfg.get(explicit_key)
        src_path = Path(explicit_src) if explicit_src else None
        if src_path is None:
            if neutral_src is not None and neutral_src.exists():
                src_path = neutral_src
        shutil.copy2(src_path, target_path)

    NetBody25, normalize_kp = _import_repo_symbols(repo_src)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = _to_edict(yaml.safe_load(f))
    cfg.model_params.human_model.smpl_dir = str(smpl_family_dir)

    net = NetBody25(cfg.model_params).to(device).eval()
    net.load_state_dict(_load_checkpoint_state(checkpoint_path), strict=True)
    for key in net.human_model.layer.keys():
        net.human_model.layer[key] = net.human_model.layer[key].to(device)

    return net, normalize_kp

def _make_target_body25(
    fusion_frames: list[dict[str, np.ndarray]],
    init_body25_world: np.ndarray,
    custom_to_body25: dict[str, str],
    compute_mid_hip: bool = True,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    target = init_body25_world.copy()
    used_custom: set[str] = set()
    used_body25: set[str] = set()
    for frame_idx, fusion in enumerate(fusion_frames):
        for custom_name, body25_name in custom_to_body25.items():
            if custom_name not in fusion:
                continue
            target[frame_idx, BODY25_INDEX[body25_name]] = fusion[custom_name]
            used_custom.add(custom_name)
            used_body25.add(body25_name)
        if compute_mid_hip and "left_hip" in fusion and "right_hip" in fusion:
            target[frame_idx, BODY25_INDEX["mid_hip"]] = 0.5 * (fusion["left_hip"] + fusion["right_hip"])
            used_body25.add("mid_hip")
    filled_body25 = [name for name in BODY25_NAMES if name not in used_body25]
    return target.astype(np.float32), {
        "used_custom_joint_names": sorted(used_custom),
        "used_body25_joint_names": sorted(used_body25, key=lambda n: BODY25_INDEX[n]),
        "filled_body25_joint_names": filled_body25,
    }

def _compute_init_body25_world(net, pose_t: torch.Tensor, trans_t: torch.Tensor, betas_t: torch.Tensor) -> torch.Tensor:
    root_orient, body_pose, _ = net.split_pose_from_smplh(pose_t)
    smpl_out = net.human_model.layer["neutral"](
        betas=betas_t[:, :10],
        global_orient=root_orient,
        body_pose=body_pose,
    )
    joints_local = torch.einsum("bvc,jv->bjc", smpl_out.vertices, net.openpose_regressor)
    return joints_local + trans_t[:, None, :3]

def _compose_pose_like_template(pose_template: torch.Tensor, pred_root_orient: torch.Tensor, pred_body_pose: torch.Tensor) -> torch.Tensor:
    B = pose_template.shape[0]
    out = pose_template.clone()
    root_flat = pred_root_orient.reshape(B, -1)
    body_flat = pred_body_pose.reshape(B, -1)
    out[:, : min(3, out.shape[1])] = root_flat[:, : min(3, out.shape[1])]
    if out.shape[1] > 3:
        n_body = min(body_flat.shape[1], out.shape[1] - 3)
        out[:, 3 : 3 + n_body] = body_flat[:, :n_body]
    return out

def _align_trans_to_target_midhip(pred_joints_local: torch.Tensor, target_body25_world: torch.Tensor) -> torch.Tensor:
    mid_idx = BODY25_INDEX["mid_hip"]
    return target_body25_world[:, mid_idx] - pred_joints_local[:, mid_idx]

def _compute_body25_world_np(net, pose_np: np.ndarray, trans_np: np.ndarray, betas_np: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    out = np.zeros((pose_np.shape[0], 25, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, pose_np.shape[0], batch_size):
            end = min(start + batch_size, pose_np.shape[0])
            pose_t = torch.as_tensor(pose_np[start:end], dtype=torch.float32, device=device)
            trans_t = torch.as_tensor(trans_np[start:end], dtype=torch.float32, device=device)
            betas_t = torch.as_tensor(betas_np[start:end], dtype=torch.float32, device=device)
            out[start:end] = _compute_init_body25_world(net, pose_t, trans_t, betas_t).detach().cpu().numpy()
    return out

def infer_learnable_temporal_from_fusion(
    net, normalize_kp, raw_person: dict, fusion_frames: list[dict[str, np.ndarray]],
    custom_to_body25: dict[str, str], device: torch.device, batch_size: int = 128,
    compute_mid_hip: bool = True, first_frame_mode: str = "align_to_target",
    fallback_to_copy_if_worse: bool = False,
) -> dict[str, np.ndarray | dict]:
    frame_count = min(len(raw_person["pose"]), len(fusion_frames))
    pose_init_np = np.asarray(raw_person["pose"][:frame_count, :72], dtype=np.float32)
    trans_init_np = np.asarray(raw_person["trans"][:frame_count, :3], dtype=np.float32)
    betas_np = betas_per_frame(raw_person, frame_count)

    wham_init_body25 = _compute_body25_world_np(net, pose_init_np, trans_init_np, betas_np, device, batch_size)
    target_body25_np, mapping_info = _make_target_body25(fusion_frames, wham_init_body25, custom_to_body25, compute_mid_hip=compute_mid_hip)

    pred_pose = pose_init_np.copy()
    pred_trans = trans_init_np.copy()
    pred_joints25_world = np.zeros((frame_count, 25, 3), dtype=np.float32)
    init_joints25_used = np.zeros((frame_count, 25, 3), dtype=np.float32)

    with torch.no_grad():
        pose0_t = torch.as_tensor(pose_init_np[0:1], dtype=torch.float32, device=device)
        trans0_t = torch.as_tensor(trans_init_np[0:1], dtype=torch.float32, device=device)
        betas0_t = torch.as_tensor(betas_np[0:1], dtype=torch.float32, device=device)
        target0_t = torch.as_tensor(target_body25_np[0:1], dtype=torch.float32, device=device)

        root0, body0, _ = net.split_pose_from_smplh(pose0_t)
        smpl0 = net.human_model.layer["neutral"](betas=betas0_t[:, :10], global_orient=root0, body_pose=body0)
        joints0_local = torch.einsum("bvc,jv->bjc", smpl0.vertices, net.openpose_regressor)
        if first_frame_mode == "align_to_target":
            trans0_pred_t = _align_trans_to_target_midhip(joints0_local, target0_t)
        else:
            trans0_pred_t = trans0_t

        pred_trans[0:1] = trans0_pred_t.detach().cpu().numpy()
        pred_joints25_world[0:1] = (joints0_local + trans0_pred_t[:, None, :]).detach().cpu().numpy()
        init_joints25_used[0:1] = wham_init_body25[0:1]

        iter_pose_t = pose0_t.clone()
        iter_pose_t[:, :3] = root0.reshape(1, -1)[:, :3]
        iter_pose_t[:, 3:66] = body0.reshape(1, -1)[:, :63]
        iter_trans_t = trans0_pred_t.clone()

        for frame_idx in range(1, frame_count):
            betas_t = torch.as_tensor(betas_np[frame_idx : frame_idx + 1], dtype=torch.float32, device=device)
            target_t = torch.as_tensor(target_body25_np[frame_idx : frame_idx + 1], dtype=torch.float32, device=device)

            start_root_orient, start_body_pose, _ = net.split_pose_from_smplh(iter_pose_t)
            start_body25_t = _compute_init_body25_world(net, iter_pose_t, iter_trans_t, betas_t)
            init_joints25_used[frame_idx : frame_idx + 1] = start_body25_t.detach().cpu().numpy()

            init_norm, R, T = normalize_kp(start_body25_t, None, net.kp_index, R=None, T=None)
            target_norm, _, _ = normalize_kp(target_t, None, net.kp_index, R=R, T=T)
            input_joints = torch.stack([init_norm, target_norm], dim=1).permute(0, 3, 1, 2)

            pred_smpl, pred_joints_local, _pred_rotmat, pred_body_pose, pred_root_orient = net.predict(
                input_joints, start_root_orient, start_body_pose, betas_t[:, :10]
            )
            trans_pred_t = _align_trans_to_target_midhip(pred_joints_local, target_t)
            pred_joints_world_t = pred_joints_local + trans_pred_t[:, None, :]

            if fallback_to_copy_if_worse:
                copy_error = torch.mean(torch.linalg.norm(start_body25_t - target_t, dim=-1))
                pred_error = torch.mean(torch.linalg.norm(pred_joints_world_t - target_t, dim=-1))
                if copy_error < pred_error:
                    pred_root_orient = start_root_orient
                    pred_body_pose = start_body_pose
                    trans_pred_t = iter_trans_t
                    pred_joints_world_t = start_body25_t

            pose_pred_t = _compose_pose_like_template(iter_pose_t, pred_root_orient, pred_body_pose)
            pred_pose[frame_idx : frame_idx + 1] = pose_pred_t.detach().cpu().numpy()
            pred_trans[frame_idx : frame_idx + 1] = trans_pred_t.detach().cpu().numpy()
            pred_joints25_world[frame_idx : frame_idx + 1] = pred_joints_world_t.detach().cpu().numpy()

            iter_pose_t = pose_pred_t.detach().clone()
            iter_trans_t = trans_pred_t.detach().clone()

    return {
        "pose_init": pose_init_np,
        "pose_pred": pred_pose,
        "trans_init": trans_init_np,
        "trans_pred": pred_trans,
        "betas": betas_np,
        "mapping_info": mapping_info,
    }

def _metadata_for_frame(result: dict, frame_idx: int) -> dict:
    return {
        "pose_init": np.asarray(result["pose_init"])[frame_idx].tolist(),
        "pose_pred": np.asarray(result["pose_pred"])[frame_idx].tolist(),
        "betas": np.asarray(result["betas"])[frame_idx].tolist(),
        "trans_pred": np.asarray(result["trans_pred"])[frame_idx].tolist(),
        "initial_loss": result.get("initial_loss", 0.0),
        "final_loss": result.get("final_loss", 0.0),
    }

def run_learnable_smplify(config: dict) -> None:
    paths = config["paths"]
    inputs = resolve_inputs(config)
    runtime_cfg = config.get("runtime", {})
    learnable_cfg = copy.deepcopy(config.get("learnable_smplify") or config.get("learnable", {}))

    if not learnable_cfg["enabled"]:
        print("[Learnable] Disabled by config: learnable_smplify.enabled=false")
        return

    input_dir = Path(paths["fused_output_dir"])
    output_dir = Path(paths["learnable_output_dir"])
    cam1_pkl = Path(inputs["cam1_pkl"])
    cam2_pkl = Path(inputs["cam2_pkl"])

    j_regressor_3d_path = paths["j_regressor_3d"]

    repo_root = Path(__file__).resolve().parent
    learnable_cfg.setdefault("repo_src", str(repo_root / "Learnable-SMPLify" / "src"))
    learnable_cfg.setdefault("net_config", str(repo_root / "Learnable-SMPLify" / "src" / "config" / "net.yaml"))
    learnable_cfg.setdefault("smpl_family_dir", "models")
    learnable_cfg.setdefault("j_regressor_body25", "models/J_regressor_body25.npy")
    learnable_cfg["smpl_neutral_path"] = paths["smpl_model"]
    if "checkpoint" not in learnable_cfg or not learnable_cfg["checkpoint"]:
        raise ValueError("Missing config parameter: learnable.checkpoint")

    if runtime_cfg["clean_output"]:
        _clean_output(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in OUTPUT_SUBDIRS:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    device_name = learnable_cfg.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    print("[Learnable] Learnable-SMPLify after judgement/fusion")
    judgement_results = load_fused_results(input_dir)
    frame_count = len(judgement_results)
    frame_ids = [int(frame.get("frame_id", idx + 1)) for idx, frame in enumerate(judgement_results)]

    wham_data = load_wham_data(cam1_pkl, cam2_pkl, frame_count)
    net, normalize_kp = load_netbody25(learnable_cfg, device)

    map_data = load_keypoints3d_map(config["paths"]["keypoints3d_map"])
    custom_to_body25 = _get_custom_to_body25(map_data)
    target_names = [kp["name"] for kp in map_data["keypoints"]]

    batch_size = int(learnable_cfg.get("batch_size", 128))

    camera_results: dict[str, dict] = {}
    keypoints_output = [{"camera1": {}, "camera2": {}} for _ in range(frame_count)]
    metadata_output = [copy.deepcopy(frame) for frame in judgement_results]

    for camera_name in ("camera1", "camera2"):
        fusion_frames = extract_fusion_joint_dicts(judgement_results, camera_name, target_names)
        
        # 1. Run NetBody25 to get init pose/trans
        result = infer_learnable_temporal_from_fusion(
            net=net,
            normalize_kp=normalize_kp,
            raw_person=wham_data[camera_name],
            fusion_frames=fusion_frames,
            custom_to_body25=custom_to_body25,
            device=device,
            batch_size=batch_size,
        )
        
        # 2. Post-optimize
        best_pose, best_trans, initial_loss, final_loss = post_optimize_smpl_sequence(
            pose_init=result["pose_pred"],
            trans_init=result["trans_pred"],
            betas_np=result["betas"],
            fusion_frames=fusion_frames,
            net=net,
            j_regressor_3d_path=j_regressor_3d_path,
            map_data=map_data,
            device=device,
        )
        
        result["pose_pred"] = best_pose
        result["trans_pred"] = best_trans
        result["initial_loss"] = initial_loss
        result["final_loss"] = final_loss
        camera_results[camera_name] = result

        # Produce 21 keypoints
        # compute them from best_pose, best_trans
        smpl_layer = net.human_model.layer["neutral"]
        with torch.no_grad():
            for frame_idx in range(frame_count):
                pose_t = torch.tensor(best_pose[frame_idx:frame_idx+1], dtype=torch.float32, device=device)
                trans_t = torch.tensor(best_trans[frame_idx:frame_idx+1], dtype=torch.float32, device=device)
                betas_t = torch.tensor(result["betas"][frame_idx:frame_idx+1], dtype=torch.float32, device=device)
                
                root_orient, body_pose, _ = net.split_pose_from_smplh(pose_t)
                smpl_out = smpl_layer(
                    betas=betas_t[:, :10],
                    global_orient=root_orient,
                    body_pose=body_pose
                )
                j_reg = np.load(j_regressor_3d_path)
                j_reg_t = torch.tensor(j_reg, dtype=torch.float32, device=device)
                joints_all = torch.einsum("jv, bvc -> bjc", j_reg_t, smpl_out.vertices)
                
                joints_world = (joints_all + trans_t.unsqueeze(1))[0].cpu().numpy()
                
                out_dict = {}
                for kp in map_data["keypoints"]:
                    coord = joints_world[kp["regressor_index"]]
                    out_dict[kp["name"]] = [float(c) for c in coord]
                
                keypoints_output[frame_idx][camera_name] = out_dict
                metadata_output[frame_idx].setdefault("learnable_smplify", {})[camera_name] = _metadata_for_frame(result, frame_idx)

    metadata = {
        "method": "learnable_smplify_netbody25_plus_adam",
        "frame_count": frame_count,
        "output_joint_format": "canonical_21",
        "timestamp": datetime.now().isoformat(),
    }
    write_json(output_dir / "metadata.json", metadata)

    for i in range(frame_count):
        out_name = f"{LEARNABLE_FILE_PREFIX}{frame_ids[i]}.json"
        write_json(output_dir / "keypoints3d" / out_name, keypoints_output[i])
        metadata_data = {
            k: v for k, v in metadata_output[i].items()
            if k not in ("camera1", "camera2", "optimized", "final")
        }
        write_json(output_dir / "metadata" / out_name, metadata_data)

    print(f"[Learnable] Done. Output: {output_dir}")
