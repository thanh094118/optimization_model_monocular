from collections import Counter

import joblib
import numpy as np
import smplx
import torch
from keypoints_map import get_smpl_joint_map

SMPL_JOINT_MAP = get_smpl_joint_map()
JOINT_NAMES = tuple(SMPL_JOINT_MAP.keys())


def _extract_person_payload(data):
    if isinstance(data, dict) or "defaultdict" in str(type(data)):
        if 0 in data:
            return data[0]
        if "0" in data:
            return data["0"]
        for value in data.values():
            if isinstance(value, dict):
                return value
    if isinstance(data, list):
        for value in data:
            if isinstance(value, dict):
                return value
    return None


def load_pkl_data(file_path):
    person_data = _extract_person_payload(joblib.load(file_path))
    if person_data is None:
        raise ValueError(f"Cannot parse payload from {file_path}")
    for key in ("pose", "trans", "betas"):
        if key not in person_data:
            raise KeyError(f"Missing key '{key}' in {file_path}")
    return person_data


def get_all_joints_from_smpl(model, person_data):
    pose = np.asarray(person_data["pose"], dtype=np.float32)
    trans = np.asarray(person_data["trans"], dtype=np.float32)
    betas = np.asarray(person_data["betas"], dtype=np.float32)
    t = pose.shape[0]
    if betas.ndim == 1:
        betas = np.tile(betas, (t, 1))
    pose_t = torch.as_tensor(np.ascontiguousarray(pose), dtype=torch.float32)
    trans_t = torch.as_tensor(np.ascontiguousarray(trans), dtype=torch.float32)
    betas_t = torch.as_tensor(np.ascontiguousarray(betas), dtype=torch.float32)
    with torch.no_grad():
        output = model(
            betas=betas_t,
            global_orient=pose_t[:, :3],
            body_pose=pose_t[:, 3:],
            transl=trans_t,
        )
    joints_all = output.joints.cpu().numpy()
    result = np.empty((t, len(JOINT_NAMES), 3), dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES):
        result[:, i, :] = joints_all[:, SMPL_JOINT_MAP[name], :]
    pelvis = (result[:, 7, :] + result[:, 11, :]) * 0.5
    return np.concatenate([pelvis[:, None, :], result], axis=1)


def normalize_pa_mpjpe(seq_with_pelvis):
    pelvis = seq_with_pelvis[:, 0, :]
    centered = seq_with_pelvis[:, 1:, :] - pelvis[:, None, :]
    right_shoulder = centered[:, 1, :]
    left_shoulder = centered[:, 4, :]
    vec = right_shoulder - left_shoulder
    angle = np.arctan2(vec[:, 2], vec[:, 0])
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x = centered[:, :, 0]
    z = centered[:, :, 2]
    rotated = np.empty_like(centered)
    rotated[:, :, 0] = x * cos_a[:, None] + z * sin_a[:, None]
    rotated[:, :, 1] = centered[:, :, 1]
    rotated[:, :, 2] = -x * sin_a[:, None] + z * cos_a[:, None]
    zeros = np.zeros((rotated.shape[0], 1, 3), dtype=np.float32)
    return np.concatenate([zeros, rotated], axis=1)


def _procrustes_distance(frame1, frame2):
    mu1 = np.mean(frame1, axis=0)
    mu2 = np.mean(frame2, axis=0)
    a = frame1 - mu1
    b = frame2 - mu2
    ss = np.sum(b * b)
    scale = 1.0 if ss == 0 else np.sqrt(np.sum(a * a) / ss)
    h = (b * scale).T @ a
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    b_aligned = (b * scale) @ r + mu1
    diff = frame1 - b_aligned
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def find_time_offset_procrustes(seq1, seq2):
    n1 = len(seq1)
    offsets = []
    for i, frame2 in enumerate(seq2):
        best_j = 0
        best_d = float("inf")
        for j in range(n1):
            d = _procrustes_distance(seq1[j], frame2)
            if d < best_d:
                best_d = d
                best_j = j
        offsets.append(best_j - i)
    return Counter(offsets).most_common(1)[0][0]


def compute_offset_from_pkls(pkl1_path, pkl2_path, smpl_path, detail_print=False):
    cam1_data = load_pkl_data(str(pkl1_path))
    cam2_data = load_pkl_data(str(pkl2_path))
    t = min(len(cam1_data["pose"]), len(cam2_data["pose"]))
    model = smplx.create(
        smpl_path,
        model_type="smpl",
        batch_size=max(1, t),
        use_pca=False,
        flat_hand_mean=True,
        gender="neutral",
    ).eval()
    seq1 = normalize_pa_mpjpe(get_all_joints_from_smpl(model, cam1_data))
    seq2 = normalize_pa_mpjpe(get_all_joints_from_smpl(model, cam2_data))
    return int(find_time_offset_procrustes(seq1, seq2))
