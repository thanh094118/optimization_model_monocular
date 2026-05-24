from collections import Counter

import joblib
import numpy as np
import smplx
import torch
from scipy.spatial.distance import cdist
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
    for key in ("pose", "betas"):
        if key not in person_data:
            raise KeyError(f"Missing key '{key}' in {file_path}")
    return person_data


def get_canonical_joints_3d(model, person_data):
    pose = np.asarray(person_data["pose"], dtype=np.float32)
    betas = np.asarray(person_data["betas"], dtype=np.float32)
    t = pose.shape[0]
    if betas.ndim == 1:
        betas = np.tile(betas, (t, 1))
    pose_t = torch.as_tensor(np.ascontiguousarray(pose), dtype=torch.float32)
    betas_t = torch.as_tensor(np.ascontiguousarray(betas), dtype=torch.float32)
    global_orient = torch.zeros((t, 3), dtype=torch.float32)
    transl = torch.zeros((t, 3), dtype=torch.float32)
    with torch.no_grad():
        output = model(
            betas=betas_t,
            global_orient=global_orient,
            body_pose=pose_t[:, 3:],
            transl=transl,
        )
    joints_all = output.joints.cpu().numpy()
    result = np.empty((t, len(JOINT_NAMES), 3), dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES):
        result[:, i, :] = joints_all[:, SMPL_JOINT_MAP[name], :]
    return result


def build_canonical_sequence(joints_3d):
    return joints_3d.reshape(joints_3d.shape[0], -1)


def dtw_mode_offset(seq1, seq2):
    t1 = seq1.shape[0]
    t2 = seq2.shape[0]
    cost = cdist(seq1, seq2, metric="euclidean")
    dp = np.full((t1 + 1, t2 + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, t1 + 1):
        row = dp[i]
        row_prev = dp[i - 1]
        for j in range(1, t2 + 1):
            row[j] = cost[i - 1, j - 1] + min(row_prev[j - 1], row_prev[j], row[j - 1])

    i, j = t1, t2
    offsets = []
    while i > 0 and j > 0:
        offsets.append((j - 1) - (i - 1))
        a = dp[i - 1, j - 1]
        b = dp[i - 1, j]
        c = dp[i, j - 1]
        if a <= b and a <= c:
            i -= 1
            j -= 1
        elif b <= c:
            i -= 1
        else:
            j -= 1
    return Counter(offsets).most_common(1)[0][0]


def compute_offset_from_pkls(pkl1_path, pkl2_path, smpl_model_path, verbose=False):
    cam1_data = load_pkl_data(str(pkl1_path))
    cam2_data = load_pkl_data(str(pkl2_path))
    t = min(len(cam1_data["pose"]), len(cam2_data["pose"]))
    model = smplx.create(
        smpl_model_path,
        model_type="smpl",
        batch_size=max(1, t),
        use_pca=False,
        flat_hand_mean=True,
        gender="neutral",
    ).eval()
    seq1 = build_canonical_sequence(get_canonical_joints_3d(model, cam1_data))
    seq2 = build_canonical_sequence(get_canonical_joints_3d(model, cam2_data))
    return int(dtw_mode_offset(seq1, seq2))
