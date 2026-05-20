from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import ConvexHull

W, H = 2048, 2048
fx = fy = (W * W + H * H) ** 0.5
cx, cy = W / 2.0, H / 2.0
N_NEIGHBORS = 10
TORSO_PART_IDS = {0, 3, 6, 9, 13, 14}
OCCLUSION_CHECK_JOINTS = {
    "left_elbow", "left_hand", "right_elbow", "right_hand",
    "left_knee", "left_ankle", "right_knee", "right_ankle",
}
_TORSO_MASK = None


def load_torso_mask(seg_path):
    global _TORSO_MASK
    if not seg_path:
        print("[Fusion] WARNING: segmentation path not configured. Occlusion disabled.")
        _TORSO_MASK = None
        return None
    seg_path = Path(seg_path)
    if not seg_path.exists():
        print("[Fusion] WARNING: {} not found. Occlusion disabled.".format(seg_path))
        _TORSO_MASK = None
        return None
    with seg_path.open("rb") as f:
        seg = pickle.load(f, encoding="latin1")
    _TORSO_MASK = np.isin(seg["smpl_index"], list(TORSO_PART_IDS))
    print("[Fusion] Torso mask loaded: {} / 6890 vertices".format(int(_TORSO_MASK.sum())))
    return _TORSO_MASK


def _as_xyz(point):
    arr = np.asarray(point, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Expected shape (3,), got {}".format(arr.shape))
    return arr


def get_orientation_flag(joints, epsilon=1e-2):
    required = ("right_shoulder", "left_shoulder", "right_hip", "left_hip")
    if not all(name in joints for name in required):
        return {name: 0 for name in joints}
    rs = _as_xyz(joints["right_shoulder"])
    ls = _as_xyz(joints["left_shoulder"])
    rh = _as_xyz(joints["right_hip"])
    lh = _as_xyz(joints["left_hip"])
    mid_shoulders = (rs + ls) / 2.0
    mid_hips = (rh + lh) / 2.0
    v_lr = rs - ls
    v_spine = mid_shoulders - mid_hips
    forward_vec = np.cross(v_lr, v_spine)
    norm = float(np.linalg.norm(forward_vec))
    if norm < 1e-8:
        return {name: 0 for name in joints}
    forward_vec /= norm
    torso_center = (mid_shoulders + mid_hips) / 2.0
    flags = {}
    for name, pos in joints.items():
        dot = float(np.dot(_as_xyz(pos) - torso_center, forward_vec))
        flags[name] = 0 if abs(dot) < epsilon else (1 if dot > 0 else -1)
    return flags


def _project_to_image(points_3d):
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    valid = Z > 1e-6
    u = np.where(valid, fx * (X / np.where(valid, Z, 1.0)) + cx, np.nan)
    v = np.where(valid, fy * (Y / np.where(valid, Z, 1.0)) + cy, np.nan)
    return np.stack([u, v], axis=1)


def _extract_torso_data(verts, torso_mask):
    V_torso = verts[torso_mask]
    uv = _project_to_image(V_torso)
    valid = ~np.isnan(uv).any(axis=1)
    all_torso_uv = uv[valid]
    all_torso_z = V_torso[valid, 2]
    if len(all_torso_uv) < 3:
        return None, None, None
    try:
        hull_2d_obj = ConvexHull(all_torso_uv)
        return hull_2d_obj, all_torso_uv, all_torso_z
    except Exception:
        return None, None, None


def _point_in_hull2d(pt, hull):
    p = np.array([pt[0], pt[1], 1.0])
    return bool(np.all(hull.equations @ p <= 1e-10))


def _local_torso_z(u_k, v_k, all_torso_uv, all_torso_z):
    dists_2d = np.sqrt((all_torso_uv[:, 0] - u_k) ** 2 + (all_torso_uv[:, 1] - v_k) ** 2)
    k = min(N_NEIGHBORS, len(dists_2d))
    nn_idx = np.argpartition(dists_2d, k - 1)[:k]
    return float(all_torso_z[nn_idx].min())


def compute_visibility_from_mesh_vertices(joints, verts, occlusion_tau=0.05):
    visibility = {name: True for name in joints.keys()}
    verts = np.asarray(verts, dtype=float)
    if _TORSO_MASK is None or len(verts) != 6890:
        return visibility
    hull_2d_obj, all_torso_uv, all_torso_z = _extract_torso_data(verts, _TORSO_MASK)
    if hull_2d_obj is None:
        return visibility
    for name, pos in joints.items():
        if name not in OCCLUSION_CHECK_JOINTS:
            continue
        kp_3d = _as_xyz(pos)
        if kp_3d[2] <= 0:
            continue
        u_k = fx * (kp_3d[0] / kp_3d[2]) + cx
        v_k = fy * (kp_3d[1] / kp_3d[2]) + cy
        if not _point_in_hull2d(np.array([u_k, v_k]), hull_2d_obj):
            continue
        z_local = _local_torso_z(u_k, v_k, all_torso_uv, all_torso_z)
        visibility[name] = not (kp_3d[2] > (z_local + occlusion_tau))
    return visibility
