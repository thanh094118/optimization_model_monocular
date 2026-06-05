from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import ConvexHull

from fusion_pipeline.config import (
    CONFIDENCE_DELTA_CAP,
    HARMONIC_ALPHA,
    HARMONIC_BETA,
    HARMONIC_EPSILON,
    NON_REPLACEABLE_ANCHORS,
    OCCLUSION_CHECK_JOINTS,
    OCCLUSION_IMAGE_HEIGHT,
    OCCLUSION_IMAGE_WIDTH,
    OCCLUSION_NEIGHBORS,
    ORIENTATION_EPSILON,
    RIGID_BONES_RATIO,
    ROTATION_PARENT_JOINTS,
    TORSO_PART_IDS,
)

_FX = _FY = (OCCLUSION_IMAGE_WIDTH * OCCLUSION_IMAGE_WIDTH + OCCLUSION_IMAGE_HEIGHT * OCCLUSION_IMAGE_HEIGHT) ** 0.5
_CX = OCCLUSION_IMAGE_WIDTH / 2.0
_CY = OCCLUSION_IMAGE_HEIGHT / 2.0
_TORSO_MASK = None


def as_xyz(point):
    arr = np.asarray(point, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Expected shape (3,), got {}".format(arr.shape))
    return arr


_as_xyz = as_xyz


def get_orientation_flag(joints, epsilon=ORIENTATION_EPSILON):
    required = ("right_shoulder", "left_shoulder", "right_hip", "left_hip")
    if not all(name in joints for name in required):
        return {name: 0 for name in joints}
    rs = as_xyz(joints["right_shoulder"])
    ls = as_xyz(joints["left_shoulder"])
    rh = as_xyz(joints["right_hip"])
    lh = as_xyz(joints["left_hip"])
    mid_shoulders = (rs + ls) / 2.0
    mid_hips = (rh + lh) / 2.0
    v_lr = rs - ls
    v_spine = mid_shoulders - mid_hips
    forward_vec = np.cross(v_lr, v_spine)
    norm = float(np.linalg.norm(forward_vec))
    if norm < 1e-8:
        return {name: 0 for name in joints}
    forward_vec /= norm
    flags = {}
    for name, pos in joints.items():
        parent = ROTATION_PARENT_JOINTS.get(name)
        if parent is None or parent not in joints:
            flags[name] = 0
            continue
        dot = float(np.dot(as_xyz(pos) - as_xyz(joints[parent]), forward_vec))
        flags[name] = 0 if abs(dot) < epsilon else (1 if dot > 0 else -1)
    return flags


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


def _project_to_image(points_3d):
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    valid = Z > 1e-6
    u = np.where(valid, _FX * (X / np.where(valid, Z, 1.0)) + _CX, np.nan)
    v = np.where(valid, _FY * (Y / np.where(valid, Z, 1.0)) + _CY, np.nan)
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
    k = min(OCCLUSION_NEIGHBORS, len(dists_2d))
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
        kp_3d = as_xyz(pos)
        if kp_3d[2] <= 0:
            continue
        u_k = _FX * (kp_3d[0] / kp_3d[2]) + _CX
        v_k = _FY * (kp_3d[1] / kp_3d[2]) + _CY
        if not _point_in_hull2d(np.array([u_k, v_k]), hull_2d_obj):
            continue
        z_local = _local_torso_z(u_k, v_k, all_torso_uv, all_torso_z)
        visibility[name] = not (kp_3d[2] > (z_local + occlusion_tau))
    return visibility


def compute_harmonic_precision(
    cam1,
    cam2,
    joint_names,
    vis1,
    vis2,
    alpha=HARMONIC_ALPHA,
    beta=HARMONIC_BETA,
    epsilon=HARMONIC_EPSILON,
):
    neighbors = {}
    for child, parent in RIGID_BONES_RATIO.keys():
        neighbors.setdefault(child, []).append(parent)
        neighbors.setdefault(parent, []).append(child)

    def calc_P(cam, vis):
        P = {}
        for name in joint_names:
            if name not in cam:
                P[name] = 0.0
                continue
            C = 1.0 if vis.get(name, True) else 0.0
            L = float(np.linalg.norm(as_xyz(cam[name])))
            P[name] = C / (1.0 + alpha * (L ** 2))
        return P

    def calc_H(P):
        H = {}
        for name in joint_names:
            p = P[name]
            nb = [P[n] for n in neighbors.get(name, []) if n in P]
            b = beta * (sum(nb) / len(nb)) if nb else p
            H[name] = (2.0 * b * p) / (b + p + epsilon)
        return H

    P1, P2 = calc_P(cam1, vis1), calc_P(cam2, vis2)
    H1, H2 = calc_H(P1), calc_H(P2)
    weights = {name: (H1[name] + H2[name]) / 2.0 for name in joint_names}
    return weights, H1, H2


def _confidence_value(confidence_by_joint, name):
    if not confidence_by_joint or name not in confidence_by_joint:
        return None
    value = confidence_by_joint[name]
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) < 3:
            return None
        value = value[2]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return max(0.0, value)


def _harmonic_blend(base_confidence, external_confidence, epsilon=HARMONIC_EPSILON):
    if external_confidence is None:
        return float(base_confidence)
    base_confidence = max(0.0, float(base_confidence))
    external_confidence = max(0.0, float(external_confidence))
    return float((2.0 * base_confidence * external_confidence) / (base_confidence + external_confidence + epsilon))


def _blend_detector_confidences(joint_names, base_confidences, external_confidences):
    return {
        name: _harmonic_blend(base_confidences[name], _confidence_value(external_confidences, name))
        for name in joint_names
    }


def detect_cross_view_errors(cam1, cam2, names, vis1, vis2, confidence2d1=None, confidence2d2=None):
    flags1 = get_orientation_flag(cam1)
    flags2 = get_orientation_flag(cam2)
    m_set = {
        n
        for n in names
        if (flags1.get(n, 0) == 1 and flags2.get(n, 0) == -1)
        or (flags1.get(n, 0) == -1 and flags2.get(n, 0) == 1)
    }

    _, H1_old, H2_old = compute_harmonic_precision(cam1, cam2, names, vis1, vis2)
    H1_all = _blend_detector_confidences(names, H1_old, confidence2d1)
    H2_all = _blend_detector_confidences(names, H2_old, confidence2d2)
    all_weights = {name: (H1_all[name] + H2_all[name]) / 2.0 for name in names}
    abs_diffs = [abs(H1_all[n] - H2_all[n]) for n in names]
    delta = min(float(np.percentile(abs_diffs, 75)) if abs_diffs else 0.0, CONFIDENCE_DELTA_CAP)
    k1_set = {n for n in names if H1_all[n] > H2_all[n] + delta}
    k2_set = {n for n in names if H2_all[n] > H1_all[n] + delta}
    k1_set.difference_update(NON_REPLACEABLE_ANCHORS)
    k2_set.difference_update(NON_REPLACEABLE_ANCHORS)

    l_list = [n for n in names if n not in (m_set | k1_set | k2_set)]
    return {
        "M": m_set,
        "K1": k1_set,
        "K2": k2_set,
        "L": l_list,
        "weights": all_weights,
        "H1": H1_all,
        "H2": H2_all,
        "flags1": flags1,
        "flags2": flags2,
    }


def _pairwise_joint_distance_stats(data):
    cam1, cam2 = data.get("camera1", {}), data.get("camera2", {})
    distances = []
    for j in sorted(set(cam1) & set(cam2)):
        try:
            d = np.linalg.norm(np.asarray(cam1[j])[:3] - np.asarray(cam2[j])[:3])
            distances.append(float(d))
        except Exception:
            pass
    if not distances:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.array(distances)
    return float(np.percentile(arr, 25)), float(np.percentile(arr, 75)), float(np.mean(arr)), float(np.median(arr))


def make_raw_judgement_fallback(data, index, error=None):
    stats = _pairwise_joint_distance_stats(data)
    common = sorted(set(data["camera1"]) & set(data["camera2"]))
    return {
        "M": [],
        "K1": [],
        "K2": [],
        "A_new": common,
        "F": [],
        "before_stats": stats,
        "after_stats": stats,
        "optimized": data,
        "fallback_reason": str(error),
        "joint_confidence": {"camera1": {j: 1.0 for j in common}, "camera2": {j: 1.0 for j in common}},
        "vis1": {j: True for j in common},
        "vis2": {j: True for j in common},
    }
