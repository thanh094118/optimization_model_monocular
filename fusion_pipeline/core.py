import copy
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from fusion_pipeline.geometry import _as_xyz, get_orientation_flag, compute_visibility_from_mesh_vertices
from json_io import write_json

RANSAC_THRESHOLD = 0.05
RANSAC_MAX_COMBOS = 500
HEIGHT = 1.63
RIGID_BONES_RATIO = {
    ("left_elbow", "left_shoulder"): 0.186,
    ("left_hand", "left_elbow"): 0.146,
    ("right_elbow", "right_shoulder"): 0.186,
    ("right_hand", "right_elbow"): 0.146,
    ("left_knee", "left_hip"): 0.245,
    ("left_ankle", "left_knee"): 0.246,
    ("right_knee", "right_hip"): 0.245,
    ("right_ankle", "right_knee"): 0.246,
}


def _to_arrays(cam1, cam2, names):
    src = np.array([_as_xyz(cam1[name]) for name in names], dtype=float)
    dst = np.array([_as_xyz(cam2[name]) for name in names], dtype=float)
    return src, dst


def estimate_umeyama(src, dst):
    n, m = src.shape
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    sigma = np.mean(np.sum(src_c ** 2, axis=1))
    h = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(h)
    s_mat = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(vt.T) < 0:
        s_mat[m - 1, m - 1] = -1
    r = u @ s_mat @ vt
    scale = 1.0 if sigma < 1e-12 else float(np.trace(np.diag(d) @ s_mat) / sigma)
    t = mu_d - scale * (r @ mu_s)
    return scale, r, t


def apply_similarity(point, transform):
    scale, r, t = transform
    return scale * (r @ _as_xyz(point)) + t


def ransac_umeyama(cam1, cam2, names, threshold=RANSAC_THRESHOLD, max_combos=RANSAC_MAX_COMBOS, rng=None):
    if len(names) < 3:
        return (1.0, np.eye(3), np.zeros(3)), list(names)
    src_all, dst_all = _to_arrays(cam1, cam2, names)
    n = len(names)
    c3 = n * (n - 1) * (n - 2) // 6
    if rng is None:
        rng = np.random.default_rng(42)
    triplets = list(combinations(range(n), 3)) if c3 <= max_combos else [tuple(rng.choice(n, 3, replace=False)) for _ in range(max_combos)]
    best_inliers = []
    for tri in triplets:
        tri = list(tri)
        try:
            tf = estimate_umeyama(src_all[tri], dst_all[tri])
        except np.linalg.LinAlgError:
            continue
        pred = np.array([apply_similarity(src_all[i], tf) for i in range(n)])
        err = np.linalg.norm(pred - dst_all, axis=1)
        inliers = np.where(err < threshold)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if not best_inliers:
        best_inliers = list(range(n))
    inlier_names = [names[i] for i in best_inliers]
    tf_refined = estimate_umeyama(src_all[best_inliers], dst_all[best_inliers])
    return tf_refined, inlier_names


def get_diff_f(f_name, anchors, cam1, cam2, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=0.25):
    p1_f = _as_xyz(cam1[f_name])
    p2_f = _as_xyz(cam2[f_name])
    sum_wdiff, sum_w = 0.0, 0.0
    for a in anchors:
        ca1 = float(conf1.get(a, 1.0)) if conf1 else 1.0
        ca2 = float(conf2.get(a, 1.0)) if conf2 else 1.0
        va1 = 1.0 if (vis1 and vis1.get(a, True)) else float(occluded_factor)
        va2 = 1.0 if (vis2 and vis2.get(a, True)) else float(occluded_factor)
        w = ((ca1 + ca2) / 2.0) * (va1 * va2)
        d1 = np.linalg.norm(p1_f - _as_xyz(cam1[a]))
        d2 = np.linalg.norm(p2_f - _as_xyz(cam2[a]))
        sum_wdiff += w * abs(d1 - d2)
        sum_w += w
    return sum_wdiff / max(sum_w, 1e-12)


def calculate_stats(cam1, cam2, f_list, anchors, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=0.25, f_weights=None, huber_delta=0.05):
    if not f_list or not anchors:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = [get_diff_f(f, anchors, cam1, cam2, conf1=conf1, conf2=conf2, vis1=vis1, vis2=vis2, occluded_factor=occluded_factor) for f in f_list]
    if f_weights:
        diffs = [d * f_weights[n] for d, n in zip(diffs, f_list)]
    arr_diffs = np.array(diffs, dtype=float)
    q1 = float(np.percentile(arr_diffs, 25))
    q3 = float(np.percentile(arr_diffs, 75))
    mean_val = float(np.mean(arr_diffs))
    median_val = float(np.median(arr_diffs))
    huber = np.where(arr_diffs <= huber_delta, 0.5 * arr_diffs ** 2, huber_delta * (arr_diffs - 0.5 * huber_delta))
    huber_loss = float(np.mean(huber))
    return q1, q3, mean_val, median_val, huber_loss


def compute_dynamic_scale(cam_dict, f_list, ratios):
    sum_len, sum_ratio = 0.0, 0.0
    for (c, p), r in ratios.items():
        if c not in f_list and p not in f_list and c in cam_dict and p in cam_dict:
            sum_len += float(np.linalg.norm(cam_dict[c] - cam_dict[p]))
            sum_ratio += r
    return (sum_len / sum_ratio) if sum_ratio > 0 else HEIGHT


def compute_harmonic_precision(cam1, cam2, joint_names, vis1, vis2, alpha=0.001, beta=0.8, epsilon=1e-6):
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
            L = float(np.linalg.norm(_as_xyz(cam[name])))
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


def optimize_f_points(data, anchors, f_list, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=0.25, regularization=False, regularization_lambda=1.0, prev_data=None, temporal_lambda=1.0, max_iter=1000):
    cam1 = {k: _as_xyz(v) for k, v in data["camera1"].items()}
    cam2 = {k: _as_xyz(v) for k, v in data["camera2"].items()}
    f_weights = {}
    for name in f_list:
        c1 = float(conf1.get(name, 1.0)) if conf1 else 1.0
        c2 = float(conf2.get(name, 1.0)) if conf2 else 1.0
        v1 = 1.0 if (vis1 and vis1.get(name, True)) else float(occluded_factor)
        v2 = 1.0 if (vis2 and vis2.get(name, True)) else float(occluded_factor)
        f_weights[name] = ((c1 + c2) / 2.0) * (v1 * v2)

    num_f = len(f_list)
    if not f_list:
        return {"camera1": cam1, "camera2": cam2}, None

    def proximity_penalty(x):
        penalty = 0.0
        for i, name in enumerate(f_list):
            c1v = float(conf1.get(name, 1.0)) if conf1 else 1.0
            c2v = float(conf2.get(name, 1.0)) if conf2 else 1.0
            d1 = x[i * 3:i * 3 + 3] - cam1[name]
            d2 = x[(num_f + i) * 3:(num_f + i) * 3 + 3] - cam2[name]
            penalty += c1v * float(np.dot(d1, d1)) + c2v * float(np.dot(d2, d2))
        return penalty

    def temporal_penalty(x):
        if prev_data is None:
            return 0.0
        prev_cam1 = prev_data.get("camera1", {})
        prev_cam2 = prev_data.get("camera2", {})
        penalty = 0.0
        for i, name in enumerate(f_list):
            if name in prev_cam1:
                p1_curr = x[i * 3:i * 3 + 3]
                p1_prev = np.asarray(prev_cam1[name], dtype=float)
                penalty += float(np.sum((p1_curr - p1_prev) ** 2))
            if name in prev_cam2:
                p2_curr = x[(num_f + i) * 3:(num_f + i) * 3 + 3]
                p2_prev = np.asarray(prev_cam2[name], dtype=float)
                penalty += float(np.sum((p2_curr - p2_prev) ** 2))
        return penalty

    def objective(x):
        p1, p2 = dict(cam1), dict(cam2)
        for i, name in enumerate(f_list):
            p1[name] = x[i * 3:i * 3 + 3]
            p2[name] = x[(num_f + i) * 3:(num_f + i) * 3 + 3]
        _, _, _, _, huber_loss = calculate_stats(p1, p2, f_list, anchors, conf1=conf1, conf2=conf2, vis1=vis1, vis2=vis2, occluded_factor=occluded_factor, f_weights=f_weights, huber_delta=0.05)
        obj_val = huber_loss
        if regularization:
            obj_val += regularization_lambda * proximity_penalty(x)
        if prev_data is not None:
            obj_val += temporal_lambda * temporal_penalty(x)
        return obj_val

    constraints = []
    dyn_scale_cam1 = compute_dynamic_scale(cam1, f_list, RIGID_BONES_RATIO)
    dyn_scale_cam2 = compute_dynamic_scale(cam2, f_list, RIGID_BONES_RATIO)
    for (child, parent), ratio in RIGID_BONES_RATIO.items():
        if child not in cam1 or parent not in cam1 or (child not in f_list and parent not in f_list):
            continue
        target1 = ratio * dyn_scale_cam1
        lower_sq1 = (0.85 * target1) ** 2
        upper_sq1 = (1.15 * target1) ** 2

        def constr_lower1(x, c=child, p=parent, low=lower_sq1):
            pts = dict(cam1)
            for i, name in enumerate(f_list):
                pts[name] = x[i * 3:i * 3 + 3]
            dist_sq = float(np.dot(pts[c] - pts[p], pts[c] - pts[p]))
            return dist_sq - low

        def constr_upper1(x, c=child, p=parent, up=upper_sq1):
            pts = dict(cam1)
            for i, name in enumerate(f_list):
                pts[name] = x[i * 3:i * 3 + 3]
            dist_sq = float(np.dot(pts[c] - pts[p], pts[c] - pts[p]))
            return up - dist_sq

        constraints.append({"type": "ineq", "fun": constr_lower1})
        constraints.append({"type": "ineq", "fun": constr_upper1})

        target2 = ratio * dyn_scale_cam2
        lower_sq2 = (0.85 * target2) ** 2
        upper_sq2 = (1.15 * target2) ** 2

        def constr_lower2(x, c=child, p=parent, low=lower_sq2):
            pts = dict(cam2)
            for i, name in enumerate(f_list):
                pts[name] = x[(num_f + i) * 3:(num_f + i) * 3 + 3]
            dist_sq = float(np.dot(pts[c] - pts[p], pts[c] - pts[p]))
            return dist_sq - low

        def constr_upper2(x, c=child, p=parent, up=upper_sq2):
            pts = dict(cam2)
            for i, name in enumerate(f_list):
                pts[name] = x[(num_f + i) * 3:(num_f + i) * 3 + 3]
            dist_sq = float(np.dot(pts[c] - pts[p], pts[c] - pts[p]))
            return up - dist_sq

        constraints.append({"type": "ineq", "fun": constr_lower2})
        constraints.append({"type": "ineq", "fun": constr_upper2})

    x0 = []
    for name in f_list:
        x0.extend(cam1[name])
    for name in f_list:
        x0.extend(cam2[name])

    res = minimize(objective, np.array(x0, dtype=float), constraints=constraints, method="SLSQP", options={"maxiter": max_iter})
    if not res.success:
        raise RuntimeError("Optimization failed: {}".format(res.message))

    p1_opt, p2_opt = dict(cam1), dict(cam2)
    for i, name in enumerate(f_list):
        p1_opt[name] = res.x[i * 3:i * 3 + 3].tolist()
        p2_opt[name] = res.x[(num_f + i) * 3:(num_f + i) * 3 + 3].tolist()
    return {"camera1": p1_opt, "camera2": p2_opt}, res


def run_phase3_pipeline(data_in, verts_by_cam=None, occlusion_tau=0.05, regularization=False, regularization_lambda=1.0, temporal_lambda=2.0, max_iter=1000, ransac_threshold=0.05, ransac_max_combos=500, frame_idx=None, prev_optimized_data=None, debug1_dir=None, debug2_dir=None):
    cam1 = {k: _as_xyz(v) for k, v in data_in["camera1"].items()}
    cam2 = {k: _as_xyz(v) for k, v in data_in["camera2"].items()}
    names = sorted(set(cam1.keys()) & set(cam2.keys()))
    if not names:
        raise ValueError("No common joints between camera1 and camera2")
    cam1 = {k: cam1[k] for k in names}
    cam2 = {k: cam2[k] for k in names}

    if verts_by_cam is not None:
        vis1 = compute_visibility_from_mesh_vertices(cam1, verts_by_cam["camera1"], occlusion_tau)
        vis2 = compute_visibility_from_mesh_vertices(cam2, verts_by_cam["camera2"], occlusion_tau)
    else:
        vis1 = {n: True for n in names}
        vis2 = {n: True for n in names}

    flags1 = get_orientation_flag(cam1)
    flags2 = get_orientation_flag(cam2)
    m_set = {n for n in names if (flags1.get(n, 0) == 1 and flags2.get(n, 0) == -1) or (flags1.get(n, 0) == -1 and flags2.get(n, 0) == 1)}

    all_weights, H1_all, H2_all = compute_harmonic_precision(cam1, cam2, names, vis1, vis2)
    abs_diffs = [abs(H1_all[n] - H2_all[n]) for n in names]
    delta = min(float(np.percentile(abs_diffs, 75)) if abs_diffs else 0.0, 0.05)
    k1_set = {n for n in names if H1_all[n] > H2_all[n] + delta}
    k2_set = {n for n in names if H2_all[n] > H1_all[n] + delta}
    not_K = {"left_hip", "left_shoulder", "right_hip", "right_shoulder"}
    k1_set.difference_update(not_K)
    k2_set.difference_update(not_K)

    l_list = [n for n in names if n not in (m_set | k1_set | k2_set)]
    t12, a_list = ransac_umeyama(cam1, cam2, l_list, threshold=ransac_threshold, max_combos=ransac_max_combos)
    if len(a_list) >= 3:
        src_21 = np.array([cam2[n] for n in a_list], dtype=float)
        dst_21 = np.array([cam1[n] for n in a_list], dtype=float)
        t21 = estimate_umeyama(src_21, dst_21)
    else:
        t21 = (1.0, np.eye(3), np.zeros(3))

    cam1_corr = dict(cam1)
    cam2_corr = dict(cam2)
    for n in k1_set:
        cam2_corr[n] = apply_similarity(cam1[n], t12)
    for n in k2_set:
        cam1_corr[n] = apply_similarity(cam2[n], t21)

    if debug1_dir is not None:
        write_json(debug1_dir / "fused_data_{}.json".format(frame_idx), {"M": sorted(m_set), "K1": sorted(k1_set), "K2": sorted(k2_set), "optimized": {"camera1": cam1_corr, "camera2": cam2_corr}, "step": "after_K1_K2", "joint_confidence": {"camera1": H1_all, "camera2": H2_all}, "vis1": vis1, "vis2": vis2})

    for n in m_set:
        if n not in k1_set and n not in k2_set:
            if H1_all.get(n, 0.5) > H2_all.get(n, 0.5):
                cam2_corr[n] = apply_similarity(cam1[n], t12)
            else:
                cam1_corr[n] = apply_similarity(cam2[n], t21)

    if debug2_dir is not None:
        write_json(debug2_dir / "fused_data_{}.json".format(frame_idx), {"M": sorted(m_set), "K1": sorted(k1_set), "K2": sorted(k2_set), "optimized": {"camera1": cam1_corr, "camera2": cam2_corr}, "step": "after_M", "joint_confidence": {"camera1": H1_all, "camera2": H2_all}, "vis1": vis1, "vis2": vis2})

    a_new = sorted(set(a_list) | k1_set | k2_set)
    f_list = [n for n in names if n not in set(a_new)]
    before_stats = calculate_stats(cam1_corr, cam2_corr, names, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)
    optimized_data, res = optimize_f_points({"camera1": cam1_corr, "camera2": cam2_corr}, a_new, f_list, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, regularization=regularization, regularization_lambda=regularization_lambda, prev_data=prev_optimized_data, temporal_lambda=temporal_lambda, max_iter=max_iter)
    after_stats = calculate_stats(optimized_data["camera1"], optimized_data["camera2"], names, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)

    flags1_after = get_orientation_flag(optimized_data["camera1"])
    flags2_after = get_orientation_flag(optimized_data["camera2"])
    m_after = {n for n in names if (flags1_after.get(n, 0) == 1 and flags2_after.get(n, 0) == -1) or (flags1_after.get(n, 0) == -1 and flags2_after.get(n, 0) == 1)}

    return {
        "M": sorted(m_set),
        "M_after": sorted(m_after),
        "M_resolved": len(m_after) == 0 and len(m_set) > 0,
        "K1": sorted(k1_set),
        "K2": sorted(k2_set),
        "A_new": a_new,
        "F": f_list,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "optimized": {"camera1": {k: list(v) for k, v in optimized_data["camera1"].items()}, "camera2": {k: list(v) for k, v in optimized_data["camera2"].items()}},
        "joint_confidence": {"camera1": H1_all, "camera2": H2_all},
        "vis1": {k: bool(v) for k, v in vis1.items()},
        "vis2": {k: bool(v) for k, v in vis2.items()},
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
        "M": [], "K1": [], "K2": [], "A_new": common, "F": [],
        "before_stats": stats, "after_stats": stats,
        "optimized": data,
        "fallback_reason": str(error),
        "joint_confidence": {"camera1": {j: 1.0 for j in common}, "camera2": {j: 1.0 for j in common}},
        "vis1": {j: True for j in common},
        "vis2": {j: True for j in common},
    }
