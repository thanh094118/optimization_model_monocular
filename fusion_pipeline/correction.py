from itertools import combinations

import numpy as np

from fusion_pipeline.detector import as_xyz


def _to_arrays(cam1, cam2, names):
    src = np.array([as_xyz(cam1[name]) for name in names], dtype=float)
    dst = np.array([as_xyz(cam2[name]) for name in names], dtype=float)
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
    return scale * (r @ as_xyz(point)) + t


def ransac_umeyama(cam1, cam2, names, threshold, max_combos, rng=None):
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


def estimate_bidirectional_similarity(cam1, cam2, candidate_names, threshold, max_combos):
    t12, anchor_names = ransac_umeyama(cam1, cam2, candidate_names, threshold=threshold, max_combos=max_combos)
    if len(anchor_names) >= 3:
        src_21 = np.array([cam2[n] for n in anchor_names], dtype=float)
        dst_21 = np.array([cam1[n] for n in anchor_names], dtype=float)
        t21 = estimate_umeyama(src_21, dst_21)
    else:
        t21 = (1.0, np.eye(3), np.zeros(3))
    return t12, t21, anchor_names


def apply_confidence_corrections(cam1, cam2, k1_set, k2_set, t12, t21):
    cam1_corr = dict(cam1)
    cam2_corr = dict(cam2)
    for name in k1_set:
        cam2_corr[name] = apply_similarity(cam1[name], t12)
    for name in k2_set:
        cam1_corr[name] = apply_similarity(cam2[name], t21)
    return cam1_corr, cam2_corr


def apply_rotation_mismatch_corrections(cam1_corr, cam2_corr, cam1, cam2, m_set, k1_set, k2_set, h1, h2, t12, t21):
    cam1_fixed = dict(cam1_corr)
    cam2_fixed = dict(cam2_corr)
    for name in m_set:
        if name in k1_set or name in k2_set:
            continue
        if h1.get(name, 0.5) > h2.get(name, 0.5):
            cam2_fixed[name] = apply_similarity(cam1[name], t12)
        else:
            cam1_fixed[name] = apply_similarity(cam2[name], t21)
    return cam1_fixed, cam2_fixed
