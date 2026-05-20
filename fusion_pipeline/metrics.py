import numpy as np
from fusion_pipeline.geometry import _as_xyz
from fusion_pipeline.constants import RIGID_BONES_RATIO, HEIGHT


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


def calculate_stats(cam1, cam2, f_list, anchors, conf1=None, conf2=None, vis1=None, vis2=None,
                    occluded_factor=0.25, f_weights=None, huber_delta=0.05):
    if not f_list or not anchors:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = [
        get_diff_f(f, anchors, cam1, cam2, conf1=conf1, conf2=conf2, vis1=vis1, vis2=vis2,
                   occluded_factor=occluded_factor)
        for f in f_list
    ]
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

    def calc_p(cam, vis):
        p_map = {}
        for name in joint_names:
            if name not in cam:
                p_map[name] = 0.0
                continue
            c = 1.0 if vis.get(name, True) else 0.0
            length = float(np.linalg.norm(_as_xyz(cam[name])))
            p_map[name] = c / (1.0 + alpha * (length ** 2))
        return p_map

    def calc_h(p_map):
        h_map = {}
        for name in joint_names:
            p = p_map[name]
            nb = [p_map[n] for n in neighbors.get(name, []) if n in p_map]
            b = beta * (sum(nb) / len(nb)) if nb else p
            h_map[name] = (2.0 * b * p) / (b + p + epsilon)
        return h_map

    p1, p2 = calc_p(cam1, vis1), calc_p(cam2, vis2)
    h1, h2 = calc_h(p1), calc_h(p2)
    weights = {name: (h1[name] + h2[name]) / 2.0 for name in joint_names}
    return weights, h1, h2


def pairwise_joint_distance_stats(data):
    cam1, cam2 = data.get("camera1", {}), data.get("camera2", {})
    distances = []
    for j in sorted(set(cam1) & set(cam2)):
        try:
            d = np.linalg.norm(np.asarray(cam1[j])[:3] - np.asarray(cam2[j])[:3])
            distances.append(float(d))
        except Exception:
            pass
    if not distances:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    arr = np.array(distances)
    return (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)), float(np.mean(arr)), float(np.median(arr)))
