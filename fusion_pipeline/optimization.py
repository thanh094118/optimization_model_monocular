import numpy as np
from scipy.optimize import minimize

from fusion_pipeline.config import (
    BONE_LENGTH_MAX_SCALE,
    BONE_LENGTH_MIN_SCALE,
    DEFAULT_OCCLUDED_FACTOR,
    HEIGHT,
    HUBER_DELTA,
    RIGID_BONES_RATIO,
)
from fusion_pipeline.detector import as_xyz


def get_diff_f(f_name, anchors, cam1, cam2, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR):
    p1_f = as_xyz(cam1[f_name])
    p2_f = as_xyz(cam2[f_name])
    sum_wdiff, sum_w = 0.0, 0.0
    for a in anchors:
        ca1 = float(conf1.get(a, 1.0)) if conf1 else 1.0
        ca2 = float(conf2.get(a, 1.0)) if conf2 else 1.0
        va1 = 1.0 if (vis1 and vis1.get(a, True)) else float(occluded_factor)
        va2 = 1.0 if (vis2 and vis2.get(a, True)) else float(occluded_factor)
        w = ((ca1 + ca2) / 2.0) * (va1 * va2)
        d1 = np.linalg.norm(p1_f - as_xyz(cam1[a]))
        d2 = np.linalg.norm(p2_f - as_xyz(cam2[a]))
        sum_wdiff += w * abs(d1 - d2)
        sum_w += w
    return sum_wdiff / max(sum_w, 1e-12)


def calculate_stats(cam1, cam2, f_list, anchors, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR, f_weights=None, huber_delta=HUBER_DELTA):
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


def optimize_f_points(data, anchors, f_list, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR, regularization=False, regularization_lambda=1.0, prev_data=None, temporal_lambda=1.0, max_iter=1000):
    cam1 = {k: as_xyz(v) for k, v in data["camera1"].items()}
    cam2 = {k: as_xyz(v) for k, v in data["camera2"].items()}
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
        _, _, _, _, huber_loss = calculate_stats(p1, p2, f_list, anchors, conf1=conf1, conf2=conf2, vis1=vis1, vis2=vis2, occluded_factor=occluded_factor, f_weights=f_weights, huber_delta=HUBER_DELTA)
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
        lower_sq1 = (BONE_LENGTH_MIN_SCALE * target1) ** 2
        upper_sq1 = (BONE_LENGTH_MAX_SCALE * target1) ** 2

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
        lower_sq2 = (BONE_LENGTH_MIN_SCALE * target2) ** 2
        upper_sq2 = (BONE_LENGTH_MAX_SCALE * target2) ** 2

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
