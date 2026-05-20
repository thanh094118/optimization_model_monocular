import numpy as np
from scipy.optimize import minimize
from fusion_pipeline.geometry import _as_xyz
from fusion_pipeline.metrics import calculate_stats, compute_dynamic_scale
from fusion_pipeline.constants import RIGID_BONES_RATIO


def optimize_f_points(data, anchors, f_list, conf1=None, conf2=None, vis1=None, vis2=None,
                      occluded_factor=0.25, regularization=False, regularization_lambda=1.0,
                      prev_data=None, temporal_lambda=1.0, max_iter=1000):
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
        _, _, _, _, huber_loss = calculate_stats(
            p1, p2, f_list, anchors,
            conf1=conf1, conf2=conf2, vis1=vis1, vis2=vis2,
            occluded_factor=occluded_factor, f_weights=f_weights, huber_delta=0.05,
        )
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

    res = minimize(objective, np.array(x0, dtype=float), constraints=constraints, method="SLSQP",
                   options={"maxiter": int(max_iter)})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    p1_opt, p2_opt = dict(cam1), dict(cam2)
    for i, name in enumerate(f_list):
        p1_opt[name] = res.x[i * 3:i * 3 + 3].tolist()
        p2_opt[name] = res.x[(num_f + i) * 3:(num_f + i) * 3 + 3].tolist()
    return {"camera1": p1_opt, "camera2": p2_opt}, res
