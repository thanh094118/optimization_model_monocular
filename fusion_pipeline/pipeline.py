from fusion_pipeline.correction import (
    apply_confidence_corrections,
    apply_rotation_mismatch_corrections,
    estimate_bidirectional_similarity,
)
from fusion_pipeline.detection import compute_visibility_from_mesh_vertices, detect_cross_view_errors
from fusion_pipeline.geometry import _as_xyz, get_orientation_flag
from fusion_pipeline.optimization import calculate_stats, optimize_f_points
from json_io import write_json


def run_phase3_pipeline(data_in, verts_by_cam=None, occlusion_tau=0.05, regularization=False, regularization_lambda=1.0, temporal_lambda=2.0, max_iter=1000, ransac_threshold=0.05, ransac_max_combos=500, frame_idx=None, prev_optimized_data=None, debug1_dir=None, debug2_dir=None, confidence2d_by_cam=None):
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

    confidence2d_by_cam = confidence2d_by_cam or {}
    detected = detect_cross_view_errors(
        cam1,
        cam2,
        names,
        vis1,
        vis2,
        confidence2d1=confidence2d_by_cam.get("camera1"),
        confidence2d2=confidence2d_by_cam.get("camera2"),
    )
    m_set = detected["M"]
    k1_set = detected["K1"]
    k2_set = detected["K2"]
    l_list = detected["L"]
    all_weights = detected["weights"]
    H1_all = detected["H1"]
    H2_all = detected["H2"]

    t12, t21, a_list = estimate_bidirectional_similarity(
        cam1,
        cam2,
        l_list,
        threshold=ransac_threshold,
        max_combos=ransac_max_combos,
    )

    cam1_corr, cam2_corr = apply_confidence_corrections(cam1, cam2, k1_set, k2_set, t12, t21)
    if debug1_dir is not None:
        write_json(debug1_dir / "fused_data_{}.json".format(frame_idx), {"camera1": {k: list(v) for k, v in cam1_corr.items()}, "camera2": {k: list(v) for k, v in cam2_corr.items()}})

    cam1_corr, cam2_corr = apply_rotation_mismatch_corrections(
        cam1_corr,
        cam2_corr,
        cam1,
        cam2,
        m_set,
        k1_set,
        k2_set,
        H1_all,
        H2_all,
        t12,
        t21,
    )
    if debug2_dir is not None:
        write_json(debug2_dir / "fused_data_{}.json".format(frame_idx), {"camera1": {k: list(v) for k, v in cam1_corr.items()}, "camera2": {k: list(v) for k, v in cam2_corr.items()}})

    a_new = sorted(set(a_list) | k1_set | k2_set)
    f_list = [n for n in names if n not in set(a_new)]
    before_stats = calculate_stats(cam1_corr, cam2_corr, names, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)
    optimized_data, _ = optimize_f_points(
        {"camera1": cam1_corr, "camera2": cam2_corr},
        a_new,
        f_list,
        conf1=H1_all,
        conf2=H2_all,
        vis1=vis1,
        vis2=vis2,
        regularization=regularization,
        regularization_lambda=regularization_lambda,
        prev_data=prev_optimized_data,
        temporal_lambda=temporal_lambda,
        max_iter=max_iter,
    )
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
