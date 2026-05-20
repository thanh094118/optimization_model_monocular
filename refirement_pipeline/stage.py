import os
import sys
import pickle
import traceback
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
import joblib
from loguru import logger

from refirement_pipeline.wham_loader import load_wham_output
from refirement_pipeline.subject_loader import load_subject_params


def _add_external_repo_to_path(external_repo_root):
    if not external_repo_root:
        return

    root = os.path.abspath(external_repo_root)

    if root not in sys.path:
        sys.path.insert(0, root)


def _load_external_modules(external_repo_root):
    _add_external_repo_to_path(external_repo_root)

    from refirement_pipeline.utils import utils_optim as ut
    from refirement_pipeline.utils.optimization_formulations import OptimizeExtrinsics, OptimizePose
    from refirement_pipeline.utils.utils_activity_classification import predict_activity_from_video
    from refirement_pipeline.utils.utilsCameraPy3 import rotateIntrinsics, getVideoRotation
    from refirement_pipeline.utils.utilsChecker import detectGait

    return {
        "ut": ut,
        "OptimizeExtrinsics": OptimizeExtrinsics,
        "OptimizePose": OptimizePose,
        "predict_activity_from_video": predict_activity_from_video,
        "rotateIntrinsics": rotateIntrinsics,
        "getVideoRotation": getVideoRotation,
        "detectGait": detectGait,
    }


def run_refirement_optimization(config):
    cfg = config.get("refirement", {})

    if not cfg.get("enabled", False):
        logger.info("Running refirement because stage was explicitly requested.")

    external = _load_external_modules(cfg.get("external_repo_root", "refirement"))

    subject = load_subject_params(cfg["subject_params"])

    return run_optimization_only(
        external=external,
        data_dir=cfg["data_dir"],
        wham_file=cfg.get("wham_file", "wham_output.pkl"),
        video_path=cfg["video"],
        intrinsics_pth=cfg["intrinsics"],
        parameters_yaml=cfg.get("parameters_yaml", "configs/parameters.yaml"),
        height_m=subject["height_m"],
        mass_kg=subject["mass_kg"],
        sex=subject["sex"],
        activity=cfg.get("activity"),
        rotation=cfg.get("rotation"),
        use_gpu=cfg.get("use_gpu", True),
        static_cam=cfg.get("static_cam", True),
        n_iter_opt2=cfg.get("n_iter_opt2", 75),
        filter_freq=cfg.get("filter_freq", 6),
        smoothness_diff_n=cfg.get("smoothness_diff_n", 1),
        print_loss_terms=cfg.get("print_loss_terms", False),
        plotting=cfg.get("plotting", False),
        save_smpl_for_viz=cfg.get("save_smpl_for_viz", True),
    )


def run_optimization_only(
    external,
    data_dir,
    wham_file,
    video_path,
    intrinsics_pth,
    parameters_yaml,
    height_m,
    mass_kg,
    sex,
    activity=None,
    rotation=None,
    use_gpu=True,
    static_cam=True,
    n_iter_opt2=75,
    filter_freq=6,
    smoothness_diff_n=1,
    print_loss_terms=False,
    plotting=False,
    save_smpl_for_viz=True,
    weights_opt2=None,
):
    ut = external["ut"]
    OptimizeExtrinsics = external["OptimizeExtrinsics"]
    OptimizePose = external["OptimizePose"]
    predict_activity_from_video = external["predict_activity_from_video"]
    rotateIntrinsics = external["rotateIntrinsics"]
    getVideoRotation = external["getVideoRotation"]
    detectGait = external["detectGait"]

    output_path = data_dir
    trial_name = os.path.basename(data_dir.rstrip("/"))

    output_paths = {
        "output_dir": output_path,
        "wham_output_pkl": None,
        "trimmed_video": None,
        "optimized_pkl": None,
        "keypoints_3d_cam_pkl": None,
        "vertices_3d_cam_pkl": None,
        "plot_objective": None,
        "plot_2d": None,
        "trc_file": None,
        "scaled_model_file": None,
        "ik_results_file": None,
        "predicted_activity": None,
        "activity_detection_method": None,
    }

    torch.cuda.empty_cache()

    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        logger.info("Using GPU for optimization.")
    else:
        device = "cpu"
        if use_gpu:
            logger.warning("No GPU available, falling back to CPU.")

    wham_pkl_path = os.path.join(data_dir, wham_file)

    if not os.path.exists(wham_pkl_path):
        raise FileNotFoundError("WHAM pkl not found: {}".format(wham_pkl_path))

    wham_regression_results = load_wham_output(wham_pkl_path)
    output_paths["wham_output_pkl"] = wham_pkl_path

    logger.info("Loaded WHAM pkl: {} people".format(len(wham_regression_results)))

    intrinsics = ut.load_intrinsics(intrinsics_pth)

    cap, frame_rate, n_frames_video, _ = ut.get_video_info(
        video_path=video_path,
        release=False,
    )

    frame_rate = round(frame_rate, 0)
    logger.info("Frame rate: {}, total frames: {}".format(frame_rate, n_frames_video))

    if rotation is None:
        rotation = getVideoRotation(video_path)

    logger.info("Video rotation: {}°".format(rotation))

    vid_width = None
    vid_height = None

    if cap is not None:
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if "calib_portrait_h" in intrinsics and "calib_portrait_w" in intrinsics:
            calib_h = intrinsics.pop("calib_portrait_h")
            calib_w = intrinsics.pop("calib_portrait_w")

            if rotation in [0, 180]:
                actual_portrait_h = float(vid_width)
                actual_portrait_w = float(vid_height)
            else:
                actual_portrait_h = float(vid_height)
                actual_portrait_w = float(vid_width)

            scale_x = actual_portrait_w / calib_w
            scale_y = actual_portrait_h / calib_h

            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                intrinsics["fx"] *= scale_x
                intrinsics["fy"] *= scale_y
                intrinsics["cx"] *= scale_x
                intrinsics["cy"] *= scale_y

    if rotation in [0, 180] and vid_width is not None and vid_height is not None:
        image_size = [vid_width, vid_height]
        intrinsics = rotateIntrinsics(intrinsics, rotation, imageSize=image_size)

    num_people = len(wham_regression_results)

    if num_people > 1:
        logger.warning("Detected {} people, selecting first non-empty.".format(num_people))
        wham_result = None

        for i in range(num_people):
            if len(wham_regression_results[i]) > 0:
                wham_result = wham_regression_results[i]
                break

        if wham_result is None:
            raise RuntimeError("No valid person in WHAM result.")
    else:
        wham_result = wham_regression_results[0]

    if "contact" in wham_result:
        try:
            wham_result["contact"] = ut.filter_array(
                wham_result["contact"],
                order=4,
                cutoff_freq=6,
                sampling_rate=frame_rate,
            )
        except Exception as e:
            logger.warning("Error filtering contact: {}. Using zeros.".format(e))
            n_wham = len(wham_result.get("trans_world", []))
            wham_result["contact"] = np.zeros((n_wham, 4), dtype=np.float32)
    else:
        logger.warning("No contact key. Using zeros.")
        n_wham = len(wham_result.get("trans_world", []))
        wham_result["contact"] = np.zeros((n_wham, 4), dtype=np.float32)

    wham_result = {
        k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
        for k, v in wham_result.items()
    }

    try:
        wham_result["tracking_results_for_reproj"]["keypoints"] = (
            wham_result["tracking_results_for_reproj"]["keypoints"].astype(np.float32)
        )
    except Exception as e:
        logger.warning("Could not convert keypoints dtype: {}".format(e))

    key2d = ut.get_openpose_keypoints(
        wham_result["tracking_results_for_reproj"]["keypoints"],
        filter_freq=filter_freq,
        sample_rate=frame_rate,
        device=device,
    )

    if "frame_id" in wham_result:
        frame_ids = wham_result["frame_id"]
    elif "frame_ids" in wham_result:
        frame_ids = wham_result["frame_ids"]
    else:
        logger.warning("No frame_id found. Using sequential indices.")
        frame_ids = np.arange(len(wham_result["trans_world"]))

    n_wham_frames = len(frame_ids)
    opencap_mono_frame_range_wham_ref = range(n_wham_frames)

    if isinstance(frame_ids, torch.Tensor):
        frame_ids_arr = frame_ids.detach().cpu().numpy().astype(np.int64).ravel()
    else:
        frame_ids_arr = np.asarray(frame_ids, dtype=np.int64).ravel()

    first_frame_id = int(frame_ids_arr[0])
    last_frame_id = int(frame_ids_arr[-1])
    opencap_mono_frame_range_video_ref = range(first_frame_id, last_frame_id + 1)

    trimmed_video_path = ut.save_trimmed_video(
        data_dir,
        opencap_mono_frame_range_video_ref,
        video_path,
        ffmpeg=True,
    )

    output_paths["trimmed_video"] = trimmed_video_path

    beta = ut.compute_mean_beta(wham_result["betas"])
    gender = "female" if sex in ("f", "female") else "male"
    smpl_model, _ = ut.load_smpl(device=device, gender=gender)

    n_frames = len(opencap_mono_frame_range_wham_ref)

    r_world_to_cam = torch.empty(wham_result["poses_root_world"].shape, device=device)
    t_world_to_cam = torch.empty(wham_result["trans_world"].shape, device=device)
    key3d_op_smpl = torch.empty(n_frames, 25, 3, device=device)

    r_world_to_cam = r_world_to_cam[opencap_mono_frame_range_wham_ref, :, :]
    t_world_to_cam = t_world_to_cam[opencap_mono_frame_range_wham_ref, :]

    for i_frame in range(n_frames):
        cam_i_frame = 0 if static_cam else i_frame

        if i_frame == cam_i_frame:
            r_world_to_root = wham_result["poses_root_world"][cam_i_frame, :, :]
            r_cam_to_root = wham_result["poses_root_cam"][cam_i_frame, :, :]

            r_world_to_cam[cam_i_frame, :, :] = torch.matmul(
                r_world_to_root,
                r_cam_to_root.T,
            )

            t_world_to_cam[cam_i_frame, :] = (
                wham_result["trans_world"][cam_i_frame, :]
                - torch.matmul(
                    r_world_to_cam[cam_i_frame, :, :],
                    wham_result["trans_cam"][cam_i_frame, :],
                )
            )

        smpl_result = ut.pred_smpl(
            smpl_model,
            trans=wham_result["trans_world"][i_frame, :].reshape((1, 1, 3)),
            root_orient=wham_result["pose_world"][i_frame, :3].reshape((1, 1, 3)),
            body_pose=wham_result["pose_world"][i_frame, 3:].reshape((1, 1, -1)),
            betas=beta.reshape((1, len(beta))),
        )

        key3d_op_smpl[i_frame, :, :] = smpl_result["joints3d_op"]

    if static_cam and n_frames > 1:
        r_world_to_cam = r_world_to_cam[0:1, :, :].expand(n_frames, -1, -1).clone()
        t_world_to_cam = t_world_to_cam[0:1, :].expand(n_frames, -1).clone()

    ankle_indices = [11, 14]
    ankle_positions = key3d_op_smpl[:, ankle_indices, :].cpu().numpy()
    ankle_velocities = np.diff(ankle_positions, axis=0)

    is_gait = detectGait(
        ankle_velocities[:, 0, 1],
        ankle_velocities[:, 1, 1],
        frame_rate,
    )

    predicted_activity = None
    activity_detection_method = None

    if activity is not None:
        predicted_activity = activity
        activity_detection_method = "user_provided"
    else:
        try:
            predicted_activity, _ = predict_activity_from_video(video_path)

            if predicted_activity is not None:
                activity_detection_method = "video_classifier"

                if "walking" in predicted_activity.lower():
                    is_gait = True
        except Exception as e:
            logger.warning("Activity classifier failed: {}".format(e))

    with open(parameters_yaml, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if weights_opt2 is not None:
        pass
    else:
        act_lower = (predicted_activity or "").lower()

        if "treadmill" in act_lower:
            weights_opt2 = params["weights_opt2_treadmill"]
            filter_freq = params["filter_freq"]["walking"]
        elif is_gait:
            weights_opt2 = params["weights_opt2_walking"]
            filter_freq = params["filter_freq"]["walking"]

            if predicted_activity is None:
                predicted_activity = "walking"
                activity_detection_method = "ankle_velocity_heuristic"
        elif "squat" in act_lower:
            weights_opt2 = params["weights_opt2_squats"]
            filter_freq = params["filter_freq"]["squats"]
        elif "sit-to-stand" in act_lower or "sts" in act_lower:
            weights_opt2 = params["weights_opt2_sts"]
            filter_freq = params["filter_freq"]["STS"]
        else:
            weights_opt2 = params["weights_opt2_other"]
            filter_freq = params["filter_freq"]["other"]

            if predicted_activity is None:
                predicted_activity = "other"
                activity_detection_method = "fallback_other"

    output_paths["predicted_activity"] = predicted_activity
    output_paths["activity_detection_method"] = activity_detection_method

    try:
        optimizer_ext = OptimizeExtrinsics(
            r_world_to_cam,
            t_world_to_cam,
            key2d,
            key3d_op_smpl,
            intrinsics,
            height=height_m,
            smpl_model=smpl_model,
            beta=beta,
            static_cam=static_cam,
            iterations=10,
            printer=False,
        )

        output_opt1 = optimizer_ext.optimize()

        if (
            not torch.isfinite(output_opt1["t"]).all()
            or not torch.isfinite(output_opt1["R"]).all()
        ):
            output_opt1["t"] = t_world_to_cam[0]
            output_opt1["R"] = r_world_to_cam[0]

        t_world_to_cam_opt1 = output_opt1["t"].repeat(n_frames, 1)
        r_world_to_cam_opt1 = output_opt1["R"].unsqueeze(0).repeat(n_frames, 1, 1)
        beta = output_opt1["beta"]

        key2d_smpl_opt1 = ut.reproject(
            intrinsics,
            r_world_to_cam_opt1,
            t_world_to_cam_opt1,
            key3d_op_smpl,
        )

        has_contact = (
            "contact" in wham_result
            and torch.is_tensor(wham_result["contact"])
            and wham_result["contact"].abs().sum() > 0
        )

        if not has_contact:
            weights_opt2 = dict(weights_opt2)
            weights_opt2["contact_velocity"] = 0
            weights_opt2["contact_position"] = 0
            weights_opt2["flat_floor"] = 0
            weights_opt2["stability"] = 0

        optimizer_pose = OptimizePose(
            r_world_to_cam_opt1,
            t_world_to_cam_opt1,
            wham_result["pose_world"][:, :3],
            wham_result["trans_world"],
            intrinsics,
            key2d,
            smpl_model,
            wham_result["pose_world"][:, 3:],
            beta,
            wham_result["contact"],
            frame_rate,
            optimize_camera=True,
            print_loss_terms=print_loss_terms,
            iterations=n_iter_opt2,
            weights=weights_opt2,
            cutoff_frequency=filter_freq,
            smoothness_diff_n=smoothness_diff_n,
            output_dir=output_path,
            video_path=trimmed_video_path if trimmed_video_path is not None else video_path,
            frame_ids=frame_ids,
            create_contact_visualizations=False,
        )

        output_opt2 = optimizer_pose.optimize()

        t_root_in_world = output_opt2["t_root_in_world"]
        r_root_in_world = output_opt2["r_root_in_world"]
        t_world_to_cam = output_opt2["t_world_to_cam"].squeeze(0)
        r_world_to_cam = output_opt2["r_world_to_cam"].squeeze(0)
        body_pose = output_opt2["body_pose"]
        beta = output_opt2["beta"]

        if predicted_activity is not None and "treadmill" in predicted_activity.lower():
            n_t = t_root_in_world.shape[0]
            t_lin = torch.linspace(0, 1, n_t, device=t_root_in_world.device)

            for axis in [0, 1, 2]:
                vals = t_root_in_world[:, axis]
                t_mean = t_lin.mean()
                v_mean = vals.mean()
                a = ((t_lin - t_mean) * (vals - v_mean)).sum() / (
                    (t_lin - t_mean) ** 2
                ).sum()
                b = v_mean - a * t_mean
                t_root_in_world[:, axis] = vals - (a * t_lin + b) + vals[0]

    except Exception as e:
        logger.error("Optimization failed: {}".format(e))
        logger.error(traceback.format_exc())

        if device == "cuda":
            torch.set_default_tensor_type("torch.FloatTensor")

        return output_paths

    if save_smpl_for_viz:
        try:
            key3d_world = output_opt2["key_3d"].squeeze(0)
            r_world_to_cam_opt = r_world_to_cam
            t_world_to_cam_opt = t_world_to_cam

            key3d_cam_T = torch.einsum(
                "...ij,...kj->...ik",
                r_world_to_cam_opt,
                key3d_world,
            )

            key3d_cam = key3d_cam_T.transpose(-1, -2) + t_world_to_cam_opt.unsqueeze(1)

            kp3d_path = os.path.join(output_path, "{}_keypoints_3d_cam.pkl".format(trial_name))
            with open(kp3d_path, "wb") as f:
                pickle.dump(key3d_cam.cpu().numpy(), f)

            output_paths["keypoints_3d_cam_pkl"] = kp3d_path

            pose_world_combined = torch.cat(
                [
                    r_root_in_world.reshape(len(r_root_in_world), -1)[:, :3],
                    body_pose.reshape(len(body_pose), -1),
                ],
                dim=1,
            )

            vertices_list = []
            faces_np = None

            for i_frame in range(len(t_root_in_world)):
                smpl_res = ut.pred_smpl(
                    smpl_model,
                    trans=t_root_in_world[i_frame : i_frame + 1].reshape((1, 1, 3)),
                    root_orient=pose_world_combined[i_frame, :3].reshape((1, 1, 3)),
                    body_pose=pose_world_combined[i_frame, 3:].reshape((1, 1, -1)),
                    betas=beta.reshape((1, len(beta))),
                )

                if "verts3d_all" in smpl_res:
                    verts = smpl_res["verts3d_all"]

                    while verts.dim() > 2:
                        verts = verts.squeeze(0)

                    vertices_list.append(verts)

                    if faces_np is None and "faces" in smpl_res:
                        faces_np = smpl_res["faces"]

            vertices_world = None

            if len(vertices_list) == len(t_root_in_world):
                vertices_world = torch.stack(vertices_list, dim=0)

                verts_cam_T = torch.einsum(
                    "...ij,...kj->...ik",
                    r_world_to_cam_opt,
                    vertices_world,
                )

                vertices_cam = (
                    verts_cam_T.transpose(-1, -2)
                    + t_world_to_cam_opt.unsqueeze(1)
                )

                verts_path = os.path.join(
                    output_path,
                    "{}_vertices_3d_cam.pkl".format(trial_name),
                )

                with open(verts_path, "wb") as f:
                    pickle.dump(vertices_cam.cpu().numpy(), f)

                output_paths["vertices_3d_cam_pkl"] = verts_path

            n_out = len(t_root_in_world)
            beta_np = beta.detach().cpu().numpy()
            betas_exp = np.tile(beta_np, (n_out, 1))
            pose_combined_np = pose_world_combined.detach().cpu().numpy()

            if pose_combined_np.shape[1] != 72:
                if pose_combined_np.shape[1] > 72:
                    pose_combined_np = pose_combined_np[:, :72]
                else:
                    pad = 72 - pose_combined_np.shape[1]
                    pose_combined_np = np.pad(pose_combined_np, ((0, 0), (0, pad)))

            verts_np = (
                vertices_world.detach().cpu().numpy()
                if vertices_world is not None
                else None
            )

            joints_np = (
                output_opt2["key_3d"].detach().cpu().numpy().squeeze()
                if "key_3d" in output_opt2
                else None
            )

            wham_fmt = {
                "0": {
                    "pose_world": pose_combined_np,
                    "pose": pose_combined_np,
                    "trans_world": t_root_in_world.detach().cpu().numpy(),
                    "trans": t_root_in_world.detach().cpu().numpy(),
                    "betas": betas_exp,
                    "verts": verts_np,
                    "joints": joints_np,
                    "faces": faces_np,
                    "frame_ids": np.arange(n_out),
                    "n_frames": n_out,
                    "trial_name": trial_name,
                    "optimization_type": "standalone_optimized",
                    "cam_R": r_world_to_cam.detach().cpu().numpy(),
                    "cam_T": t_world_to_cam.detach().cpu().numpy(),
                }
            }

            opt_pkl_path = os.path.join(
                output_path,
                "{}_optimized.pkl".format(trial_name),
            )

            joblib.dump(wham_fmt, opt_pkl_path)
            output_paths["optimized_pkl"] = opt_pkl_path

        except Exception as e:
            logger.error("Error saving viz outputs: {}".format(e))
            logger.error(traceback.format_exc())

    if plotting:
        try:
            from utils.utils_vis import (
                plot_objective_function,
                plot_2d_keypoints_interactive_plotly,
            )

            plot_obj_path = plot_objective_function(
                output_opt2,
                show=False,
                save_path=output_path,
            )

            output_paths["plot_objective"] = plot_obj_path

            key2d_smpl_opt2 = ut.reproject(
                intrinsics,
                r_world_to_cam,
                t_world_to_cam,
                output_opt2["key_3d"].squeeze(),
            )

            key2d_plot = {
                "image": key2d.cpu().numpy(),
                "opt1": key2d_smpl_opt1.cpu().squeeze(0).numpy(),
                "opt2": key2d_smpl_opt2.detach().cpu().squeeze(0).numpy(),
            }

            fig_2d_path = plot_2d_keypoints_interactive_plotly(
                key2d_plot,
                save_path=output_path,
                range_mono=opencap_mono_frame_range_video_ref,
            )

            output_paths["plot_2d"] = fig_2d_path

        except Exception as e:
            logger.warning("Plotting failed: {}".format(e))

    if device == "cuda":
        torch.set_default_tensor_type("torch.FloatTensor")

    for k, v in output_paths.items():
        if v is not None:
            logger.info("{}: {}".format(k, v))

    return output_paths