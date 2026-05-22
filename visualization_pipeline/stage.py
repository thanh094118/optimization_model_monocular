# -*- coding: utf-8 -*-
"""Visualization phase: compare Fusion optimized pose, source video, and Learnable-SMPLify output."""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import matplotlib
import numpy as np

# Use non-interactive backend so the phase works from normal Python CLI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


FULL_SKELETON = [
    ("right_shoulder", "left_shoulder"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_hip", "left_hip"),
    ("neck", "right_shoulder"),
    ("neck", "left_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
]


def _frame_index(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if not match:
        raise ValueError(f"Cannot extract frame index from {path.name}")
    return int(match.group())


def _clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_video in output_dir.glob("compare_*.mp4"):
        old_video.unlink()
    for old_video in output_dir.glob("project_*.mp4"):
        old_video.unlink()


def load_optimized_from_fused(fused_dir: Path) -> list[dict]:
    """Load optimized poses from fused_data_*.json files."""
    if not fused_dir.exists():
        raise FileNotFoundError(f"Fused JSON directory not found: {fused_dir}")

    keypoints_dir = fused_dir / "keypoints3d"
    metadata_dir = fused_dir / "metadata"
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Fused keypoints directory not found: {keypoints_dir}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Fused metadata directory not found: {metadata_dir}")
    file_paths = sorted(keypoints_dir.glob("fused_data_*.json"), key=_frame_index)
    results = []
    for path in file_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        metadata_path = metadata_dir / path.name
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                data.update(json.load(f))
        if "optimized" in data:
            results.append(data["optimized"])
        else:
            results.append({
                "camera1": data.get("camera1", {}),
                "camera2": data.get("camera2", {}),
            })

    print(f"[Visualization] Loaded {len(results)} optimized frames from {fused_dir}")
    return results


def load_learnable_from_dir(learnable_dir: Path) -> list[dict]:
    """Load final / learnable_smplify poses from learnable_frame_*.json files."""
    if not learnable_dir.exists():
        raise FileNotFoundError(f"Learnable output directory not found: {learnable_dir}")

    keypoints_dir = learnable_dir / "keypoints3d"
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Learnable keypoints directory not found: {keypoints_dir}")

    file_paths = sorted(keypoints_dir.glob("learnable_frame_*.json"), key=_frame_index)
    results = []
    for path in file_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        results.append({
            "camera1": data.get("camera1", {}),
            "camera2": data.get("camera2", {}),
        })

    print(f"[Visualization] Loaded {len(results)} learnable frames from {learnable_dir}")
    return results


def _resolve_video_path(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    return str(path) if path.exists() else None


def get_video_map(config: dict) -> dict[str, str]:
    """Resolve camera videos from YAML. If omitted, fallback to MP4 files in video_search_dir/current dir."""
    vis_cfg = config.get("visualization", {})
    paths_cfg = config.get("paths", {})

    video_map = {}
    cam1_video = _resolve_video_path(paths_cfg.get("camera1_video"))
    cam2_video = _resolve_video_path(paths_cfg.get("camera2_video"))

    if cam1_video:
        video_map["camera1"] = cam1_video
    if cam2_video:
        video_map["camera2"] = cam2_video

    if video_map:
        print(f"[Visualization] Video map from config: {video_map}")
        return video_map

    search_dir = Path(vis_cfg.get("video_search_dir", "."))
    mp4_files = sorted(
        glob.glob(str(search_dir / "*.mp4")),
        key=lambda x: int(re.search(r"(\d+)\.mp4$", x).group(1))
        if re.search(r"(\d+)\.mp4$", x)
        else float("inf"),
    )

    if len(mp4_files) >= 2:
        video_map["camera1"] = mp4_files[0]
        video_map["camera2"] = mp4_files[1]
    elif len(mp4_files) == 1:
        video_map["camera1"] = mp4_files[0]
        video_map["camera2"] = mp4_files[0]

    if video_map:
        print(f"[Visualization] Video map auto-detected: {video_map}")
    else:
        print("[Visualization] No MP4 video found. Video column will be blank.")
    return video_map


def as_render_point(point):
    """Convert camera coordinate to plot coordinate: X_plot=X, Y_plot=Z, Z_plot=-Y."""
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return np.array([arr[0], arr[2], -arr[1]], dtype=float)


def plot_skeleton_safe(ax, pts, color, label):
    if not isinstance(pts, dict):
        return []

    drawn_points = []
    used_label = False

    for bone_a, bone_b in FULL_SKELETON:
        if bone_a not in pts or bone_b not in pts:
            continue
        p1 = as_render_point(pts[bone_a])
        p2 = as_render_point(pts[bone_b])
        if p1 is None or p2 is None:
            continue
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color,
            alpha=0.75,
            linewidth=2,
            label=label if not used_label else None,
        )
        used_label = True
        drawn_points.extend([p1, p2])

    scatter_points = []
    for pos_raw in pts.values():
        p = as_render_point(pos_raw)
        if p is not None:
            scatter_points.append(p)
    if scatter_points:
        arr = np.vstack(scatter_points)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color=color, s=30)
        drawn_points.extend(scatter_points)

    return drawn_points


def setup_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")
    ax.view_init(elev=20, azim=-45)


def set_auto_axes(ax, points, margin: float = 0.2) -> None:
    if not points:
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([1.0, 3.0])
        ax.set_zlim([-1.0, 1.0])
        return

    pts = np.vstack(points)
    center = pts.mean(axis=0)
    radius = max(float(np.ptp(pts, axis=0).max()) * 0.55 + margin, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _read_video_frame(cap, frame_idx: int):
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _normalize_key(k: str) -> str:
    aliases = {
        "RShoulder": "right_shoulder", "LShoulder": "left_shoulder",
        "RElbow": "right_elbow", "LElbow": "left_elbow",
        "RWrist": "right_wrist", "LWrist": "left_wrist",
        "RHip": "right_hip", "LHip": "left_hip",
        "RKnee": "right_knee", "LKnee": "left_knee",
        "RAnkle": "right_ankle", "LAnkle": "left_ankle",
        "Neck": "neck", "MidHip": "pelvis",
        "RHeel": "right_foot", "LHeel": "left_foot",
    }
    return aliases.get(k, k)


def _project_point(xyz, fx, fy, cx, cy):
    arr = np.asarray(xyz, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    x, y, z = arr[:3]
    if not np.isfinite([x, y, z]).all() or z <= 1e-6:
        return None
    u = int(round(fx * x / z + cx))
    v = int(round(fy * y / z + cy))
    return (u, v)


def _load_camera_keypoints_by_frame(directory: Path, pattern: str, camera_name: str) -> Dict[int, dict]:
    files = sorted(directory.glob(pattern), key=_frame_index)
    frame_to_kp = {}
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cam_data = data.get("optimized", {}).get(camera_name) if "optimized" in data else data.get(camera_name)
        if isinstance(cam_data, dict):
            frame_to_kp[_frame_index(p)] = {_normalize_key(k): v for k, v in cam_data.items()}
    return frame_to_kp


def _resolve_image_dir(paths: dict, camera_name: str) -> Path:
    cam_idx = "1" if camera_name == "camera1" else "2"
    candidates = [
        Path(f"output/image_video_{cam_idx}"),
        Path(f"output/images_video_{cam_idx}"),
        Path(f"output/images_cam{cam_idx}"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find image folder for {camera_name}. Tried: {candidates}")


def create_project2d_animation(camera_name: str, paths: dict, output_video: Path, target_fps: int, dpi: int = 100) -> None:
    pose_map = _load_camera_keypoints_by_frame(Path(paths["pose_output_dir"]) / "keypoints3d", "pose_data_*.json", camera_name)
    fusion_map = _load_camera_keypoints_by_frame(Path(paths["fused_output_dir"]) / "keypoints3d", "fused_data_*.json", camera_name)
    learn_map = _load_camera_keypoints_by_frame(Path(paths["learnable_output_dir"]) / "keypoints3d", "learnable_frame_*.json", camera_name)
    image_dir = _resolve_image_dir(paths, camera_name)
    image_map = {_frame_index(p): p for p in sorted(image_dir.glob("images_frame_*.jpg"), key=_frame_index)}
    if not image_map:
        image_map = {_frame_index(p): p for p in sorted(image_dir.glob("*.jpg"), key=_frame_index)}

    common_ids = sorted(set(pose_map) & set(fusion_map) & set(learn_map) & set(image_map))
    common_ids = [i for i in common_ids if i >= 1]
    if not common_ids:
        raise ValueError(f"No common frame ids for {camera_name} in project2d flow.")

    first_img = cv2.imread(str(image_map[common_ids[0]]))
    h, w = first_img.shape[:2]
    fx = fy = float((w * w + h * h) ** 0.5)
    cx, cy = w / 2.0, h / 2.0

    frame_interval = 1000 / max(target_fps, 1)
    fig = plt.figure(figsize=(24, 8))
    axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
    titles = ["Pose 3D->2D", "Fusion 3D->2D", "Learnable 3D->2D"]
    modules = [pose_map, fusion_map, learn_map]

    def draw_overlay(img_bgr, joints):
        out = img_bgr.copy()
        proj = {}
        for name, xyz in joints.items():
            p = _project_point(xyz, fx, fy, cx, cy)
            if p is not None:
                proj[name] = p
        for a, b in FULL_SKELETON:
            if a in proj and b in proj:
                cv2.line(out, proj[a], proj[b], (200, 200, 200), 2, cv2.LINE_AA)
        for p in proj.values():
            cv2.circle(out, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def update(idx):
        frame_id = common_ids[idx]
        base = cv2.imread(str(image_map[frame_id]))
        for ax, title, data_map in zip(axes, titles, modules):
            ax.cla()
            ax.imshow(draw_overlay(base, data_map[frame_id]))
            ax.set_title(f"{title} | {camera_name} | Frame {frame_id}")
            ax.axis("off")

    animation = FuncAnimation(fig, update, frames=len(common_ids), interval=frame_interval, repeat=False)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(output_video), writer="ffmpeg", fps=target_fps, dpi=dpi)
    plt.close(fig)
    print(f"[Visualization] Saved project video: {output_video}")


def create_comparison_animation(
    camera_name: str,
    optimized_poses,
    learnable_poses,
    video_map: dict[str, str],
    frame_count: int,
    output_video: Path,
    target_fps: int,
    dpi: int = 100,
) -> None:
    """Create 3-column animation: Optimized | Video | Learnable."""
    video_path = video_map.get(camera_name)
    cap = cv2.VideoCapture(video_path) if video_path and os.path.exists(video_path) else None
    if cap is None:
        print(f"[Visualization] No video for {camera_name}. Middle column will be blank.")

    frame_interval = 1000 / max(target_fps, 1)
    fig = plt.figure(figsize=(24, 8))
    ax_opt = fig.add_subplot(131, projection="3d")
    ax_vid = fig.add_subplot(132)
    ax_learn = fig.add_subplot(133, projection="3d")

    def update(frame_idx):
        ax_opt.cla()
        ax_vid.cla()
        ax_learn.cla()

        opt_points = []
        if frame_idx < len(optimized_poses):
            opt_pose = optimized_poses[frame_idx].get(camera_name)
            opt_points = plot_skeleton_safe(ax_opt, opt_pose, "purple", f"{camera_name} optimized")
        setup_axis(ax_opt, f"Optimized {camera_name} - Frame {frame_idx + 1}/{frame_count}")

        frame_rgb = _read_video_frame(cap, frame_idx)
        if frame_rgb is not None:
            ax_vid.imshow(frame_rgb)
            ax_vid.set_title(f"Video {camera_name} - Frame {frame_idx + 1}")
        else:
            ax_vid.text(0.5, 0.5, "No video frame", ha="center", va="center", transform=ax_vid.transAxes)
            ax_vid.set_title(f"Video {camera_name}")
        ax_vid.axis("off")

        learn_points = []
        if frame_idx < len(learnable_poses):
            learn_pose = learnable_poses[frame_idx].get(camera_name)
            learn_points = plot_skeleton_safe(ax_learn, learn_pose, "tab:blue", f"{camera_name} learnable")
        setup_axis(ax_learn, f"After Learnable {camera_name} - Frame {frame_idx + 1}/{frame_count}")

        combined = opt_points + learn_points
        set_auto_axes(ax_opt, combined)
        set_auto_axes(ax_learn, combined)

        for ax in (ax_opt, ax_learn):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper left")

    animation = FuncAnimation(fig, update, frames=frame_count, interval=frame_interval, repeat=False)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(output_video), writer="ffmpeg", fps=target_fps, dpi=dpi)
    plt.close(fig)

    if cap is not None:
        cap.release()

    print(f"[Visualization] Saved video: {output_video}")


def run_visualization(config: dict) -> None:
    runtime_cfg = config.get("runtime", {})
    paths = config.get("paths", {})
    vis_cfg = config.get("visualization", {})

    if not vis_cfg.get("enabled", True):
        print("[Visualization] Disabled by config: visualization.enabled=false")
        return

    fused_dir = Path(paths["fused_output_dir"])
    learnable_dir = Path(paths["learnable_output_dir"])
    output_dir = Path(paths.get("visualization_output_dir", "output/visualization"))

    if runtime_cfg.get("clean_output", True):
        _clean_output(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[Visualization] Optimized vs Video vs Learnable")
    print("=" * 60)

    optimized_poses = load_optimized_from_fused(fused_dir)
    learnable_poses = load_learnable_from_dir(learnable_dir)
    frame_count = min(len(optimized_poses), len(learnable_poses))
    if frame_count == 0:
        raise ValueError("No frames to visualize. Check fused_jsons and learnable_results.")

    max_frames = vis_cfg.get("max_frames")
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))

    optimized_poses = optimized_poses[:frame_count]
    learnable_poses = learnable_poses[:frame_count]

    target_fps = int(vis_cfg.get("target_fps", 10))
    dpi = int(vis_cfg.get("dpi", 100))
    cameras = vis_cfg.get("cameras", ["camera1", "camera2"])
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    video_map = get_video_map(config)

    print(f"[Visualization] Optimized frames: {len(optimized_poses)}")
    print(f"[Visualization] Learnable frames: {len(learnable_poses)}")
    print(f"[Visualization] Rendering frames: {frame_count}")
    print(f"[Visualization] FPS: {target_fps}")
    print(f"[Visualization] Output dir: {output_dir}")

    for camera_name in cameras:
        print(f"[Visualization] Processing {camera_name}")
        output_video = output_dir / f"compare_{camera_name}_opt_vs_video_vs_learnable_{timestamp}.mp4"
        create_comparison_animation(
            camera_name=camera_name,
            optimized_poses=optimized_poses,
            learnable_poses=learnable_poses,
            video_map=video_map,
            frame_count=frame_count,
            output_video=output_video,
            target_fps=target_fps,
            dpi=dpi,
        )

    for camera_name in cameras:
        output_video = output_dir / f"project_{camera_name}_pose_fusion_learnable_{timestamp}.mp4"
        create_project2d_animation(
            camera_name=camera_name,
            paths=paths,
            output_video=output_video,
            target_fps=target_fps,
            dpi=dpi,
        )

    print(f"[Visualization] Done. Output: {output_dir}")
