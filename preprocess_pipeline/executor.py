from __future__ import annotations

import os
import shlex
import subprocess
import shutil
from glob import glob
from pathlib import Path
from os.path import join
from typing import Optional

from preprocess_pipeline.calib import export_camera_jsons, resolve_selected_offset_from_camera_profile
from preprocess_pipeline.extract_2d import export_tracking_2d_to_camera_profiles
from preprocess_pipeline.logs import log_module_done, log_module_start, log_offset_selected
from preprocess_pipeline.offset_paper import (
    compute_offset_from_pkls as compute_paper_offset,
)
from config_loader import resolve_inputs, resolve_preprocess_output_dir

EXTENSIONS = [".mp4", ".MP4", ".avi", ".AVI"]


def _run_cmd(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def _get_video_fps(video_path: str, ffprobe: str = "ffprobe") -> float:
    cmd = (
        f'{ffprobe} -v error -select_streams v:0 -show_entries stream=r_frame_rate '
        f'-of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}'
    )
    raw = subprocess.check_output(cmd, shell=True, text=True).strip()
    if "/" in raw:
        num, den = raw.split("/", 1)
        fps = float(num) / float(den)
    else:
        fps = float(raw)
    if fps <= 0:
        raise ValueError(f"FPS không hợp lệ cho video: {video_path}")
    return fps


def _clean_preprocess_output(output_dir: str, video_paths, clean_images: bool = True) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_json in out_dir.glob("data_cam*.json"):
        old_json.unlink()
    if clean_images:
        for video_path in video_paths:
            video = Path(video_path)
            if not video.exists():
                continue
            image_dir = out_dir / f"images_{video.stem}"
            if image_dir.exists():
                shutil.rmtree(image_dir)


def _clean_extracted_images(output_folder: str) -> None:
    output_path = Path(output_folder)
    if not output_path.exists():
        return
    for image_dir in output_path.glob("images_*"):
        if image_dir.is_dir():
            for old_image in image_dir.glob("images_frame_*.jpg"):
                old_image.unlink()


def extract_images(
    video_paths,
    output_folder: str,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    restart: bool = False,
    debug: bool = False,
) -> None:
    videos = [str(Path(p)) for p in video_paths if p and Path(p).exists()]

    if not videos:
        print("Không tìm thấy video input hợp lệ để extract frame")
        return

    print(f"Tìm thấy {len(videos)} video(s)")
    for videoname in videos:
        video_path = Path(videoname)
        video_basename = video_path.stem
        outpath = join(output_folder, f"images_{video_basename}")
        os.makedirs(outpath, exist_ok=True)
        fps = _get_video_fps(videoname, ffprobe=ffprobe)
        fps_str = f"{fps:.10f}".rstrip("0").rstrip(".")

        existing_frames = glob(join(outpath, "images_frame_*.jpg"))
        if existing_frames and not restart:
            print(
                f"[Preprocess] Extract skip | input={videoname} | output={outpath} | "
                f"fps={fps_str} | frames={len(existing_frames)}"
            )
            continue

        output_pattern = shlex.quote(join(outpath, "images_frame_%d.jpg"))
        cmd = (
            f'{ffmpeg} -i {shlex.quote(videoname)} -vf "fps={fps_str}" '
            f"-q:v 1 {output_pattern}"
        )
        if not debug:
            cmd += " -loglevel error"

        print(
            f"[Preprocess] Extract start | input={videoname} | output={outpath} | fps={fps_str}"
        )
        _run_cmd(cmd)
        frame_count = len(glob(join(outpath, "images_frame_*.jpg")))
        print(
            f"[Preprocess] Extract done | input={videoname} | output={outpath} | "
            f"fps={fps_str} | frames={frame_count}"
        )


def run_offset_estimation(
    pkl1: str,
    pkl2: str,
    smpl_model_path: str,
    j_regressor_path: str,
    map_path: str,
    max_frames: Optional[int] = None,
) -> int:
    if not Path(pkl1).exists() or not Path(pkl2).exists():
        print(f"Bỏ qua offset: không tìm thấy đủ 2 file .pkl\n  cam1={pkl1}\n  cam2={pkl2}")
        return 0

    print(f"[Preprocess] Offset input | cam1={os.path.basename(pkl1)} | cam2={os.path.basename(pkl2)}")
    offset = compute_paper_offset(
        pkl1,
        pkl2,
        smpl_model_path,
        j_regressor_path,
        map_path,
        verbose=False,
        max_frames=max_frames,
    )

    print(f"[Preprocess] Offset={offset}")
    return int(offset)


def run_preprocess(config: dict, extract_frames: bool = True) -> tuple[int, str]:
    output_dir = resolve_preprocess_output_dir(config)
    smpl_model_path = config["paths"]["smpl_model"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    paths = config.get("paths", {})
    inputs = resolve_inputs(config)
    j_regressor_path = paths.get("j_regressor_3d", "models/J_regressor_body25_plus_palm27.npy")
    cam1_pkl = inputs["cam1_pkl"]
    cam2_pkl = inputs["cam2_pkl"]
    cam1_video = inputs["camera1_video"]
    cam2_video = inputs["camera2_video"]
    if config["runtime"]["clean_output"]:
        _clean_preprocess_output(output_dir, [cam1_video, cam2_video], clean_images=extract_frames)

    log_module_start(output_dir=output_dir)
    if extract_frames and config["runtime"]["clean_output"]:
        _clean_extracted_images(output_dir)
    max_frames = config.get("preprocess", {}).get("offset_estimation", {}).get("max_frames")
    if max_frames in (None, ""):
        max_frames = None
    else:
        max_frames = int(max_frames)
    offset = run_offset_estimation(
        pkl1=cam1_pkl,
        pkl2=cam2_pkl,
        smpl_model_path=smpl_model_path,
        j_regressor_path=j_regressor_path,
        map_path=paths["keypoints3d_map"],
        max_frames=max_frames,
    )

    if extract_frames:
        extract_images(
            video_paths=[cam1_video, cam2_video],
            output_folder=output_dir,
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            restart=False,
            debug=False,
        )
    export_camera_jsons(config, offset=offset)
    export_tracking_2d_to_camera_profiles(config)

    selected_offset, selected_path = resolve_selected_offset_from_camera_profile(config)
    config.setdefault("runtime", {})["selected_offset"] = selected_offset

    log_offset_selected(selected_offset, str(selected_path))
    log_module_done(output_dir=output_dir)
    return selected_offset, "offset"
