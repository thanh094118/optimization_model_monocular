from __future__ import annotations

import os
import shlex
import subprocess
from glob import glob
from pathlib import Path
from os.path import join

from preprocess_pipeline.config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMPL_MODEL_PATH,
)
from preprocess_pipeline.calib import export_camera_jsons, resolve_selected_offset_from_camera_profile
from preprocess_pipeline.extract_2d import export_tracking_2d_to_camera_profiles
from preprocess_pipeline.logs import log_module_done, log_module_start, log_offset_selected
from preprocess_pipeline.offset_paper import compute_offset_from_pkls as compute_paper_offset

EXTENSIONS = [".mp4", ".MP4"]


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


def extract_images(input_folder: str, output_folder: str, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", restart: bool = False, debug: bool = False) -> None:
    videos = sorted(sum([glob(join(input_folder, "*" + ext)) for ext in EXTENSIONS], []))

    if not videos:
        print(f"Không tìm thấy video .mp4 trong folder: {input_folder}")
        return

    print(f"Tìm thấy {len(videos)} video(s)")
    for videoname in videos:
        video_basename = os.path.splitext(os.path.basename(videoname))[0]
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

        cmd = (
            f'{ffmpeg} -i {shlex.quote(videoname)} -vf "fps={fps_str}" '
            f'-q:v 1 -start_number 1 {shlex.quote(join(outpath, "images_frame_%d.jpg"))}'
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


def run_offset_estimation(input_folder: str, smpl_model_path: str) -> int:
    pkl_files = sorted(glob(join(input_folder, "*.pkl")))
    if len(pkl_files) < 2:
        print(f"Bỏ qua offset: cần ít nhất 2 file .pkl trong {input_folder}")
        return 0

    pkl1, pkl2 = pkl_files[0], pkl_files[1]
    print(f"[Preprocess] Offset input | cam1={os.path.basename(pkl1)} | cam2={os.path.basename(pkl2)}")

    paper_offset = compute_paper_offset(pkl1, pkl2, smpl_model_path, verbose=False)

    print(f"[Preprocess] Offset paper={paper_offset}")
    return int(paper_offset)


def run_preprocess(config: dict) -> tuple[int, str]:
    input_dir = config.get("preprocess", {}).get("input_dir", DEFAULT_INPUT_DIR)
    output_dir = config.get("preprocess", {}).get("output_dir", DEFAULT_OUTPUT_DIR)
    smpl_model_path = config.get("preprocess", {}).get("smpl_model_path", DEFAULT_SMPL_MODEL_PATH)

    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_module_start(input_dir=input_dir, output_dir=output_dir)
    extract_images(
        input_folder=input_dir,
        output_folder=output_dir,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        restart=False,
        debug=False,
    )
    offset_paper = run_offset_estimation(
        input_folder=input_dir,
        smpl_model_path=smpl_model_path,
    )
    export_camera_jsons(config, offset_paper=offset_paper)
    export_tracking_2d_to_camera_profiles(config)

    selected_offset, selected_method, selected_path = resolve_selected_offset_from_camera_profile(config)
    config.setdefault("runtime", {})["selected_offset"] = selected_offset
    config["runtime"]["offset_method"] = selected_method

    log_offset_selected(selected_method, selected_offset, str(selected_path))
    log_module_done(output_dir=output_dir)
    return selected_offset, selected_method
