from __future__ import annotations

import os
import shlex
import subprocess
import shutil
from glob import glob
from pathlib import Path
from os.path import join
from typing import Dict, Optional, Tuple

from preprocess_pipeline.config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMPL_MODEL_PATH,
)
from preprocess_pipeline.calib import export_camera_jsons, resolve_selected_offset_from_camera_profile
from preprocess_pipeline.extract_2d import export_tracking_2d_to_camera_profiles
from preprocess_pipeline.logs import log_module_done, log_module_start, log_offset_selected
from preprocess_pipeline.offset_paper import (
    compute_offset_from_pkls as compute_paper_offset,
    load_pkl_data as load_offset_pkl_data,
)

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


def _parse_optional_int(value):
    if value is None or value == "":
        return None
    return int(value)


def _resolve_selected_frame_window(config: dict, synced_total_frames: int) -> tuple[int, int]:
    preprocess_cfg = config.get("preprocess", {})
    start_frame = _parse_optional_int(preprocess_cfg.get("start_frame"))
    end_frame = _parse_optional_int(preprocess_cfg.get("end_frame"))

    if start_frame is None:
        start_frame = 1
    if end_frame is None:
        end_frame = synced_total_frames

    start_frame = max(1, int(start_frame))
    end_frame = min(int(end_frame), int(synced_total_frames))
    if end_frame < start_frame:
        raise ValueError(
            f"Invalid preprocess frame range: start_frame={start_frame}, end_frame={end_frame}, synced_total_frames={synced_total_frames}"
        )
    return start_frame, end_frame


def _compute_synced_frame_count(pkl1: str, pkl2: str, offset: int) -> int:
    cam1 = load_offset_pkl_data(str(pkl1))
    cam2 = load_offset_pkl_data(str(pkl2))
    cam1_start = max(0, -int(offset))
    cam2_start = max(0, int(offset))
    return max(0, min(len(cam1["pose"]) - cam1_start, len(cam2["pose"]) - cam2_start))


def _clean_preprocess_output(output_dir: str, video_paths, clean_images: bool = True) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_json in out_dir.glob("data_cam*.json"):
        old_json.unlink()
    for old_pickle in out_dir.glob("cameraIntrinsics_*.pickle"):
        old_pickle.unlink()
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
    frame_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    start_number: int = 1,
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
        frame_range = frame_ranges.get(video_basename) if frame_ranges else None
        if frame_range is not None:
            start_idx, end_idx = frame_range
            vf = f"select='between(n\\,{int(start_idx)}\\,{int(end_idx)})'"
            cmd = (
                f'{ffmpeg} -i {shlex.quote(videoname)} -vf "{vf}" '
                f"-vsync 0 -q:v 1 -start_number {int(start_number)} {output_pattern}"
            )
        else:
            cmd = (
                f'{ffmpeg} -i {shlex.quote(videoname)} -vf "fps={fps_str}" '
                f"-q:v 1 -start_number {int(start_number)} {output_pattern}"
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


def run_offset_estimation(pkl1: str, pkl2: str, smpl_model_path: str, j_regressor_path: str, map_path: str) -> int:
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
        max_frames=100,
    )

    print(f"[Preprocess] Offset={offset}")
    return int(offset)


def run_preprocess(config: dict, extract_frames: bool = True) -> tuple[int, str]:
    input_dir = config.get("preprocess", {}).get("input_dir", DEFAULT_INPUT_DIR)
    output_dir = config.get("preprocess", {}).get("output_dir", DEFAULT_OUTPUT_DIR)
    smpl_model_path = config.get("preprocess", {}).get("smpl_model_path", DEFAULT_SMPL_MODEL_PATH)

    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    j_regressor_path = config.get("paths", {}).get("j_regressor_3d", "models/J_regressor_body25_plus_palm27.npy")
    cam1_pkl = config["paths"]["cam1_pkl"]
    cam2_pkl = config["paths"]["cam2_pkl"]
    cam1_video = config["paths"]["camera1_video"]
    cam2_video = config["paths"]["camera2_video"]
    if config.get("runtime", {}).get("clean_output", True):
        _clean_preprocess_output(output_dir, [cam1_video, cam2_video], clean_images=extract_frames)

    log_module_start(input_dir=input_dir, output_dir=output_dir)
    if extract_frames and config.get("runtime", {}).get("clean_output", True):
        _clean_extracted_images(output_dir)
    offset = run_offset_estimation(
        pkl1=cam1_pkl,
        pkl2=cam2_pkl,
        smpl_model_path=smpl_model_path,
        j_regressor_path=j_regressor_path,
        map_path=config["paths"]["keypoints3d_map"],
    )
    synced_total_frames = _compute_synced_frame_count(cam1_pkl, cam2_pkl, offset)
    selected_start_frame, selected_end_frame = _resolve_selected_frame_window(config, synced_total_frames)
    cam1_start = max(0, -int(offset))
    cam2_start = max(0, int(offset))
    cam1_source_start = cam1_start + selected_start_frame - 1
    cam1_source_end = cam1_start + selected_end_frame - 1
    cam2_source_start = cam2_start + selected_start_frame - 1
    cam2_source_end = cam2_start + selected_end_frame - 1

    if extract_frames:
        extract_images(
            video_paths=[cam1_video, cam2_video],
            output_folder=output_dir,
            frame_ranges={
                Path(cam1_video).stem: (cam1_source_start, cam1_source_end),
                Path(cam2_video).stem: (cam2_source_start, cam2_source_end),
            },
            start_number=selected_start_frame,
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            restart=False,
            debug=False,
        )
    export_camera_jsons(config, offset=offset)
    export_tracking_2d_to_camera_profiles(config)

    selected_offset, selected_method, selected_path = resolve_selected_offset_from_camera_profile(config)
    config.setdefault("runtime", {})["selected_offset"] = selected_offset
    config["runtime"]["offset_method"] = selected_method
    config["runtime"]["selected_frame_start"] = selected_start_frame
    config["runtime"]["selected_frame_end"] = selected_end_frame
    config["runtime"]["selected_source_frame_indices"] = {
        "camera1": cam1_source_start,
        "camera2": cam2_source_start,
    }

    log_offset_selected(selected_method, selected_offset, str(selected_path))
    log_module_done(output_dir=output_dir)
    return selected_offset, selected_method
