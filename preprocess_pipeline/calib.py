from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import cv2

from json_io import read_json, write_json
from config_loader import resolve_inputs


def _extract_person_payload(wham_data):
    if isinstance(wham_data, dict):
        if 0 in wham_data:
            return wham_data[0]
        if "0" in wham_data:
            return wham_data["0"]
        for value in wham_data.values():
            if isinstance(value, dict):
                return value
    if isinstance(wham_data, list):
        for value in wham_data:
            if isinstance(value, dict):
                return value
    return None


def _load_wham_camera_payload(pkl_path: Path) -> dict:
    if not pkl_path.exists():
        raise FileNotFoundError(f"WHAM PKL not found: {pkl_path}")
    person = _extract_person_payload(joblib.load(pkl_path))
    if person is None:
        raise ValueError(f"Cannot extract person payload from WHAM PKL: {pkl_path}")
    return person


def _mean_shape_over_frames(shape_value) -> list[list[float]]:
    shape = np.asarray(shape_value, dtype=np.float32)
    if shape.ndim == 1:
        shape = shape[None, :]
    if shape.ndim != 2:
        raise ValueError(f"Expected betas/shape to have ndim 1 or 2, got {shape.ndim}")
    return shape.mean(axis=0, keepdims=True).astype(np.float32).tolist()


def _estimate_intrinsics(video_path: Path) -> list[list[float]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise ValueError(f"Cannot read video size for {video_path}")

    fx = fy = float((w * w + h * h) ** 0.5)
    cx = float(w / 2.0)
    cy = float(h / 2.0)
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def export_camera_jsons(config: dict, offset: int) -> None:
    output_dir = Path(config.get("preprocess", {}).get("output_dir", "output/preprocess_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(config)
    source_by_id = {
        "cam1": str(Path(inputs.get("cam1_pkl", ""))),
        "cam2": str(Path(inputs.get("cam2_pkl", ""))),
    }
    video_by_id = {
        "cam1": Path(inputs.get("camera1_video", "input/video_1.mp4")),
        "cam2": Path(inputs.get("camera2_video", "input/video_2.mp4")),
    }
    pkl_by_id = {
        "cam1": Path(inputs.get("cam1_pkl", "")),
        "cam2": Path(inputs.get("cam2_pkl", "")),
    }

    for cam_id in ["cam1", "cam2"]:
        video_path = video_by_id.get(cam_id)
        pkl_path = pkl_by_id.get(cam_id)
        if video_path is None or not video_path.exists():
            print(f"[Preprocess] Video not found for {cam_id}: {video_path}")
            continue

        wham_payload = None
        if pkl_path is not None and pkl_path.exists():
            try:
                wham_payload = _load_wham_camera_payload(pkl_path)
            except Exception as exc:
                print(f"[Preprocess] PKL skip for {cam_id}: {exc}")

        payload = {
            "camera_id": cam_id,
            "source_pkl": source_by_id.get(cam_id, ""),
            "offset": int(offset),
            "intrinsics_source": "intri_esti",
            "intrinsics_estimation": _estimate_intrinsics(video_path),
        }

        if wham_payload is not None and "betas" in wham_payload:
            payload["betas"] = _mean_shape_over_frames(wham_payload["betas"])

        write_json(output_dir / f"data_{cam_id}.json", payload)


def load_camera_profile(config: dict, cam_id: str) -> dict:
    output_dir = Path(config.get("preprocess", {}).get("output_dir", "output/preprocess_results"))
    path = output_dir / f"data_{cam_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Camera profile not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid camera profile format: {path}")
    return data


def resolve_selected_offset_from_camera_profile(config: dict) -> tuple[int, Path]:
    cam_profile = load_camera_profile(config, "cam1")
    if "offset" in cam_profile:
        return int(cam_profile["offset"]), Path("data_cam1.json")
    raise KeyError("Missing 'offset' in camera profile data_cam1.json")


def resolve_selected_intrinsics(config: dict, cam_id: str) -> list[list[float]]:
    profile = load_camera_profile(config, cam_id)
    return profile["intrinsics_estimation"]
