from __future__ import annotations

from pathlib import Path

import cv2

from json_io import read_json, write_json


def _parse_4x4(values: str) -> list[list[float]]:
    nums = [float(x) for x in values.strip().split()]
    if len(nums) != 16:
        raise ValueError(f"Expected 16 values, got {len(nums)}")
    return [nums[i * 4 : (i + 1) * 4] for i in range(4)]


def _extract_camera_blocks(calib_path: Path) -> dict[str, dict[str, list[list[float]]]]:
    lines = calib_path.read_text(encoding="utf-8").splitlines()
    cameras: dict[str, dict[str, list[list[float]]]] = {}
    current_name: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("name"):
            parts = line.split()
            if len(parts) >= 2:
                current_name = parts[-1]
                cameras[current_name] = {}
            continue
        if current_name is None:
            continue
        if line.startswith("intrinsic"):
            cameras[current_name]["intrinsic"] = _parse_4x4(line[len("intrinsic") :])
        elif line.startswith("extrinsic"):
            cameras[current_name]["extrinsic"] = _parse_4x4(line[len("extrinsic") :])

    return cameras


def export_camera_jsons(config: dict, offset_paper: int, offset_colab: int) -> None:
    calib_cfg = config.get("preprocess", {}).get("calibration", {})
    calib_file = Path(calib_cfg.get("file", "input/camera.calibration"))
    output_dir = Path(calib_cfg.get("output_dir", "output/preprocess_pipeline"))
    mapping = calib_cfg.get("camera_name_map", {"cam1": "2", "cam2": "8"})

    cams = _extract_camera_blocks(calib_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = config.get("paths", {})
    source_by_id = {
        "cam1": str(Path(paths.get("cam1_pkl", ""))),
        "cam2": str(Path(paths.get("cam2_pkl", ""))),
    }
    video_by_id = {
        "cam1": Path(paths.get("camera1_video", "input/video_1.mp4")),
        "cam2": Path(paths.get("camera2_video", "input/video_2.mp4")),
    }

    for cam_id, calib_name in mapping.items():
        key = str(calib_name)
        if key not in cams:
            raise ValueError(f"Camera name {key!r} not found in {calib_file}")

        intr = cams[key]["intrinsic"]
        extr = cams[key]["extrinsic"]
        video_path = video_by_id.get(cam_id)
        if video_path is None or not video_path.exists():
            raise FileNotFoundError(f"Video not found for {cam_id}: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w <= 0 or h <= 0:
            raise ValueError(f"Cannot read video size for {cam_id}: {video_path}")
        fx = fy = float((w * w + h * h) ** 0.5)
        cx = float(w / 2.0)
        cy = float(h / 2.0)
        payload = {
            "camera_id": cam_id,
            "source_pkl": source_by_id.get(cam_id, ""),
            "offset_paper": int(offset_paper),
            "offset_colab": int(offset_colab),
            "intrinsics_cam": [row[:3] for row in intr[:3]],
            "intrinsics_estimation": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            "extrinsic_cam": [row[:3] for row in extr[:3]],
            "xyz": [extr[0][3], extr[1][3], extr[2][3]],
        }
        write_json(output_dir / f"data_{cam_id}.json", payload)


def load_camera_profile(config: dict, cam_id: str) -> dict:
    calib_cfg = config.get("preprocess", {}).get("calibration", {})
    output_dir = Path(calib_cfg.get("output_dir", "output/preprocess_results"))
    path = output_dir / f"data_{cam_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Camera profile not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid camera profile format: {path}")
    return data


def resolve_selected_offset_from_camera_profile(config: dict) -> tuple[int, str, Path]:
    offset_cfg = config.get("preprocess", {}).get("offset", {})
    method = str(offset_cfg.get("method", "paper")).strip().lower()
    if method not in {"paper", "colab"}:
        raise ValueError(f"Invalid preprocess.offset.method={method!r}, expected 'paper' or 'colab'")

    cam_profile = load_camera_profile(config, "cam1")
    key = f"offset_{method}"
    if key not in cam_profile:
        raise KeyError(f"Missing {key!r} in camera profile data_cam1.json")
    return int(cam_profile[key]), method, Path("data_cam1.json")


def resolve_selected_intrinsics(config: dict, cam_id: str) -> list[list[float]]:
    calib_cfg = config.get("preprocess", {}).get("calibration", {})
    source = str(calib_cfg.get("intrinsics_source", "intri_cam")).strip().lower()
    if source not in {"intri_cam", "intri_esti"}:
        raise ValueError("preprocess.calibration.intrinsics_source must be 'intri_cam' or 'intri_esti'")
    profile = load_camera_profile(config, cam_id)
    key = "intrinsics_cam" if source == "intri_cam" else "intrinsics_estimation"
    intr = profile.get(key)
    if not isinstance(intr, list) or len(intr) != 3:
        raise ValueError(f"Invalid {key} in data_{cam_id}.json")
    return intr
