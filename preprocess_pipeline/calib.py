from __future__ import annotations

import joblib
import pickle
from pathlib import Path
import numpy as np
import cv2

from json_io import read_json, write_json

def parse_skeletool_calibration(filepath: Path) -> dict:
    cameras = {}
    current = None
    if not filepath.exists():
        return cameras

    for raw_line in filepath.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Skeletool"):
            continue

        if line.startswith("name"):
            cam_id = line.split()[1]
            current = cam_id
            cameras[cam_id] = {}
            continue

        if current is None:
            continue

        parts = line.split()
        key = parts[0]

        if key == "sensor":
            cameras[current]["sensor"] = (float(parts[1]), float(parts[2]))
        elif key == "size":
            W = int(parts[1])
            H = int(parts[2])
            cameras[current]["W"] = W
            cameras[current]["H"] = H
            cameras[current]["imageSize"] = np.array([[float(H)], [float(W)]])
        elif key == "intrinsic":
            vals = list(map(float, parts[1:17]))
            mat4 = np.array(vals).reshape(4, 4)
            K = mat4[:3, :3]
            cameras[current]["intrinsicMat"] = K
            cameras[current]["fx"] = K[0, 0]
            cameras[current]["fy"] = K[1, 1]
            cameras[current]["cx"] = K[0, 2]
            cameras[current]["cy"] = K[1, 2]
        elif key == "extrinsic":
            vals = list(map(float, parts[1:17]))
            cameras[current]["extrinsicMat"] = np.array(vals).reshape(4, 4)
        elif key == "radial":
            cameras[current]["radial"] = int(parts[1])

    return cameras

def build_intrinsics_dict(K: np.ndarray, W: int, H: int) -> dict:
    if W > H:
        portrait_h = float(W)
        portrait_w = float(H)
    else:
        portrait_h = float(H)
        portrait_w = float(W)

    return {
        "distortion": np.zeros((1, 5), dtype=np.float64),
        "imageSize": np.array([[portrait_h], [portrait_w]]),
        "intrinsicMat": K.copy(),
        "calib_portrait_h": portrait_h,
        "calib_portrait_w": portrait_w,
    }

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


def export_camera_jsons(config: dict, offset: int) -> None:
    output_dir = Path(config.get("preprocess", {}).get("output_dir", "output/preprocess_results"))
    input_dir = Path(config.get("preprocess", {}).get("input_dir", "input"))
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
    pkl_by_id = {
        "cam1": Path(paths.get("cam1_pkl", "")),
        "cam2": Path(paths.get("cam2_pkl", "")),
    }

    calib_files = list(input_dir.glob("camera_*.calibration"))
    def extract_cam_id(filepath: Path) -> int:
        try:
            return int(filepath.stem.split("_")[-1])
        except ValueError:
            return 9999 
    calib_files.sort(key=extract_cam_id)
    
    calib_file_mapping = {}
    if len(calib_files) > 0:
        calib_file_mapping["cam1"] = calib_files[0]
    if len(calib_files) > 1:
        calib_file_mapping["cam2"] = calib_files[1]

    for cam_id in ["cam1", "cam2"]:
        video_path = video_by_id.get(cam_id)
        pkl_path = pkl_by_id.get(cam_id)
        if video_path is None or not video_path.exists():
            print(f"[Preprocess] Video not found for {cam_id}: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w <= 0 or h <= 0:
            raise ValueError(f"Cannot read video size for {cam_id}: {video_path}")

        fx_esti = fy_esti = float((w * w + h * h) ** 0.5)
        cx_esti = float(w / 2.0)
        cy_esti = float(h / 2.0)
        K_esti = np.array([
            [fx_esti, 0.0, cx_esti],
            [0.0, fy_esti, cy_esti],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        calib_file = calib_file_mapping.get(cam_id)
        intrinsics_source = "intri_esti"
        intrinsics_cam = None
        extrinsic_cam = None
        tvec = None
        K_final = K_esti
        W_final = int(w)
        H_final = int(h)

        if calib_file is not None and calib_file.exists():
            cams = parse_skeletool_calibration(calib_file)
            if cams:
                intrinsics_source = "intri_cam"
                cam_data = list(cams.values())[0]
                if "intrinsicMat" in cam_data:
                    K_final = cam_data["intrinsicMat"]
                    intrinsics_cam = K_final.tolist()
                
                if "W" in cam_data and "H" in cam_data:
                    W_final = cam_data["W"]
                    H_final = cam_data["H"]

                if "extrinsicMat" in cam_data:
                    extr = cam_data["extrinsicMat"]
                    extrinsic_cam = extr[:3, :3].tolist()
                    tvec = [float(extr[0, 3]), float(extr[1, 3]), float(extr[2, 3])]

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
            "intrinsics_source": intrinsics_source,
            "intrinsics_estimation": K_esti.tolist(),
        }

        if wham_payload is not None:
            if "trans_cam" in wham_payload:
                payload["trans_cam"] = np.asarray(wham_payload["trans_cam"]).tolist()
            if "betas" in wham_payload:
                mean_betas = _mean_shape_over_frames(wham_payload["betas"])
                payload["betas"] = mean_betas

        if intrinsics_cam is not None:
            payload["intrinsics_cam"] = intrinsics_cam
        if extrinsic_cam is not None:
            payload["extrinsic_cam"] = extrinsic_cam
        if tvec is not None:
            payload["tvec"] = tvec

        write_json(output_dir / f"data_{cam_id}.json", payload)

        pickle_dict = build_intrinsics_dict(K_final, W_final, H_final)
        pickle_path = output_dir / f"cameraIntrinsics_{cam_id}.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(pickle_dict, f)
        print(f"[Preprocess] Exported {pickle_path.name} from {intrinsics_source}")

def load_camera_profile(config: dict, cam_id: str) -> dict:
    output_dir = Path(config.get("preprocess", {}).get("output_dir", "output/preprocess_results"))
    path = output_dir / f"data_{cam_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Camera profile not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid camera profile format: {path}")
    return data

def resolve_selected_offset_from_camera_profile(config: dict) -> tuple[int, str, Path]:
    method = str(config.get("preprocess", {}).get("offset_method", "offset")).strip().lower()
    if method not in {"offset"}:
        raise ValueError(f"Invalid preprocess.offset_method={method!r}, expected 'offset'")

    cam_profile = load_camera_profile(config, "cam1")
    if "offset" in cam_profile:
        return int(cam_profile["offset"]), method, Path("data_cam1.json")
    if "offset_paper" in cam_profile:
        return int(cam_profile["offset_paper"]), method, Path("data_cam1.json")
    raise KeyError("Missing 'offset' in camera profile data_cam1.json")

def resolve_selected_intrinsics(config: dict, cam_id: str) -> list[list[float]]:
    profile = load_camera_profile(config, cam_id)
    source = profile.get("intrinsics_source", "intri_esti")
    key = "intrinsics_cam" if source == "intri_cam" else "intrinsics_estimation"
    intr = profile.get(key)
    if not isinstance(intr, list) or len(intr) != 3:
        raise ValueError(f"Invalid {key} in data_{cam_id}.json")
    return intr
