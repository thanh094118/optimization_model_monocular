from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np

from json_io import read_json, write_json

TRACKING_RESULTS_MAP = {
    "neck": 0,
    "right_shoulder": 6,
    "right_elbow": 8,
    "right_hand": 10,
    "left_shoulder": 5,
    "left_elbow": 7,
    "left_hand": 9,
    "right_hip": 12,
    "right_knee": 14,
    "right_ankle": 16,
    "left_hip": 11,
    "left_knee": 13,
    "left_ankle": 15,
    "right_foot": 21,
    "left_foot": 18,
}


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _extract_person_payload(wham_data):
    if isinstance(wham_data, Mapping):
        if 0 in wham_data:
            return wham_data[0]
        if "0" in wham_data:
            return wham_data["0"]
        for value in wham_data.values():
            if isinstance(value, Mapping):
                return value
    if isinstance(wham_data, list):
        for value in wham_data:
            if isinstance(value, Mapping):
                return value
    return None


def _load_tracking_payload(pkl_path: Path) -> dict | None:
    if not pkl_path.exists():
        print(f"[Preprocess] 2D skip: PKL not found: {pkl_path}")
        return None

    person = _extract_person_payload(joblib.load(pkl_path))
    if person is None:
        print(f"[Preprocess] 2D skip: cannot find person payload in {pkl_path}")
        return None

    tracking = person.get("tracking_results_for_reproj")
    if not isinstance(tracking, Mapping):
        print(f"[Preprocess] 2D skip: tracking_results_for_reproj missing in {pkl_path}")
        return None
    return dict(tracking)


def _build_2d_camera_payload(tracking: dict) -> tuple[dict, list | None]:
    if "keypoints" not in tracking or "frame_id" not in tracking:
        missing = [k for k in ("frame_id", "keypoints") if k not in tracking]
        raise KeyError(f"Missing tracking key(s): {missing}")

    frame_ids = _to_numpy(tracking["frame_id"]).astype(int).reshape(-1)
    keypoints = _to_numpy(tracking["keypoints"]).astype(float)
    if keypoints.ndim != 3 or keypoints.shape[2] < 3:
        raise ValueError(f"Expected keypoints shape (frames, joints, 3+), got {keypoints.shape}")
    if keypoints.shape[0] != frame_ids.shape[0]:
        raise ValueError(f"frame_id/keypoints frame mismatch: {frame_ids.shape[0]} vs {keypoints.shape[0]}")

    max_index = max(TRACKING_RESULTS_MAP.values())
    if keypoints.shape[1] <= max_index:
        raise ValueError(f"keypoints has {keypoints.shape[1]} joints, need index {max_index}")

    by_frame = {}
    for row_idx, frame_id in enumerate(frame_ids):
        by_frame[str(int(frame_id))] = {
            joint_name: keypoints[row_idx, src_idx, :3].tolist()
            for joint_name, src_idx in TRACKING_RESULTS_MAP.items()
        }

    init_betas = tracking.get("init_betas")
    shape = _to_numpy(init_betas).astype(float).tolist() if init_betas is not None else None
    payload = {
        "tracking_results_map": dict(TRACKING_RESULTS_MAP),
        "frame_ids": frame_ids.tolist(),
        "keypoints": by_frame,
    }
    return payload, shape


def export_tracking_2d_to_camera_profiles(config: dict) -> None:
    output_dir = Path(config.get("preprocess", {}).get("output_dir", "output/preprocess_results"))
    paths = config.get("paths", {})
    pkl_by_cam = {
        "cam1": Path(paths.get("cam1_pkl", "")),
        "cam2": Path(paths.get("cam2_pkl", "")),
    }

    for cam_id, pkl_path in pkl_by_cam.items():
        profile_path = output_dir / f"data_{cam_id}.json"
        if not profile_path.exists():
            print(f"[Preprocess] 2D skip: camera profile not found: {profile_path}")
            continue

        tracking = _load_tracking_payload(pkl_path)
        if tracking is None:
            continue

        try:
            keypoints_2d, shape = _build_2d_camera_payload(tracking)
        except Exception as exc:
            print(f"[Preprocess] 2D skip: {pkl_path.name}: {exc}")
            continue

        profile = read_json(profile_path)
        profile[f"2D_camera_{cam_id}"] = keypoints_2d
        if shape is not None:
            profile["shape"] = shape
            profile["init_betas"] = shape
        write_json(profile_path, profile)
        print(f"[Preprocess] Exported 2D keypoints and shape to {profile_path.name}")
