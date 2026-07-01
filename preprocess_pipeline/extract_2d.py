from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import yaml

from keypoints_map import load_keypoints2d_map
import json_io

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


def _load_tracking_payload(pkl_path: Path) -> Optional[dict]:
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


def _load_keypoints2d_map(map_path: Path) -> dict[str, object]:
    return load_keypoints2d_map(str(map_path))


def _resolve_keypoint_spec(spec, source_keypoints: np.ndarray) -> list[float]:
    if isinstance(spec, int):
        return source_keypoints[spec, :3].tolist()
    if isinstance(spec, Mapping) and "average" in spec:
        pts = [source_keypoints[int(idx), :3] for idx in spec["average"]]
        return np.mean(pts, axis=0).tolist()
    raise ValueError(f"Unsupported spec: {spec}")


def _build_2d_camera_payload(tracking: dict, keypoints2d_map: dict[str, object]) -> tuple[dict, Optional[list]]:
    if "keypoints" not in tracking or "frame_id" not in tracking:
        missing = [k for k in ("frame_id", "keypoints") if k not in tracking]
        raise KeyError(f"Missing tracking key(s): {missing}")

    frame_ids = _to_numpy(tracking["frame_id"]).astype(int).reshape(-1)
    keypoints = _to_numpy(tracking["keypoints"]).astype(float)
    if keypoints.ndim != 3 or keypoints.shape[2] < 3:
        raise ValueError(f"Expected keypoints shape (frames, joints, 3+), got {keypoints.shape}")
    if keypoints.shape[0] != frame_ids.shape[0]:
        raise ValueError(f"frame_id/keypoints frame mismatch: {frame_ids.shape[0]} vs {keypoints.shape[0]}")

    by_frame = {}
    for row_idx, frame_id in enumerate(frame_ids):
        source_keypoints = keypoints[row_idx]
        resolved_keypoints: dict[str, list[float]] = {}
        for joint_name, spec in keypoints2d_map.items():
            resolved_keypoints[joint_name] = _resolve_keypoint_spec(spec, source_keypoints)
            
        if len(resolved_keypoints) != 21:
            raise ValueError(f"Expected 21 keys, got {len(resolved_keypoints)}")
            
        by_frame[str(int(frame_id))] = resolved_keypoints

    payload = {
        "frame_ids": frame_ids.tolist(),
        "keypoints": by_frame,
    }
    return payload, None


def export_tracking_2d_to_camera_profiles(config: dict) -> None:
    output_dir = Path(config.get("preprocess", {}).get("output_dir", "output/preprocess_results"))
    paths = config.get("paths", {})
    pkl_by_cam = {
        "cam1": Path(paths.get("cam1_pkl", "")),
        "cam2": Path(paths.get("cam2_pkl", "")),
    }
    keypoints2d_map = _load_keypoints2d_map(Path(paths["keypoints2d_map"]))

    for cam_id, pkl_path in pkl_by_cam.items():
        profile_path = output_dir / f"data_{cam_id}.json"
        if not profile_path.exists():
            print(f"[Preprocess] 2D skip: camera profile not found: {profile_path}")
            continue

        tracking = _load_tracking_payload(pkl_path)
        if tracking is None:
            continue

        try:
            keypoints_2d, _ = _build_2d_camera_payload(tracking, keypoints2d_map)
        except Exception as exc:
            print(f"[Preprocess] 2D skip: {pkl_path.name}: {exc}")
            continue

        profile = json_io.read_json(profile_path)
        profile[f"2D_camera_{cam_id}"] = keypoints_2d
        json_io.write_json(profile_path, profile)
        print(f"[Preprocess] Exported 2D keypoints to {profile_path.name}")
