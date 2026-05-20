import copy
import numpy as np


def _sync_feature_array(person_data):
    pose = np.asarray(person_data["pose"], dtype=np.float32)
    features = pose[:, 3:] if pose.shape[1] > 3 else pose

    trans = np.asarray(person_data.get("trans", []), dtype=np.float32)
    if trans.ndim == 2 and trans.shape[0] == pose.shape[0] and trans.shape[1] >= 3:
        speed = np.zeros((pose.shape[0], 1), dtype=np.float32)
        if pose.shape[0] > 1:
            speed[1:, 0] = np.linalg.norm(np.diff(trans[:, :3], axis=0), axis=1)
        features = np.concatenate([features, speed], axis=1)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    median = np.median(features, axis=0, keepdims=True)
    mad = np.median(np.abs(features - median), axis=0, keepdims=True)
    return (features - median) / np.maximum(mad, 1e-3)


def _offset_score(left_features, right_features, offset, min_overlap=8):
    if offset >= 0:
        overlap = min(len(left_features), len(right_features) - offset)
        left_part = left_features[:overlap]
        right_part = right_features[offset:offset + overlap]
    else:
        left_start = -offset
        overlap = min(len(left_features) - left_start, len(right_features))
        left_part = left_features[left_start:left_start + overlap]
        right_part = right_features[:overlap]

    if overlap < min_overlap:
        return float("inf")
    return float(np.median(np.median(np.abs(left_part - right_part), axis=1)))


def estimate_camera_frame_offset(cam1_data, cam2_data, max_offset=40, min_overlap=8, min_improvement_ratio=0.05):
    left_features = _sync_feature_array(cam1_data)
    right_features = _sync_feature_array(cam2_data)
    left_len = int(left_features.shape[0])
    right_len = int(right_features.shape[0])

    max_offset = max(0, int(max_offset))
    allowed = min(max_offset, max(left_len, right_len) - min_overlap)
    if allowed < 0:
        allowed = 0

    scores = {
        offset: _offset_score(left_features, right_features, offset, min_overlap)
        for offset in range(-allowed, allowed + 1)
    }
    zero_score = scores.get(0, float("inf"))
    best_offset = min(scores, key=lambda key: (scores[key], abs(key)))
    best_score = scores[best_offset]
    improvement = zero_score - best_score
    required = max(abs(zero_score) * float(min_improvement_ratio), 1e-6)
    applied = best_offset != 0 and np.isfinite(best_score) and improvement >= required
    offset = int(best_offset if applied else 0)

    left_start = max(0, -offset)
    right_start = max(0, offset)
    frame_count = max(0, min(left_len - left_start, right_len - right_start))
    return {
        "offset": offset,
        "left_start": int(left_start),
        "right_start": int(right_start),
        "frame_count": int(frame_count),
        "applied": bool(applied),
        "score": float(best_score),
    }


def slice_person_frames(person_data, start, frame_count):
    start = int(start)
    frame_count = int(frame_count)
    total_frames = int(np.asarray(person_data["pose"]).shape[0])
    end = min(total_frames, start + frame_count)
    out = copy.deepcopy(dict(person_data))

    for key, value in list(out.items()):
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_frames:
            out[key] = value[start:end].copy()
    return out
