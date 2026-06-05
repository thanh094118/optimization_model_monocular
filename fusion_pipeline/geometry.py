import numpy as np

ROTATION_PARENT_JOINTS = {
    "left_hand": "left_shoulder",
    "right_hand": "right_shoulder",
    "left_wrist": "left_shoulder",
    "right_wrist": "right_shoulder",
    "left_shoulder": "left_hip",
    "right_shoulder": "right_hip",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
}


def _as_xyz(point):
    arr = np.asarray(point, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Expected shape (3,), got {}".format(arr.shape))
    return arr


def get_orientation_flag(joints, epsilon=1e-2):
    required = ("right_shoulder", "left_shoulder", "right_hip", "left_hip")
    if not all(name in joints for name in required):
        return {name: 0 for name in joints}
    rs = _as_xyz(joints["right_shoulder"])
    ls = _as_xyz(joints["left_shoulder"])
    rh = _as_xyz(joints["right_hip"])
    lh = _as_xyz(joints["left_hip"])
    mid_shoulders = (rs + ls) / 2.0
    mid_hips = (rh + lh) / 2.0
    v_lr = rs - ls
    v_spine = mid_shoulders - mid_hips
    forward_vec = np.cross(v_lr, v_spine)
    norm = float(np.linalg.norm(forward_vec))
    if norm < 1e-8:
        return {name: 0 for name in joints}
    forward_vec /= norm
    flags = {}
    for name, pos in joints.items():
        parent = ROTATION_PARENT_JOINTS.get(name)
        if parent is None or parent not in joints:
            flags[name] = 0
            continue
        dot = float(np.dot(_as_xyz(pos) - _as_xyz(joints[parent]), forward_vec))
        flags[name] = 0 if abs(dot) < epsilon else (1 if dot > 0 else -1)
    return flags
