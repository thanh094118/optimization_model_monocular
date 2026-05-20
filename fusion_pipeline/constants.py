RANSAC_THRESHOLD = 0.05
RANSAC_MAX_COMBOS = 500
HEIGHT = 1.63
DIST_CONF_REF = 2.0

RIGID_BONES_RATIO = {
    ("left_elbow", "left_shoulder"): 0.186,
    ("left_hand", "left_elbow"): 0.146,
    ("right_elbow", "right_shoulder"): 0.186,
    ("right_hand", "right_elbow"): 0.146,
    ("left_knee", "left_hip"): 0.245,
    ("left_ankle", "left_knee"): 0.246,
    ("right_knee", "right_hip"): 0.245,
    ("right_ankle", "right_knee"): 0.246,
}

W, H = 2048, 2048
fx = fy = (W * W + H * H) ** 0.5
cx, cy = W / 2.0, H / 2.0
N_NEIGHBORS = 10
TORSO_PART_IDS = {0, 3, 6, 9, 13, 14}

OCCLUSION_CHECK_JOINTS = {
    "left_elbow", "left_hand",
    "right_elbow", "right_hand",
    "left_knee", "left_ankle",
    "right_knee", "right_ankle",
}
