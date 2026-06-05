FUSED_FILE_PREFIX = "fused_data_"
OUTPUT_SUBDIRS = ("keypoints3d", "metadata")
RANSAC_THRESHOLD = 0.05
RANSAC_MAX_COMBOS = 500
HEIGHT = 1.63
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
