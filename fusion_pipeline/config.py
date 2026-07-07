OUTPUT_SUBDIRS = ("keypoints3d", "metadata")

HEIGHT = 1.63
RIGID_BONES_RATIO = {
    ("left_elbow", "left_shoulder"): 0.186,
    ("left_wrist", "left_elbow"): 0.146,
    ("right_elbow", "right_shoulder"): 0.186,
    ("right_wrist", "right_elbow"): 0.146,
    ("left_knee", "left_hip"): 0.245,
    ("left_ankle", "left_knee"): 0.246,
    ("right_knee", "right_hip"): 0.245,
    ("right_ankle", "right_knee"): 0.246,
}

ROTATION_PARENT_JOINTS = {
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "left_toe": "left_ankle",
    "left_foot": "left_ankle",
    "right_toe": "right_ankle",
    "right_foot": "right_ankle",
}

NON_REPLACEABLE_ANCHORS = {"left_hip", "left_shoulder", "right_hip", "right_shoulder"}

OCCLUSION_NEIGHBORS = 10
TORSO_PART_IDS = {0, 3, 6, 9, 13, 14}
OCCLUSION_CHECK_JOINTS = {
    "left_elbow",
    "left_wrist",
    "left_hand",
    "left_toe",
    "left_foot",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "right_toe",
    "right_foot",
    "left_knee",
    "left_ankle",
    "right_knee",
    "right_ankle",
}

ORIENTATION_EPSILON = 1e-2
HARMONIC_EPSILON = 1e-6
CONFIDENCE_DELTA_CAP = 0.05

DEFAULT_OCCLUDED_FACTOR = 0.25
HUBER_DELTA = 0.05
BONE_LENGTH_MIN_SCALE = 0.85
BONE_LENGTH_MAX_SCALE = 1.15
