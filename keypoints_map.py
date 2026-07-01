import json
from functools import lru_cache
from pathlib import Path
import yaml

EXPECTED_NAMES = [
    "head", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "pelvis",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "left_toe", "left_foot", "right_toe", "right_foot",
    "left_hand", "right_hand"
]

@lru_cache(maxsize=1)
def load_keypoints3d_map(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Keypoints map file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    keypoints = data.get("keypoints", [])
    if len(keypoints) != 21:
        raise ValueError(f"Expected exactly 21 keypoints, got {len(keypoints)}")
    
    names = []
    regressor_indices = set()
    for kp in keypoints:
        name = kp.get("name")
        idx = kp.get("regressor_index")
        if name in names:
            raise ValueError(f"Duplicate name: {name}")
        if idx in regressor_indices:
            raise ValueError(f"Duplicate regressor index: {idx}")
        if name in ["Eye", "Ear", "Heel"] or "eye" in name.lower() or "ear" in name.lower() or "heel" in name.lower():
            raise ValueError(f"Invalid name found: {name}")
        names.append(name)
        regressor_indices.add(idx)
    
    if names != EXPECTED_NAMES:
        raise ValueError(f"Keypoint names or order do not match expected 21 keys.\nExpected: {EXPECTED_NAMES}\nGot: {names}")
    
    priority1 = data.get("priority1", [])
    if len(priority1) != 13:
        raise ValueError(f"Priority1 must have exactly 13 keys, got {len(priority1)}")
        
    priority2 = data.get("priority2", [])
    if len(priority2) != 21:
        raise ValueError(f"Priority2 must have exactly 21 keys, got {len(priority2)}")
        
    return data

@lru_cache(maxsize=1)
def load_keypoints2d_map(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"2D Keypoints map file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    map_2d = data.get("keypoints2d_map", {})
    if len(map_2d) != 21:
        raise ValueError(f"Expected exactly 21 2D keypoints mappings, got {len(map_2d)}")
        
    for name in map_2d.keys():
        if name not in EXPECTED_NAMES:
            raise ValueError(f"Unexpected key in 2D map: {name}")
            
    return map_2d

@lru_cache(maxsize=1)
def load_mapping_3dto2d(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"3D to 2D mapping file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    if len(mapping) != 21:
        raise ValueError(f"Expected exactly 21 mapping entries, got {len(mapping)}")
    
    for k, v in mapping.items():
        if k not in EXPECTED_NAMES:
            raise ValueError(f"Unexpected key in mapping: {k}")
        if v not in EXPECTED_NAMES:
            raise ValueError(f"Unexpected value in mapping: {v}")
        if k != v:
            raise ValueError(f"Mapping must be identity, got {k} -> {v}")
            
    return mapping
