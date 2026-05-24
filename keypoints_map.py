from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_KEYPOINTS_MAP_PATH = Path("configs/keypoints_map.yml")


@lru_cache(maxsize=4)
def _load_smpl_joint_map(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError("Keypoints map file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    smpl_joint_map = data.get("smpl_joint_map")
    if not isinstance(smpl_joint_map, dict) or not smpl_joint_map:
        raise ValueError("Missing or invalid 'smpl_joint_map' in {}".format(path))
    out = {}
    for key, value in smpl_joint_map.items():
        out[str(key)] = int(value)
    return out


def get_smpl_joint_map(path: str = None) -> dict:
    target = Path(path) if path else DEFAULT_KEYPOINTS_MAP_PATH
    return dict(_load_smpl_joint_map(str(target.resolve())))
