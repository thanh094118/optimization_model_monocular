import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import torch

# ============================================
# 1. Mapping 15 joints theo BODY25 / OpenPose
# ============================================
BODY25_JOINT_MAP = {
    "neck": 1,
    "right_shoulder": 2,
    "right_elbow": 3,
    "right_hand": 4,
    "left_shoulder": 5,
    "left_elbow": 6,
    "left_hand": 7,
    "right_hip": 9,
    "right_knee": 10,
    "right_ankle": 11,
    "left_hip": 12,
    "left_knee": 13,
    "left_ankle": 14,
    "right_foot": 22,
    "left_foot": 19,
}

# ============================================
# 2. Load J_regressor BODY25 của EasyMocap
# shape: (25, 6890)
# ============================================
J_REGRESSOR_PATH = "J_regressor_body25.npy"


def load_pickle_or_joblib(file_path):
    file_path = Path(file_path)

    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)


def load_vertices(data):
    results = {}

    if isinstance(data, np.ndarray):
        results[0] = data

    elif isinstance(data, dict):
        if "verts" in data and isinstance(data["verts"], np.ndarray):
            results[0] = data["verts"]
        else:
            for pid, content in data.items():
                if isinstance(content, dict) and "verts" in content:
                    results[pid] = content["verts"]
                elif isinstance(content, np.ndarray) and content.shape[-2:] == (6890, 3):
                    results[pid] = content

    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) and "verts" in item:
                results[i] = item["verts"]
            elif isinstance(item, np.ndarray) and item.shape[-2:] == (6890, 3):
                results[i] = item

    else:
        raise TypeError("File khong dung dinh dang")

    if len(results) == 0:
        raise ValueError("Khong tim thay verts")

    return results


def main():
    input_path = input("Nhap duong dan file pkl: ").strip()

    j_full = np.load(J_REGRESSOR_PATH)
    if j_full.shape != (25, 6890):
        raise ValueError(
            f"J_regressor_body25 phai co shape (25, 6890), nhung nhan {j_full.shape}"
        )

    j_regressor = torch.tensor(j_full, dtype=torch.float32)

    data = load_pickle_or_joblib(input_path)
    results = load_vertices(data)

    output_dir = Path("output/1")
    output_dir.mkdir(exist_ok=True, parents=True)

    for pid, verts in results.items():
        verts = torch.tensor(verts, dtype=torch.float32)

        if verts.ndim != 3 or verts.shape[1:] != (6890, 3):
            raise ValueError(
                f"verts cua person {pid} phai co shape (T, 6890, 3), nhung nhan {verts.shape}"
            )

        # BODY25 keypoints3d: (T, 25, 3)
        kp3d = torch.einsum("ji,tik->tjk", j_regressor, verts).cpu().numpy()

        frame_count = kp3d.shape[0]
        print(f"Person {pid}: {frame_count} frames")

        for frame_idx in range(frame_count):
            frame_joints = {}

            for name, idx in BODY25_JOINT_MAP.items():
                coord = kp3d[frame_idx, idx].tolist()
                frame_joints[name] = [round(float(c), 5) for c in coord]

            out_file = output_dir / f"person{pid}_frame{frame_idx:06d}.json"

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"camera1": frame_joints}, f, indent=2)

        print(f"Da xuat {frame_count} file JSON cho person {pid} vao {output_dir}")


if __name__ == "__main__":
    main()