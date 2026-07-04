from pathlib import Path
import numpy as np
import torch
import smplx
from keypoints_map import load_keypoints3d_map


def create_smpl_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("SMPL model not found: {}".format(model_path))
    return smplx.create(
        str(model_path),
        model_type="smpl",
        batch_size=1,
        gender="neutral",
    ).eval()


def get_3d_joints_for_frame(model, person_data, frame_idx, regressor_path, map_path):
    pose = person_data["pose"][frame_idx:frame_idx + 1]
    trans = person_data["trans"][frame_idx:frame_idx + 1]
    curr_betas = person_data["betas"][frame_idx:frame_idx + 1]

    curr_pose_t = torch.tensor(np.ascontiguousarray(pose), dtype=torch.float32)
    curr_trans_t = torch.tensor(np.ascontiguousarray(trans), dtype=torch.float32)
    curr_betas_t = torch.tensor(np.ascontiguousarray(curr_betas), dtype=torch.float32)

    with torch.no_grad():
        output = model(
            betas=curr_betas_t,
            global_orient=curr_pose_t[:, :3],
            body_pose=curr_pose_t[:, 3:],
            transl=curr_trans_t,
        )

    vertices = output.vertices[0].cpu().numpy()
    j_reg = np.load(regressor_path)
    if j_reg.shape != (27, 6890):
        raise ValueError(f"Expected (27, 6890), got {j_reg.shape}")
        
    joints_all = j_reg @ vertices
    
    map_data = load_keypoints3d_map(map_path)
    result = {}
    for kp in map_data["keypoints"]:
        x, y, z = joints_all[kp["regressor_index"]]
        result[kp["name"]] = [float(x), float(y), float(z)]
    return result
