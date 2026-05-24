from pathlib import Path
import numpy as np
import torch
import smplx
from keypoints_map import get_smpl_joint_map

SMPL_JOINT_MAP = get_smpl_joint_map()


def create_smpl_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("SMPL model not found: {}".format(model_path))
    return smplx.create(
        str(model_path),
        model_type="smpl",
        batch_size=1,
        use_pca=False,
        flat_hand_mean=True,
        gender="neutral",
    ).eval()


def get_3d_joints_for_frame(model, person_data, frame_idx):
    pose = person_data["pose"][frame_idx:frame_idx + 1]
    trans = person_data["trans"][frame_idx:frame_idx + 1]
    betas = person_data["betas"]
    curr_betas = betas[frame_idx:frame_idx + 1] if len(betas) == len(person_data["pose"]) else betas

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

    joints_3d = output.joints[0].cpu().numpy()
    result = {}
    for joint_name, joint_idx in SMPL_JOINT_MAP.items():
        x, y, z = joints_3d[joint_idx]
        result[joint_name] = [round(float(x), 5), round(float(y), 5), round(float(z), 5)]
    return result
