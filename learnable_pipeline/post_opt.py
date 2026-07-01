import numpy as np
import torch

def post_optimize_smpl_sequence(
    pose_init: np.ndarray,
    trans_init: np.ndarray,
    betas_np: np.ndarray,
    fusion_frames: list[dict],
    net,
    j_regressor_3d_path: str,
    map_data: dict,
    device: torch.device,
    iters: int = 80,
    lr: float = 0.01
) -> tuple[np.ndarray, np.ndarray, float, float]:
    
    n_frames = pose_init.shape[0]
    
    pose_var = torch.tensor(pose_init, dtype=torch.float32, device=device, requires_grad=True)
    trans_var = torch.tensor(trans_init, dtype=torch.float32, device=device, requires_grad=True)
    betas_t = torch.tensor(betas_np, dtype=torch.float32, device=device)
    
    # Load 21-joint regressor
    j_reg = np.load(j_regressor_3d_path)
    if j_reg.shape != (27, 6890):
        raise ValueError(f"Expected (27, 6890), got {j_reg.shape}")
    j_reg_t = torch.tensor(j_reg, dtype=torch.float32, device=device)
    
    row_indices = [kp["regressor_index"] for kp in map_data["keypoints"]]
    target_names = [kp["name"] for kp in map_data["keypoints"]]
    
    target_joints = np.zeros((n_frames, 21, 3), dtype=np.float32)
    vis_mask = np.zeros((n_frames, 21), dtype=np.float32)
    
    for i, frame in enumerate(fusion_frames):
        for j, name in enumerate(target_names):
            if name in frame:
                target_joints[i, j] = frame[name]
                vis_mask[i, j] = 1.0
                
    target_joints_t = torch.tensor(target_joints, dtype=torch.float32, device=device)
    vis_mask_t = torch.tensor(vis_mask, dtype=torch.float32, device=device)
    
    pose_anchor = torch.tensor(pose_init, dtype=torch.float32, device=device)
    trans_anchor = torch.tensor(trans_init, dtype=torch.float32, device=device)
    
    optimizer = torch.optim.Adam([pose_var, trans_var], lr=lr)
    smpl_layer = net.human_model.layer["neutral"]
    
    best_loss = float('inf')
    best_pose = pose_init.copy()
    best_trans = trans_init.copy()
    initial_loss = 0.0
    
    for step in range(iters):
        optimizer.zero_grad()
        
        root_orient = pose_var[:, :3]
        body_pose = pose_var[:, 3:72]
        
        smpl_out = smpl_layer(
            betas=betas_t[:, :10],
            global_orient=root_orient,
            body_pose=body_pose
        )
        
        # Regress 27 keys then select 21
        joints_all = torch.einsum("jv, bvc -> bjc", j_reg_t, smpl_out.vertices)
        joints_21 = joints_all[:, row_indices, :]
        joints_world = joints_21 + trans_var.unsqueeze(1)
        
        loss_joint = torch.mean(vis_mask_t * torch.sum((joints_world - target_joints_t) ** 2, dim=-1))
        loss_pose = torch.mean(torch.sum((pose_var - pose_anchor) ** 2, dim=-1))
        loss_trans = torch.mean(torch.sum((trans_var - trans_anchor) ** 2, dim=-1))
        
        # Temporal smooth on pose and trans
        if n_frames > 1:
            loss_temp_pose = torch.mean(torch.sum((pose_var[1:] - pose_var[:-1]) ** 2, dim=-1))
            loss_temp_trans = torch.mean(torch.sum((trans_var[1:] - trans_var[:-1]) ** 2, dim=-1))
            loss_temp = loss_temp_pose + loss_temp_trans
        else:
            loss_temp = torch.tensor(0.0, device=device)
            
        loss = 1.0 * loss_joint + 0.01 * loss_pose + 0.01 * loss_trans + 0.1 * loss_temp
        
        if step == 0:
            initial_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        if not np.isfinite(current_loss):
            raise ValueError("Non-finite loss during post-optimization")
            
        if current_loss < best_loss:
            best_loss = current_loss
            best_pose = pose_var.detach().cpu().numpy()
            best_trans = trans_var.detach().cpu().numpy()
            
    return best_pose, best_trans, initial_loss, best_loss
