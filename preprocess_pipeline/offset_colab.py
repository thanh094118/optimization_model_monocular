import json
import numpy as np
from scipy import stats
from pathlib import Path
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
import smplx
import joblib
import copy

# ----------------------------------------------------------------------
# CẤU HÌNH
# ----------------------------------------------------------------------
DETAIL_PRINT = True

ORIGINAL_JOINT_NAMES = [
    "neck", "right_shoulder", "right_elbow", "right_hand",
    "left_shoulder", "left_elbow", "left_hand",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "left_hip", "left_knee", "left_ankle", "left_foot"
]

SMPL_JOINT_MAP = {
    "neck": 12, "right_shoulder": 17, "right_elbow": 19, "right_hand": 21,
    "left_shoulder": 16, "left_elbow": 18, "left_hand": 20,
    "right_hip": 2, "right_knee": 5, "right_ankle": 8,
    "left_hip": 1, "left_knee": 4, "left_ankle": 7,
    "right_foot": 11, "left_foot": 10,
}

# ----------------------------------------------------------------------
# 1. XỬ LÝ JSON (từ folder)
# ----------------------------------------------------------------------
def load_joints_from_json(folder_path, camera_name):
    """Đọc các file pose_data_*.json, trả về (T, 16, 3) với pelvis ảo ở đầu."""
    json_files = sorted(Path(folder_path).glob("pose_data_*.json"))
    if not json_files:
        raise ValueError(f"Không tìm thấy file pose_data_*.json trong {folder_path}")
    joints_list = []
    for fpath in json_files:
        with open(fpath, 'r') as f:
            data = json.load(f)
        cam_data = data.get(camera_name, {})
        orig = []
        for jname in ORIGINAL_JOINT_NAMES:
            if jname in cam_data:
                orig.append(np.array(cam_data[jname], dtype=np.float32))
            else:
                orig.append(np.zeros(3))
        orig = np.stack(orig)  # (15,3)
        right_hip = orig[7]
        left_hip = orig[11]
        pelvis = (right_hip + left_hip) / 2.0
        all_joints = np.vstack([pelvis, orig])  # (16,3)
        joints_list.append(all_joints)
    return np.stack(joints_list, axis=0)  # (T,16,3)

# ----------------------------------------------------------------------
# 2. XỬ LÝ PKL (dùng SMPL)
# ----------------------------------------------------------------------
def load_pkl_data(file_path):
    data = joblib.load(file_path)
    if isinstance(data, dict) or "defaultdict" in str(type(data)):
        if 0 in data:
            person_data = data[0]
        elif "0" in data:
            person_data = data["0"]
        else:
            person_data = next((v for v in data.values() if isinstance(v, dict)), None)
    elif isinstance(data, list):
        person_data = next((v for v in data if isinstance(v, dict)), None)
    else:
        person_data = None
    if person_data is None:
        raise ValueError(f"Không đọc được payload từ {file_path}.")
    required_keys = ['pose', 'trans', 'betas']
    for key in required_keys:
        if key not in person_data:
            raise KeyError(f"Thiếu key '{key}' trong {file_path}.")
    return person_data

def get_all_joints_from_smpl(model, person_data):
    pose = np.asarray(person_data['pose'], dtype=np.float32)
    trans = np.asarray(person_data['trans'], dtype=np.float32)
    betas = np.asarray(person_data['betas'], dtype=np.float32)
    T = pose.shape[0]
    if betas.ndim == 1:
        betas = np.tile(betas, (T, 1))
    pose_t = torch.tensor(np.ascontiguousarray(pose), dtype=torch.float32)
    trans_t = torch.tensor(np.ascontiguousarray(trans), dtype=torch.float32)
    betas_t = torch.tensor(np.ascontiguousarray(betas), dtype=torch.float32)
    global_orient = pose_t[:, :3]
    body_pose = pose_t[:, 3:]
    with torch.no_grad():
        output = model(betas=betas_t, global_orient=global_orient,
                       body_pose=body_pose, transl=trans_t)
    joints_all = output.joints.cpu().numpy()  # (T,24,3)
    result = np.zeros((T, len(ORIGINAL_JOINT_NAMES), 3), dtype=np.float32)
    for i, name in enumerate(ORIGINAL_JOINT_NAMES):
        smpl_idx = SMPL_JOINT_MAP[name]
        result[:, i, :] = joints_all[:, smpl_idx, :]
    # Thêm pelvis ảo (giống JSON)
    right_hip = result[:, 7, :]
    left_hip  = result[:, 11, :]
    pelvis = (right_hip + left_hip) / 2.0
    return np.concatenate([pelvis.reshape(T,1,3), result], axis=1)  # (T,16,3)

# ----------------------------------------------------------------------
# 3. CHUẨN HÓA PA-MPJPE (chung cho cả hai loại)
# ----------------------------------------------------------------------
def normalize_pa_mpjpe(seq_with_pelvis):
    """
    seq_with_pelvis: (T, 16, 3) - khớp 0 là pelvis, 1..15 là các khớp gốc.
    Trả về (T, 16, 3) đã tịnh tiến (pelvis=0) và xoay theo hướng vai.
    """
    T = seq_with_pelvis.shape[0]
    pelvis = seq_with_pelvis[:, 0, :]          # (T,3)
    orig = seq_with_pelvis[:, 1:, :]           # (T,15,3)
    centered = orig - pelvis.reshape(T,1,3)    # (T,15,3)
    # Vai phải (index 1), vai trái (index 4) trong 15 khớp gốc
    right_shoulder = centered[:, 1, :]
    left_shoulder  = centered[:, 4, :]
    vec = right_shoulder - left_shoulder
    angle = np.arctan2(vec[:, 2], vec[:, 0])
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = np.zeros_like(centered)
    for t in range(T):
        x = centered[t, :, 0]
        z = centered[t, :, 2]
        rotated[t, :, 0] = x * cos_a[t] + z * sin_a[t]
        rotated[t, :, 1] = centered[t, :, 1]
        rotated[t, :, 2] = -x * sin_a[t] + z * cos_a[t]
    pelvis_zero = np.zeros((T, 1, 3), dtype=np.float32)
    return np.concatenate([pelvis_zero, rotated], axis=1)  # (T,16,3)

# ----------------------------------------------------------------------
# 4. PROCRUSTES & TÌM OFFSET
# ----------------------------------------------------------------------
def procrustes_alignment(source, target):
    mu_s = np.mean(source, axis=0)
    mu_t = np.mean(target, axis=0)
    src_centered = source - mu_s
    tgt_centered = target - mu_t
    ss = np.sum(src_centered**2)
    st = np.sum(tgt_centered**2)
    scale = 1.0 if ss == 0 else np.sqrt(st / ss)
    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    source_aligned = scale * (src_centered @ R) + mu_t
    return source_aligned

def frame_distance_procrustes(frame1, frame2):
    frame2_aligned = procrustes_alignment(frame2, frame1)
    diff = frame1 - frame2_aligned
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def find_time_offset_procrustes(seq1, seq2):
    n1, n2 = len(seq1), len(seq2)
    offsets = []
    for i in range(n2):
        frame2 = seq2[i]
        distances = [frame_distance_procrustes(seq1[j], frame2) for j in range(n1)]
        best_j = np.argmin(distances)
        offsets.append(best_j - i)
        if DETAIL_PRINT and (i+1) % 50 == 0:
            print(f" -> Đã đối chiếu {i+1}/{n2} frames...")
    delta_t = int(stats.mode(offsets, keepdims=True).mode[0])
    return delta_t, offsets


def compute_offset_from_pkls(pkl1_path, pkl2_path, smpl_path, detail_print=False):
    """Compute frame offset between two PKL files using Procrustes method."""
    global DETAIL_PRINT
    old_detail = DETAIL_PRINT
    DETAIL_PRINT = detail_print
    try:
        model = smplx.create(
            smpl_path,
            model_type='smpl',
            batch_size=1,
            use_pca=False,
            flat_hand_mean=True,
            gender='neutral',
        ).eval()

        cam1_data = load_pkl_data(str(pkl1_path))
        cam2_data = load_pkl_data(str(pkl2_path))
        seq1_raw = get_all_joints_from_smpl(model, cam1_data)
        seq2_raw = get_all_joints_from_smpl(model, cam2_data)
        seq1 = normalize_pa_mpjpe(seq1_raw)
        seq2 = normalize_pa_mpjpe(seq2_raw)
        delta_t, _ = find_time_offset_procrustes(seq1, seq2)
        return int(delta_t)
    finally:
        DETAIL_PRINT = old_detail

# ----------------------------------------------------------------------
# 5. MAIN (tự động phát hiện input)
# ----------------------------------------------------------------------
def main():
    input_path = input("Nhập đường dẫn (folder chứa JSON hoặc file PKL): ").strip()
    path = Path(input_path)
    
    # Xác định loại input
    if path.is_dir():
        # Xử lý JSON
        print("Phát hiện thư mục JSON.")
        camera1, camera2 = "camera1", "camera2"
        print("Đang tải camera1...")
        seq1_raw = load_joints_from_json(str(path), camera1)
        print("Đang tải camera2...")
        seq2_raw = load_joints_from_json(str(path), camera2)
    elif path.is_file() and path.suffix == '.pkl':
        # Xử lý PKL
        print("Phát hiện file PKL. Yêu cầu thêm file PKL thứ hai và SMPL model.")
        pkl2 = input("Nhập đường dẫn file PKL camera 2: ").strip()
        smpl_path = input("Nhập đường dẫn file SMPL_NEUTRAL.pkl: ").strip()
        print("Đang khởi tạo SMPL model...")
        model = smplx.create(smpl_path, model_type='smpl', batch_size=1,
                             use_pca=False, flat_hand_mean=True, gender='neutral').eval()
        print("Đang xử lý camera 1...")
        cam1_data = load_pkl_data(str(path))
        seq1_raw = get_all_joints_from_smpl(model, cam1_data)
        print("Đang xử lý camera 2...")
        cam2_data = load_pkl_data(pkl2)
        seq2_raw = get_all_joints_from_smpl(model, cam2_data)
    else:
        raise ValueError("Đường dẫn không hợp lệ: phải là thư mục JSON hoặc file .pkl")
    
    print(f"Camera1: {seq1_raw.shape[0]} frames")
    print(f"Camera2: {seq2_raw.shape[0]} frames")
    
    print("Đang chuẩn hóa PA-MPJPE...")
    seq1 = normalize_pa_mpjpe(seq1_raw)
    seq2 = normalize_pa_mpjpe(seq2_raw)
    
    print("Tìm offset bằng Procrustes...")
    delta_t, all_offsets = find_time_offset_procrustes(seq1, seq2)
    
    print("--- Kết quả ---")
    print(f"Delta t (mode offset): {delta_t} frames")
    if DETAIL_PRINT:
        unique, counts = np.unique(all_offsets, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Offset {u:3d}: {c:4d} lần")
    
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    out_file = f"delta_t_procrustes_{now.strftime('%y%m%d_%H%M')}.txt"
    with open(out_file, 'w') as f:
        f.write(str(delta_t))
    print(f"Đã lưu vào {out_file}")

if __name__ == "__main__":
    main()
