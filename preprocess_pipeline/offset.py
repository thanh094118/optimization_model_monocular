import json
import os
import numpy as np
from scipy import stats
from pathlib import Path

# ----------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------
# Thứ tự các khớp gốc từ JSON (không có pelvis)
ORIGINAL_JOINT_NAMES = [
    "neck", "right_shoulder", "right_elbow", "right_hand",
    "left_shoulder", "left_elbow", "left_hand",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "left_hip", "left_knee", "left_ankle", "left_foot"
]

# Danh sách khớp sau khi thêm pelvis ảo (sẽ ở vị trí đầu)
# Các khớp còn lại giữ nguyên thứ tự
FINAL_JOINT_NAMES = ["pelvis"] + ORIGINAL_JOINT_NAMES

def load_joints_with_pelvis(folder_path, camera_name):
    """
    Đọc tất cả file .json trong folder, thêm khớp pelvis ảo = trung bình của
    right_hip và left_hip. Trả về numpy array (num_frames, num_joints, 3)
    với num_joints = 16 (1 pelvis + 15 gốc)
    """
    json_files = sorted(Path(folder_path).glob("*.json"))
    joints_list = []
    for fpath in json_files:
        with open(fpath, 'r') as f:
            data = json.load(f)
        cam_data = data.get(camera_name, {})
        
        # Lấy tọa độ các khớp gốc theo đúng thứ tự
        orig_joints = []
        for jname in ORIGINAL_JOINT_NAMES:
            if jname in cam_data:
                orig_joints.append(np.array(cam_data[jname], dtype=np.float32))
            else:
                orig_joints.append(np.zeros(3))
        orig_joints = np.stack(orig_joints, axis=0)  # (15,3)
        
        # Tính pelvis ảo = trung bình right_hip (index 7) và left_hip (index 11)
        right_hip = orig_joints[7]   # vị trí của right_hip trong ORIGINAL_JOINT_NAMES
        left_hip  = orig_joints[11]  # left_hip
        pelvis = (right_hip + left_hip) / 2.0
        
        # Ghép pelvis vào đầu
        all_joints = np.vstack([pelvis, orig_joints])  # (16,3)
        joints_list.append(all_joints)
    
    return np.stack(joints_list, axis=0)  # (num_frames, 16, 3)

def align_root(joints, root_idx=0):
    """Dời gốc về khớp root_idx (mặc định 0 - pelvis)"""
    return joints - joints[:, root_idx:root_idx+1, :]

def frame_distance(frame1, frame2):
    """Khoảng cách Euclidean trung bình giữa hai frame"""
    diff = frame1 - frame2
    return np.mean(np.linalg.norm(diff, axis=1))

def find_time_offset(seq1, seq2):
    """
    seq1, seq2: numpy arrays (num_frames, num_joints, 3)
    Trả về delta_t (int) – số frame seq2 trễ hơn seq1.
    """
    n1, n2 = len(seq1), len(seq2)
    seq1_aligned = align_root(seq1)
    seq2_aligned = align_root(seq2)
    
    offsets = []
    for i in range(n2):
        frame2 = seq2_aligned[i]
        distances = [frame_distance(seq1_aligned[j], frame2) for j in range(n1)]
        best_j = np.argmin(distances)
        offsets.append(best_j - i)
    
    delta_t = int(stats.mode(offsets, keepdims=True).mode[0])
    return delta_t

def main():
    folder_path = input("Nhập đường dẫn folder chứa các file JSON: ").strip()
    camera1 = "camera1"
    camera2 = "camera2"
    
    print("Đang đọc dữ liệu camera1...")
    seq1 = load_joints_with_pelvis(folder_path, camera1)
    print(f"  -> {seq1.shape[0]} frames, {seq1.shape[1]} khớp (bao gồm pelvis ảo)")
    
    print("Đang đọc dữ liệu camera2...")
    seq2 = load_joints_with_pelvis(folder_path, camera2)
    print(f"  -> {seq2.shape[0]} frames, {seq2.shape[1]} khớp")
    
    print("Tính toán độ lệch thời gian...")
    delta_t = find_time_offset(seq1, seq2)
    
    output_file = "delta_t.txt"
    with open(output_file, 'w') as f:
        f.write(str(delta_t))
    
    print(f"\n✅ Kết quả: delta_t = {delta_t} (camera2 trễ hơn camera1 {delta_t} frame)")
    print(f"✅ Đã lưu vào {output_file}")

if __name__ == "__main__":
    main()