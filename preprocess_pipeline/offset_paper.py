import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
import torch
import smplx
import joblib


ORIGINAL_JOINT_NAMES = [
    "neck", "right_shoulder", "right_elbow", "right_hand",
    "left_shoulder", "left_elbow", "left_hand",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "left_hip", "left_knee", "left_ankle", "left_foot"
]

SMPL_JOINT_MAP = {
    "neck": 12,
    "right_shoulder": 17,
    "right_elbow": 19,
    "right_hand": 21,
    "left_shoulder": 16,
    "left_elbow": 18,
    "left_hand": 20,
    "right_hip": 2,
    "right_knee": 5,
    "right_ankle": 8,
    "left_hip": 1,
    "left_knee": 4,
    "left_ankle": 7,
    "right_foot": 11,
    "left_foot": 10,
}


def load_pkl_data(file_path):
    """
    Đọc file PKL và trả về dữ liệu của người đầu tiên.
    """
    data = joblib.load(file_path)

    if isinstance(data, dict) or "defaultdict" in str(type(data)):
        if 0 in data:
            person_data = data[0]
        elif "0" in data:
            person_data = data["0"]
        else:
            person_data = next(
                (v for v in data.values() if isinstance(v, dict)),
                None
            )
    elif isinstance(data, list):
        person_data = next((v for v in data if isinstance(v, dict)), None)
    else:
        person_data = None

    if person_data is None:
        raise ValueError(f"Không thể đọc payload từ {file_path}")

    required_keys = ["pose", "betas"]
    for key in required_keys:
        if key not in person_data:
            raise KeyError(f"Thiếu key '{key}' trong {file_path}")

    return person_data


def get_canonical_joints_3d(model, person_data):
    """
    Tạo SMPL canonical joints theo paper:

        J_canon = S(0, body_pose, beta)

    Nghĩa là:
    - global_orient = 0
    - transl = 0
    - chỉ dùng body_pose và betas

    Output:
        joints: shape (T, 15, 3)
    """
    pose = np.asarray(person_data["pose"], dtype=np.float32)       # (T, 72)
    betas = np.asarray(person_data["betas"], dtype=np.float32)     # (T, 10) hoặc (10,)

    T = pose.shape[0]

    if betas.ndim == 1:
        betas = np.tile(betas, (T, 1))

    pose_t = torch.tensor(np.ascontiguousarray(pose), dtype=torch.float32)
    betas_t = torch.tensor(np.ascontiguousarray(betas), dtype=torch.float32)

    global_orient = torch.zeros((T, 3), dtype=torch.float32)
    body_pose = pose_t[:, 3:]
    transl = torch.zeros((T, 3), dtype=torch.float32)

    with torch.no_grad():
        output = model(
            betas=betas_t,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl
        )

    joints_all = output.joints.cpu().numpy()

    result = np.zeros((T, len(ORIGINAL_JOINT_NAMES), 3), dtype=np.float32)

    for i, name in enumerate(ORIGINAL_JOINT_NAMES):
        smpl_idx = SMPL_JOINT_MAP[name]
        result[:, i, :] = joints_all[:, smpl_idx, :]

    return result


def build_canonical_sequence(joints_3d):
    """
    Tạo sequence vector cho DTW.

    Vì joints_3d đã là canonical:
    - không cần root-center
    - không cần xoay theo vai
    - không cần transl normalization

    Input:
        joints_3d: (T, J, 3)

    Output:
        seq: (T, J*3)
    """
    T = joints_3d.shape[0]
    return joints_3d.reshape(T, -1)


def dtw_offset_debug(seq1, seq2, verbose=True):
    """
    DTW chuẩn O(T1*T2).

    Input:
        seq1: (T1, D)
        seq2: (T2, D)

    Output:
        mode_offset = mode(idx2 - idx1)
    """
    if verbose:
        print("\nĐang chạy DTW chuẩn...")

    T1, T2 = seq1.shape[0], seq2.shape[0]

    cost = cdist(seq1, seq2, metric="euclidean")

    dp = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1]
            )

    distance = dp[T1, T2]
    if verbose:
        print(f"Khoảng cách DTW: {distance:.4f}")

    path = []
    i, j = T1, T2

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        choices = [
            dp[i - 1, j - 1],
            dp[i - 1, j],
            dp[i, j - 1]
        ]

        move = int(np.argmin(choices))

        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()

    offsets = [idx2 - idx1 for idx1, idx2 in path]
    counter = Counter(offsets)

    if verbose:
        print("\nPhân bố offset:")
        for off, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            ratio = count / len(offsets) * 100
            print(f"  offset {off:4d}: {count:5d} lần ({ratio:.1f}%)")

    mode_offset = counter.most_common(1)[0][0]

    if verbose:
        print(f"\nMode offset = {mode_offset}")

    return mode_offset, distance, path


def compute_offset_from_pkls(pkl1_path, pkl2_path, smpl_model_path, verbose=False):
    """Compute frame offset between two PKL files using canonical-joints DTW method."""
    model = smplx.create(
        smpl_model_path,
        model_type="smpl",
        batch_size=1,
        use_pca=False,
        flat_hand_mean=True,
        gender="neutral",
    ).eval()

    cam1_data = load_pkl_data(str(pkl1_path))
    cam2_data = load_pkl_data(str(pkl2_path))
    joints1 = get_canonical_joints_3d(model, cam1_data)
    joints2 = get_canonical_joints_3d(model, cam2_data)
    seq1 = build_canonical_sequence(joints1)
    seq2 = build_canonical_sequence(joints2)
    delta_t, _, _ = dtw_offset_debug(seq1, seq2, verbose=verbose)
    return int(delta_t)


def main():
    pkl1 = input("Nhập đường dẫn file PKL camera 1: ").strip()
    pkl2 = input("Nhập đường dẫn file PKL camera 2: ").strip()
    smpl_model_path = input("Nhập đường dẫn thư mục chứa SMPL model: ").strip()

    print("\nĐang khởi tạo mô hình SMPL...")
    model = smplx.create(
        smpl_model_path,
        model_type="smpl",
        batch_size=1,
        use_pca=False,
        flat_hand_mean=True,
        gender="neutral"
    ).eval()

    print(f"\nĐọc {pkl1} ...")
    cam1_data = load_pkl_data(pkl1)

    print(f"Đọc {pkl2} ...")
    cam2_data = load_pkl_data(pkl2)

    print("\nTạo SMPL canonical joints cho camera 1...")
    joints1 = get_canonical_joints_3d(model, cam1_data)

    print("Tạo SMPL canonical joints cho camera 2...")
    joints2 = get_canonical_joints_3d(model, cam2_data)

    print("\nTạo canonical joint sequence cho DTW...")
    seq1 = build_canonical_sequence(joints1)
    seq2 = build_canonical_sequence(joints2)

    print(f"Camera 1: {seq1.shape[0]} frames, vector dim = {seq1.shape[1]}")
    print(f"Camera 2: {seq2.shape[0]} frames, vector dim = {seq2.shape[1]}")

    delta_t, distance, path = dtw_offset_debug(seq1, seq2)

    output_file = "delta_t_dtw_debug.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"delta_t={delta_t}\n")
        f.write(f"dtw_distance={distance}\n")
        f.write(f"path_length={len(path)}\n")

    print(f"\n✅ Kết quả: delta_t = {delta_t}")
    print(f"✅ Nghĩa là camera 2 lệch so với camera 1 khoảng {delta_t} frame")
    print(f"✅ Đã lưu vào {output_file}")


if __name__ == "__main__":
    main()
