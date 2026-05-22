import json
import numpy as np
import cv2
from pathlib import Path

IMAGE_DIR  = Path('output/demo/video_2/images')
KP_DIR     = Path('output/1')   # thư mục chứa JSON từ script tạo joints
OUTPUT_DIR = Path('output/demo/video_2/keypoints_vis3')

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ========== Định nghĩa skeleton cho 15 joints ==========
SKELETON_15 = [
    # Thân trên
    ('neck', 'left_shoulder'),
    ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_hand'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_hand'),

    # Hông
    ('left_hip', 'right_hip'),

    # Thân dưới
    ('neck', 'left_hip'),
    ('neck', 'right_hip'),

    # Chân trái
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('left_ankle', 'left_foot'),

    # Chân phải
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('right_ankle', 'right_foot'),
]

# ========== Màu sắc ==========
LEFT_JOINTS = {
    'left_shoulder',
    'left_elbow',
    'left_hand',
    'left_hip',
    'left_knee',
    'left_ankle',
    'left_foot'
}

RIGHT_JOINTS = {
    'right_shoulder',
    'right_elbow',
    'right_hand',
    'right_hip',
    'right_knee',
    'right_ankle',
    'right_foot'
}

HEAD_JOINTS = {'neck'}


def get_color(name):
    if name in LEFT_JOINTS:
        return (50, 200, 255)   # cam

    if name in RIGHT_JOINTS:
        return (255, 100, 50)   # xanh dương

    if name in HEAD_JOINTS:
        return (200, 255, 50)   # xanh lá

    return (255, 200, 50)       # vàng


# ========== Load ảnh mẫu ==========
image_paths = sorted(IMAGE_DIR.glob('*.jpg')) or sorted(IMAGE_DIR.glob('*.png'))

if not image_paths:
    raise FileNotFoundError(f"Không tìm thấy ảnh trong {IMAGE_DIR}")

sample = cv2.imread(str(image_paths[0]))
H, W = sample.shape[:2]

print(f"Resolution: {W}x{H}")

# ========== Auto intrinsics ==========
fx = fy = (W * W + H * H) ** 0.5
cx, cy = W / 2.0, H / 2.0

print(f"Auto intrinsics: fx=fy={fx:.2f} cx={cx:.1f} cy={cy:.1f}")


# ========== Hàm project 3D -> 2D ==========
def project(xyz):
    x, y, z = xyz

    if z <= 1e-5:
        return None

    u = int(round(fx * x / z + cx))
    v = int(round(fy * y / z + cy))

    return (u, v)


# ========== Xử lý từng frame ==========
kp_files = sorted(KP_DIR.glob('*.json'))

print(f"Tổng số file JSON: {len(kp_files)}")

for kp_file in kp_files:

    # person0_frame000123.json
    stem = kp_file.stem
    parts = stem.split('_frame')

    if len(parts) != 2:
        continue

    frame_idx = int(parts[1])

    img_path = IMAGE_DIR / f'{frame_idx:06d}.jpg'

    if not img_path.exists():
        img_path = IMAGE_DIR / f'{frame_idx:06d}.png'

    if not img_path.exists():
        print(f"Skip: missing image {frame_idx}")
        continue

    img = cv2.imread(str(img_path))

    if img is None:
        continue

    with open(kp_file, 'r') as f:
        data = json.load(f)

    if 'camera1' not in data:
        continue

    joints_3d = data['camera1']

    # ========== Project ==========
    proj = {}

    for name, xyz in joints_3d.items():
        p = project(xyz)

        if p is not None:
            proj[name] = p

    # ========== Vẽ skeleton ==========
    for (a, b) in SKELETON_15:

        if a in proj and b in proj:

            p1, p2 = proj[a], proj[b]

            if (
                0 <= p1[0] < W and
                0 <= p1[1] < H and
                0 <= p2[0] < W and
                0 <= p2[1] < H
            ):
                cv2.line(
                    img,
                    p1,
                    p2,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA
                )

    # ========== Vẽ joints ==========
    for name, (px, py) in proj.items():

        if 0 <= px < W and 0 <= py < H:

            cv2.circle(
                img,
                (px, py),
                5,
                get_color(name),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                img,
                (px, py),
                6,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    # ========== Text ==========
    cv2.putText(
        img,
        f'Frame {frame_idx:03d} | SMPL 15 joints',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ========== Save ==========
    out_path = OUTPUT_DIR / f'{frame_idx:06d}.jpg'

    cv2.imwrite(str(out_path), img)

print(f"Hoàn tất. Kết quả lưu tại {OUTPUT_DIR}")