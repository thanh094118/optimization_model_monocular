#!/bin/bash
set -e

echo "========================================"
echo "1. CÀI ĐẶT MÔI TRƯỜNG"
echo "========================================"

echo "Cập nhật pip..."
python -m pip install --upgrade pip

echo "Cài đặt Python packages từ requirements.txt..."
python -m pip install -r requirements.txt

echo "Kiểm tra ffmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "CẢNH BÁO: Chưa có ffmpeg."
    echo "Ubuntu/WSL: sudo apt install ffmpeg"
    echo "Windows: cài ffmpeg và thêm vào PATH"
else
    ffmpeg -version | head -n 1
fi

echo "Kiểm tra import packages..."
python - <<EOF
import numpy
import torch
import smplx
import joblib
import yaml
import cv2
import matplotlib
import scipy
import loguru
import decouple
import kornia
import slahmr
print("Import OK")
EOF

echo "Hoàn tất cài đặt môi trường!"
echo ""

echo "========================================"
echo "2. TẢI VÀ CHUẨN BỊ DỮ LIỆU"
echo "========================================"

declare -A FILE_IDS=(
    ["models/SMPL_NEUTRAL.pkl"]="1rg2QnvmMgoS7Cpok3X0RIie3ZVwW_c0U"
    ["models/J_regressor_body25.npy"]="1SCsg3XuDEMkCyVO4s6p23Mj-R-7su9uQ"
    ["models/yolo26x-pose.pt"]="1iAVLQa0NWH3ICBpkBrbTUKU6DQPvwUI9"
    ["models/smpl_partSegmentation_mapping.pkl"]="1P6h-MOdTvb5q4EPntJmsEQwNurut6u2I"    
    ["input/cameraIntrinsics.pickle"]="1Lhg4qY8GEg4V6ZE8nK4rD7mP4PAbE9DS"
    ["input/wham_opencap_1.pkl"]="12xU0FGZpFaWOcQ6JKHUqyJdQClL4I1of"
    ["input/wham_opencap_2.pkl"]="1xfVxIVfp6A1D8q3YXca8_CAFzgxtBgyY"
    ["input/video_2.mp4"]="1a02Vv976w-7bLpZAj__NdTOGwtmDL_NL"
    ["input/video_8.mp4"]="1quUAbCczwASRbnx2HqNIDFusosfu4wA1"
)

download_file() {
    local filename=$1
    local file_id=$2

    if [ ! -f "$filename" ]; then
        echo "Đang tải $filename ..."
        mkdir -p "$(dirname "$filename")"
        gdown "https://drive.google.com/uc?id=${file_id}" -O "$filename"
    else
        echo "$filename đã tồn tại, bỏ qua."
    fi
}

for filename in "${!FILE_IDS[@]}"; do
    download_file "$filename" "${FILE_IDS[$filename]}"
done

echo "Hoàn tất tải dữ liệu và thiết lập dự án!"