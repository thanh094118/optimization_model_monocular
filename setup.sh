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

echo "Kiểm tra gdown..."
if ! command -v gdown >/dev/null 2>&1; then
    echo "Chưa có gdown. Đang cài gdown..."
    python -m pip install gdown
else
    gdown --version
fi

echo "Kiểm tra import packages..."
python - <<'EOF'
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

print("Import OK")
EOF

echo "Hoàn tất cài đặt môi trường!"
echo ""

echo "========================================"
echo "2. TẢI VÀ CHUẨN BỊ DỮ LIỆU"
echo "========================================"

declare -A FOLDER_URLS=(
    ["models"]="https://drive.google.com/drive/folders/1vlVOnUQDhjYXL5w1izqPAH-zPot_yUxq?usp=sharing"
    ["input"]="https://drive.google.com/drive/folders/1BzMRVjwshaqjOouKGx-ktW6zNNbnGByQ?usp=sharing"
)

download_folder() {
    local folder="$1"
    local folder_url="$2"
    local marker_file="${folder}/.download_complete"

    if [ -f "$marker_file" ]; then
        echo "Folder $folder đã được tải trước đó, bỏ qua."
        return
    fi

    echo "Đang tải folder: $folder ..."
    mkdir -p "$folder"

    gdown --folder "$folder_url" -O "$folder" --remaining-ok

    touch "$marker_file"

    echo "Hoàn tất tải folder: $folder"
}

for folder in "${!FOLDER_URLS[@]}"; do
    download_folder "$folder" "${FOLDER_URLS[$folder]}"
done

echo ""
echo "========================================"
echo "HOÀN TẤT"
echo "========================================"
echo "Hoàn tất tải dữ liệu và thiết lập dự án!"

Chạy:

chmod +x setup.sh
./setup.sh