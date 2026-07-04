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
echo "2. TẢI VÀ GIẢI NÉN DỮ LIỆU"
echo "========================================"

mkdir -p archives

declare -A ZIP_FILE_IDS=(
    ["input.zip"]="19SVgxS_vn4HoB8K66nSGSw3VOTHt4hUh"
    ["models.zip"]="1oVFBNtVLBH2mV1jLazWC5Tud2-PxJtlo"
)

declare -A ZIP_DEST_DIRS=(
    ["input.zip"]="input"
    ["models.zip"]="models"
)

download_zip() {
    local zip_name="$1"
    local file_id="$2"
    local zip_path="archives/${zip_name}"

    if [ ! -f "$zip_path" ]; then
        echo "Đang tải ${zip_name} ..."
        gdown "https://drive.google.com/uc?id=${file_id}" -O "$zip_path"
    else
        echo "${zip_name} đã tồn tại, bỏ qua tải lại."
    fi
}

extract_zip() {
    local zip_name="$1"
    local dest_dir="$2"
    local zip_path="archives/${zip_name}"
    local marker_file="${dest_dir}/.extract_complete"

    if [ -f "$marker_file" ]; then
        echo "${dest_dir} đã được giải nén trước đó, bỏ qua."
        return
    fi

    echo "Đang giải nén ${zip_name} vào ${dest_dir} ..."

    rm -rf "$dest_dir"
    mkdir -p "$dest_dir"

    python - <<EOF
import zipfile
from pathlib import Path
import shutil

zip_path = Path("${zip_path}")
dest_dir = Path("${dest_dir}")
tmp_dir = Path("${dest_dir}_tmp_extract")

if tmp_dir.exists():
    shutil.rmtree(tmp_dir)

tmp_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(tmp_dir)

items = list(tmp_dir.iterdir())

# Trường hợp zip chứa sẵn folder input/ hoặc models/
if len(items) == 1 and items[0].is_dir() and items[0].name == dest_dir.name:
    inner_dir = items[0]
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.move(str(inner_dir), str(dest_dir))
else:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        shutil.move(str(item), str(dest_dir / item.name))

shutil.rmtree(tmp_dir)
EOF

    touch "$marker_file"
    echo "Hoàn tất giải nén ${zip_name}."
}

for zip_name in "${!ZIP_FILE_IDS[@]}"; do
    download_zip "$zip_name" "${ZIP_FILE_IDS[$zip_name]}"
done

for zip_name in "${!ZIP_DEST_DIRS[@]}"; do
    extract_zip "$zip_name" "${ZIP_DEST_DIRS[$zip_name]}"
done

echo ""
echo "========================================"
echo "3. KIỂM TRA FILE CẦN THIẾT"
echo "========================================"

REQUIRED=(
    "input/testcase1/wham_opencap_1.pkl"
    "input/testcase1/wham_opencap_2.pkl"
    "input/testcase1/video_1.mp4"
    "input/testcase1/video_2.mp4"
    "input/testcase1/gtruth_results"
    "models/SMPL_NEUTRAL.pkl"
    "models/smpl_partSegmentation_mapping.pkl"
    "models/J_regressor_body25_plus_palm27.npy"
    "models/best_ckpt.pth.tar"
)

MISSING=0

for path in "${REQUIRED[@]}"; do
    if [ ! -e "$path" ]; then
        echo "MISSING: $path"
        MISSING=1
    else
        echo "OK: $path"
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "CẢNH BÁO: Thiếu một số file cần thiết."
    echo "Hãy kiểm tra lại cấu trúc bên trong input.zip và models.zip."
    exit 1
fi

echo ""
echo "========================================"
echo "HOÀN TẤT"
echo "========================================"
echo "Hoàn tất tải, giải nén và kiểm tra dữ liệu!"