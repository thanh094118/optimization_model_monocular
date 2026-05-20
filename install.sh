#!/bin/bash
set -e

echo "Cập nhật pip..."
python -m pip install --upgrade pip

echo "Cài đặt Python packages..."
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
print("Import OK")
EOF

echo "Hoàn tất cài đặt môi trường!"