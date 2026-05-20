#!/bin/bash
set -e

echo "Tạo thư mục dữ liệu..."
mkdir -p 5 videos data

declare -A FILE_IDS=(
    ["smpl.zip"]="1axzI_DohdbZfhJfsc3gWDHQ8OkVVcVJ4"
    ["calib_from_cam.zip"]="16m1RVsMvzEdrI5uQ-mxF9_sJcqsFfeOz"
    ["5/wham_opencap_1.pkl"]="1y53im5EvAs4pIH5N-KOaKHNFBd6Ku10e"
    ["5/wham_opencap_2.pkl"]="1AT2Wm-chzwcI5AvzM7jqrjzL7qC5UvZL"
    ["videos/camera1.mp4"]="1yJTVqEs79hcFyF3-Q1F5rFb8FD1azqM-"
    ["videos/camera2.mp4"]="1AHAmdbmia_Fv4AVTj6bBGg2THOcgqBTB"
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

echo "Giải nén SMPL..."
if [ -f "smpl.zip" ]; then
    unzip -o smpl.zip
fi

echo "Giải nén calibration..."
if [ -f "calib_from_cam.zip" ]; then
    unzip -o calib_from_cam.zip
fi

echo "Hoàn tất tải dữ liệu."