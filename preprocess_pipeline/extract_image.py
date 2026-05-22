import os
import shlex
import subprocess
from glob import glob
from os.path import join


EXTENSIONS = [".mp4", ".MP4"]


def run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def get_video_fps(video_path, ffprobe="ffprobe"):
    cmd = (
        f'{ffprobe} -v error -select_streams v:0 -show_entries stream=r_frame_rate '
        f'-of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}'
    )
    raw = subprocess.check_output(cmd, shell=True, text=True).strip()
    if "/" in raw:
        num, den = raw.split("/", 1)
        fps = float(num) / float(den)
    else:
        fps = float(raw)
    if fps <= 0:
        raise ValueError(f"FPS không hợp lệ cho video: {video_path}")
    return fps


def extract_images(input_folder, output_folder, ffmpeg="ffmpeg", ffprobe="ffprobe", restart=False, debug=False):
    videos = sorted(sum([glob(join(input_folder, "*" + ext)) for ext in EXTENSIONS], []))

    if not videos:
        print(f"Không tìm thấy video .mp4 trong folder: {input_folder}")
        return

    print(f"Tìm thấy {len(videos)} video(s)")
    for videoname in videos:
        video_basename = os.path.splitext(os.path.basename(videoname))[0]
        outpath = join(output_folder, f"images_{video_basename}")
        os.makedirs(outpath, exist_ok=True)

        existing_frames = glob(join(outpath, "images_frame_*.jpg"))
        if existing_frames and not restart:
            print(f"Bỏ qua {video_basename} - đã tồn tại dữ liệu")
            continue

        fps = get_video_fps(videoname, ffprobe=ffprobe)
        fps_str = f"{fps:.10f}".rstrip("0").rstrip(".")
        cmd = (
            f'{ffmpeg} -i {shlex.quote(videoname)} -vf "fps={fps_str}" '
            f'-q:v 1 -start_number 1 {shlex.quote(join(outpath, "images_frame_%d.jpg"))}'
        )
        if not debug:
            cmd += " -loglevel error"

        print(f"\nĐang xử lý: {video_basename} | fps={fps_str}")
        run(cmd)
        print(f"Hoàn thành: {video_basename}")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    extract_images(
        input_folder=input_folder,
        output_folder=output_folder,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        restart=False,
        debug=False,
    )
