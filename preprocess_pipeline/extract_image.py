import os
import shlex
import subprocess
from glob import glob
from os.path import join

from preprocess_pipeline.offset_colab import compute_offset_from_pkls as compute_colab_offset
from preprocess_pipeline.offset_paper import compute_offset_from_pkls as compute_paper_offset


EXTENSIONS = [".mp4", ".MP4"]


def run(cmd):
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
        fps = get_video_fps(videoname, ffprobe=ffprobe)
        fps_str = f"{fps:.10f}".rstrip("0").rstrip(".")

        existing_frames = glob(join(outpath, "images_frame_*.jpg"))
        if existing_frames and not restart:
            print(
                f"[Preprocess] Extract skip | input={videoname} | output={outpath} | "
                f"fps={fps_str} | frames={len(existing_frames)}"
            )
            continue

        cmd = (
            f'{ffmpeg} -i {shlex.quote(videoname)} -vf "fps={fps_str}" '
            f'-q:v 1 -start_number 1 {shlex.quote(join(outpath, "images_frame_%d.jpg"))}'
        )
        if not debug:
            cmd += " -loglevel error"

        print(
            f"[Preprocess] Extract start | input={videoname} | output={outpath} | fps={fps_str}"
        )
        run(cmd)
        frame_count = len(glob(join(outpath, "images_frame_*.jpg")))
        print(
            f"[Preprocess] Extract done | input={videoname} | output={outpath} | "
            f"fps={fps_str} | frames={frame_count}"
        )


def run_offset_estimation(input_folder, output_folder, smpl_model_path):
    pkl_files = sorted(glob(join(input_folder, "*.pkl")))
    if len(pkl_files) < 2:
        print(f"Bỏ qua offset: cần ít nhất 2 file .pkl trong {input_folder}")
        return

    pkl1, pkl2 = pkl_files[0], pkl_files[1]
    print(f"[Preprocess] Offset input | cam1={os.path.basename(pkl1)} | cam2={os.path.basename(pkl2)}")

    colab_offset = compute_colab_offset(pkl1, pkl2, smpl_model_path, detail_print=False)
    paper_offset = compute_paper_offset(pkl1, pkl2, smpl_model_path, verbose=False)

    colab_out = join(output_folder, "offset_colab.txt")
    paper_out = join(output_folder, "offset_paper.txt")

    with open(colab_out, "w", encoding="utf-8") as f:
        f.write(f"offset = {colab_offset}\n")
    with open(paper_out, "w", encoding="utf-8") as f:
        f.write(f"offset = {paper_offset}\n")

    print(f"[Preprocess] Offset colab={colab_offset}")
    print(f"[Preprocess] Offset paper={paper_offset}")
    print(f"[Preprocess] Offset saved | colab={colab_out} | paper={paper_out}")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = join("output", "preprocess_results")
    smpl_model_path = join("models", "SMPL_NEUTRAL.pkl")
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

    run_offset_estimation(
        input_folder=input_folder,
        output_folder=output_folder,
        smpl_model_path=smpl_model_path,
    )
