def log_disabled() -> None:
    print("[HBH] Disabled by config: hbh.enabled=false")


def log_start(video1: str, video2: str, out_dir: str) -> None:
    print(f"[HBH] Start | video1={video1} | video2={video2} | output={out_dir}")


def log_done(out_dir: str, n_frames: int) -> None:
    print(f"[HBH] Done | output={out_dir} | frames={n_frames}")


def log_summary(
    num_results: int,
    primary_method: str,
    selected_methods: list[str],
    output_dir: str,
    video1: str,
    video2: str,
    cam_profiles: list[str],
    ground_truth_dir: str,
) -> None:
    print(
        "[HBH] Summary | "
        f"num_results={num_results} | primary_method={primary_method} | "
        f"selected_methods={selected_methods} | "
        f"output={output_dir} | video1={video1} | video2={video2} | "
        f"camera_profiles={cam_profiles} | ground_truth_dir={ground_truth_dir}"
    )
