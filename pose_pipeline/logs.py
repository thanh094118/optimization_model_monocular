def log_disabled() -> None:
    print("[Pose] Disabled by config: pose_export.enabled=false")


def log_using_offset(offset: int, method: str) -> None:
    print(f"[Pose] Using offset={offset} ({method})")


def log_done(output_dir: str) -> None:
    print(f"[Pose] Done. Output: {output_dir}")
