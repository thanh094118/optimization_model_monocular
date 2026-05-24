def log_disabled() -> None:
    print("[Fusion] Disabled by config: fusion.enabled=false")


def log_occlusion(frame_idx: int, occluded_keys: list[str]) -> None:
    if occluded_keys:
        print(f"[Fusion] Frame {frame_idx}: Occlusion: {', '.join(occluded_keys)}")


def log_done(output_dir: str) -> None:
    print(f"[Fusion] Done. Output: {output_dir}")
