def log_disabled() -> None:
    print("[Visualization] Disabled by config: visualization.enabled=false")


def log_header() -> None:
    print("[Visualization] Optimized vs Video vs Learnable")


def log_done(output_dir: str) -> None:
    print(f"[Visualization] Done. Output: {output_dir}")
