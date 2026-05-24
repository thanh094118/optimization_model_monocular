def log_disabled() -> None:
    print("[Learnable] Disabled by config: learnable_smplify.enabled=false")


def log_header() -> None:
    print("[Learnable] Learnable-SMPLify after judgement/fusion")


def log_done(output_dir: str) -> None:
    print(f"[Learnable] Done. Output: {output_dir}")
