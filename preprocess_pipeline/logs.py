def log_module_start(output_dir: str) -> None:
    print(f"[Preprocess] Start | output={output_dir}")


def log_offset_selected(offset: int, path: str) -> None:
    print(f"[Preprocess] Selected offset | offset={offset} | file={path}")


def log_module_done(output_dir: str) -> None:
    print(f"[Preprocess] Done | output={output_dir}")
