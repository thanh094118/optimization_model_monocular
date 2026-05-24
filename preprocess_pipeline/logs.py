def log_module_start(input_dir: str, output_dir: str) -> None:
    print(f"[Preprocess] Start | input={input_dir} | output={output_dir}")


def log_offset_selected(method: str, offset: int, path: str) -> None:
    print(f"[Preprocess] Selected offset | method={method} | offset={offset} | file={path}")


def log_module_done(output_dir: str) -> None:
    print(f"[Preprocess] Done | output={output_dir}")
