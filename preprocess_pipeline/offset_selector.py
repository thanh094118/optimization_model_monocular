from pathlib import Path


def read_offset_from_txt(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Offset file not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip()
    return int(raw)


def resolve_selected_offset(config: dict) -> tuple[int, str, Path]:
    offset_cfg = config.get("preprocess", {}).get("offset", {})
    method = str(offset_cfg.get("method", "paper")).strip().lower()
    if method not in {"paper", "colab"}:
        raise ValueError(f"Invalid preprocess.offset.method={method!r}, expected 'paper' or 'colab'")

    output_dir = Path(offset_cfg.get("output_dir", "output/preprocess_results"))
    filename = "offset_paper.txt" if method == "paper" else "offset_colab.txt"
    offset_path = output_dir / filename
    offset = read_offset_from_txt(offset_path)
    return offset, method, offset_path

