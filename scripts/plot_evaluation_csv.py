from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

def _bootstrap_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
        return
    except ModuleNotFoundError:
        project_python = Path("/home/thanh/miniconda3/envs/easymocap/bin/python")
        current_python = Path(sys.executable)
        if project_python.exists() and current_python.resolve() != project_python.resolve():
            os.execv(str(project_python), [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise

_bootstrap_matplotlib()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODULES: tuple[tuple[str, str], ...] = (
    ("posed", "Pose"),
    ("fused", "Fusion"),
    ("learnable", "Learnable"),
    ("optimized", "Optimization"),
)

PRIORITIES: tuple[str, ...] = ("priority1_mm", "priority2_mm")


@dataclass(frozen=True)
class SeriesData:
    frames: list[int]
    values: dict[str, dict[str, list[float]]]


def _read_evaluation_csv(csv_path: Path) -> SeriesData:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frames: list[int] = []
    values: dict[str, dict[str, list[float]]] = {
        priority: {module: [] for module, _ in MODULES} for priority in PRIORITIES
    }

    module_columns: dict[str, dict[str, int]] = {priority: {} for priority in PRIORITIES}
    frame_col_idx: int | None = None

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_seen = False

        for row in reader:
            if not row:
                continue

            first = row[0].strip().lower()
            if first in {"sep=,", "sep="}:
                continue

            if not header_seen:
                if first != "frame":
                    raise ValueError(f"Missing header row in evaluation CSV: {csv_path}")
                header_seen = True
                frame_col_idx = 0
                for idx, column in enumerate(row[1:], start=1):
                    column = column.strip()
                    for priority in PRIORITIES:
                        suffix = f"_{priority}"
                        if column.endswith(suffix):
                            module_columns[priority][column[: -len(suffix)]] = idx
                missing = [
                    priority
                    for priority in PRIORITIES
                    if len(module_columns[priority]) != len(MODULES)
                ]
                if missing:
                    raise ValueError(
                        f"CSV is missing expected columns for: {', '.join(missing)}. "
                        f"Expected modules: {[name for name, _ in MODULES]}"
                    )
                continue

            if first == "average":
                continue

            if frame_col_idx is None:
                raise RuntimeError("Frame column index was not initialized.")

            frame_id = int(row[frame_col_idx].strip())
            frames.append(frame_id)

            for priority in PRIORITIES:
                for module_name, _ in MODULES:
                    col_idx = module_columns[priority][module_name]
                    value = float("nan")
                    if len(row) > col_idx and row[col_idx].strip():
                        value = float(row[col_idx].strip())
                    values[priority][module_name].append(value)

    if not frames:
        raise ValueError(f"No frame rows found in evaluation CSV: {csv_path}")

    return SeriesData(frames=frames, values=values)


def _iter_valid_values(series_map: dict[str, list[float]]) -> Iterable[float]:
    for module_name, _ in MODULES:
        for value in series_map[module_name]:
            if value == value:  # filter NaN without importing math
                yield value


def _plot_priority_chart(csv_path: Path, series: SeriesData, priority: str, output_path: Path) -> None:
    y_values = list(_iter_valid_values(series.values[priority]))
    if not y_values:
        raise ValueError(f"No numeric values found for {priority} in {csv_path}")

    x_min, x_max = min(series.frames), max(series.frames)
    y_min, y_max = min(y_values), max(y_values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    fig, ax = plt.subplots(figsize=(13, 6))

    line_styles = {
        "posed": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "fused": {"color": "#ff7f0e", "linestyle": "-", "marker": "s"},
        "learnable": {"color": "#2ca02c", "linestyle": "-", "marker": "^"},
        "optimized": {"color": "#d62728", "linestyle": "-", "marker": "D"},
    }

    for module_name, label in MODULES:
        ax.plot(
            series.frames,
            series.values[priority][module_name],
            label=label,
            linewidth=2.0,
            markersize=3.5,
            markevery=max(1, len(series.frames) // 20),
            **line_styles[module_name],
        )

    ax.set_title(f"{csv_path.stem} - {priority}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (mm)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", ncol=2, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot evaluation CSV results into two line charts for priority1_mm and priority2_mm."
    )
    parser.add_argument("csv_path", type=Path, help="Path to a PA-MPJPE/MPJPE/PCK CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the PNG files will be written. Defaults to ./outputs.",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv_path
    output_dir = args.output_dir or Path("outputs/full_5")

    series = _read_evaluation_csv(csv_path)

    for priority in PRIORITIES:
        output_path = output_dir / f"{csv_path.stem}_{priority}.png"
        _plot_priority_chart(csv_path, series, priority, output_path)
        print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
