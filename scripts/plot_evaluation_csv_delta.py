from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path


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
DELTA_MODULES: tuple[tuple[str, str], ...] = (
    ("posed", "Pose"),
    ("fused", "Fusion"),
    ("learnable", "Learnable"),
    ("optimized", "Optimization"),
)


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

            frame_id = int(row[0].strip())
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


def _compute_deltas(series: SeriesData) -> dict[str, dict[str, list[float]]]:
    deltas: dict[str, dict[str, list[float]]] = {priority: {} for priority in PRIORITIES}
    for priority in PRIORITIES:
        pose_values = series.values[priority]["posed"]
        deltas[priority]["posed"] = [0.0 if value == value else float("nan") for value in pose_values]
        for module_name in ("fused", "learnable", "optimized"):
            module_values = series.values[priority][module_name]
            deltas[priority][module_name] = [
                (curr - pose) if (curr == curr and pose == pose) else float("nan")
                for curr, pose in zip(module_values, pose_values)
            ]
    return deltas


def _valid_values(series_map: dict[str, list[float]]) -> list[float]:
    values: list[float] = []
    for module_name, _ in DELTA_MODULES:
        for value in series_map[module_name]:
            if value == value:
                values.append(value)
    return values


def _plot_priority_chart(csv_path: Path, frames: list[int], deltas: dict[str, list[float]], priority: str, output_path: Path) -> None:
    y_values = _valid_values(deltas)
    if not y_values:
        raise ValueError(f"No numeric values found for {priority} in {csv_path}")

    x_min, x_max = min(frames), max(frames)
    y_min = min(0.0, min(y_values))
    y_max = max(0.0, max(y_values))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    fig, ax = plt.subplots(figsize=(13, 6))

    line_styles = {
        "posed": {"color": "#444444", "linestyle": "--", "marker": "o"},
        "fused": {"color": "#ff7f0e", "linestyle": "-", "marker": "s"},
        "learnable": {"color": "#2ca02c", "linestyle": "-", "marker": "^"},
        "optimized": {"color": "#d62728", "linestyle": "-", "marker": "D"},
    }

    for module_name, label in DELTA_MODULES:
        ax.plot(
            frames,
            deltas[module_name],
            label=label if module_name != "posed" else "Pose baseline (0)",
            linewidth=2.0,
            markersize=3.5,
            markevery=max(1, len(frames) // 20),
            **line_styles[module_name],
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title(f"{csv_path.stem} - {priority} (delta vs pose)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Deviation from pose (mm)")
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
        description="Plot evaluation CSV deviations relative to the pose column."
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
    output_dir = args.output_dir or Path("outputs/test_3/a")

    series = _read_evaluation_csv(csv_path)
    deltas = _compute_deltas(series)

    for priority in PRIORITIES:
        output_path = output_dir / f"{csv_path.stem}_{priority}_delta.png"
        _plot_priority_chart(csv_path, series.frames, deltas[priority], priority, output_path)
        print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
