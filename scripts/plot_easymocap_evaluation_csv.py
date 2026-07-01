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
            os.execv(
                str(project_python),
                [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
        raise


_bootstrap_matplotlib()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS: tuple[str, ...] = ("MPJPE", "PA-MPJPE", "PCK")
PRIORITIES: tuple[str, ...] = ("priority1_mm", "priority2_mm")
MODULES: tuple[tuple[str, str], ...] = (
    ("easymocap", "EasyMocap"),
    ("fused", "Fusion"),
    ("learnable", "Learnable"),
    ("optimized", "Optimization"),
)


@dataclass(frozen=True)
class SeriesData:
    frames: list[int]
    values: dict[str, dict[str, list[float]]]


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    easy_dir: Path
    reference_dir: Path
    output_dir: Path
    frame_shift: int
    cam_pairs: tuple[tuple[str, str, str], ...]


PRESETS: dict[str, DatasetPreset] = {
    "easymocap": DatasetPreset(
        name="easymocap",
        easy_dir=Path("outputs/easymocap"),
        reference_dir=Path("a/evaluation_results2"),
        output_dir=Path("outputs/easymocap_plots"),
        frame_shift=1,
        cam_pairs=(
            ("cam1", "cam1", "cam1"),
            ("cam2", "cam2", "cam2"),
        ),
    ),
    "s3_seq1": DatasetPreset(
        name="s3_seq1",
        easy_dir=Path("outputs/easymocap_s3_seq1"),
        reference_dir=Path("a/evaluation_results5"),
        output_dir=Path("outputs/easymocap_s3_seq1_plots"),
        frame_shift=-9108,
        cam_pairs=(
            ("cam1", "cam2", "cam1"),
        ),
    ),
}


def _load_metric_csv(
    csv_path: Path,
    module_column_map: dict[str, str],
    frame_shift: int = 0,
) -> SeriesData:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    module_names = tuple(module_column_map.values())
    frames: list[int] = []
    values: dict[str, dict[str, list[float]]] = {
        priority: {module: [] for module in module_names} for priority in PRIORITIES
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
                            source_prefix = column[: -len(suffix)]
                            if source_prefix in module_column_map:
                                module_columns[priority][module_column_map[source_prefix]] = idx
                missing = [
                    priority
                    for priority in PRIORITIES
                    if not set(module_names).issubset(module_columns[priority].keys())
                ]
                if missing:
                    raise ValueError(
                        f"CSV is missing expected columns for: {', '.join(missing)}. "
                    f"Expected modules: {list(module_names)}"
                )
                continue

            if first == "average":
                continue

            frame_id = int(row[0].strip()) + frame_shift
            frames.append(frame_id)

            for priority in PRIORITIES:
                for module_name in module_names:
                    col_idx = module_columns[priority][module_name]
                    value = float("nan")
                    if len(row) > col_idx and row[col_idx].strip():
                        value = float(row[col_idx].strip())
                    values[priority][module_name].append(value)

    if not frames:
        raise ValueError(f"No frame rows found in evaluation CSV: {csv_path}")

    return SeriesData(frames=frames, values=values)


def _merge_series(
    easy_series: SeriesData,
    reference_series: SeriesData,
    easy_name: str = "easymocap",
) -> SeriesData:
    easy_map = {
        frame: {
            priority: easy_series.values[priority][easy_name][idx]
            for priority in PRIORITIES
        }
        for idx, frame in enumerate(easy_series.frames)
    }
    reference_map = {
        frame: {
            priority: {
                module: reference_series.values[priority][module][idx]
                for module, _ in MODULES[1:]
            }
            for priority in PRIORITIES
        }
        for idx, frame in enumerate(reference_series.frames)
    }

    common_frames = sorted(set(easy_map) & set(reference_map))
    if not common_frames:
        raise ValueError("No overlapping frames found after applying the frame shift.")

    merged_values: dict[str, dict[str, list[float]]] = {
        priority: {module: [] for module, _ in MODULES} for priority in PRIORITIES
    }
    for frame in common_frames:
        for priority in PRIORITIES:
            merged_values[priority]["easymocap"].append(easy_map[frame][priority])
            for module, _ in MODULES[1:]:
                merged_values[priority][module].append(reference_map[frame][priority][module])

    return SeriesData(frames=common_frames, values=merged_values)


def _compute_deltas(series: SeriesData) -> dict[str, dict[str, list[float]]]:
    deltas: dict[str, dict[str, list[float]]] = {priority: {} for priority in PRIORITIES}
    for priority in PRIORITIES:
        baseline = series.values[priority]["easymocap"]
        deltas[priority]["easymocap"] = [0.0 if value == value else float("nan") for value in baseline]
        for module_name in ("fused", "learnable", "optimized"):
            deltas[priority][module_name] = [
                (value - base) if (value == value and base == base) else float("nan")
                for value, base in zip(series.values[priority][module_name], baseline)
            ]
    return deltas


def _iter_valid_values(series_map: dict[str, list[float]]) -> Iterable[float]:
    for module_name, _ in MODULES:
        for value in series_map[module_name]:
            if value == value:
                yield value


def _plot_raw(csv_path: Path, series: SeriesData, priority: str, output_path: Path) -> None:
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
        "easymocap": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
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


def _plot_delta(csv_path: Path, series: SeriesData, priority: str, output_path: Path) -> None:
    deltas = _compute_deltas(series)[priority]
    y_values = list(_iter_valid_values(deltas))
    if not y_values:
        raise ValueError(f"No numeric delta values found for {priority} in {csv_path}")

    x_min, x_max = min(series.frames), max(series.frames)
    y_min = min(0.0, min(y_values))
    y_max = max(0.0, max(y_values))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    fig, ax = plt.subplots(figsize=(13, 6))

    line_styles = {
        "easymocap": {"color": "#444444", "linestyle": "--", "marker": "o"},
        "fused": {"color": "#ff7f0e", "linestyle": "-", "marker": "s"},
        "learnable": {"color": "#2ca02c", "linestyle": "-", "marker": "^"},
        "optimized": {"color": "#d62728", "linestyle": "-", "marker": "D"},
    }

    for module_name, label in MODULES:
        ax.plot(
            series.frames,
            deltas[module_name],
            label=label if module_name != "easymocap" else "EasyMocap baseline (0)",
            linewidth=2.0,
            markersize=3.5,
            markevery=max(1, len(series.frames) // 20),
            **line_styles[module_name],
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title(f"{csv_path.stem} - {priority} (delta vs EasyMocap)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Deviation from EasyMocap (mm)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", ncol=2, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset(preset: DatasetPreset) -> None:
    for metric in METRICS:
        for out_cam, easy_cam, ref_cam in preset.cam_pairs:
            easy_csv = preset.easy_dir / f"{metric}_{easy_cam}.csv"
            ref_csv = preset.reference_dir / f"{metric}_{ref_cam}.csv"
            if not easy_csv.exists():
                raise FileNotFoundError(f"Missing EasyMocap CSV: {easy_csv}")
            if not ref_csv.exists():
                raise FileNotFoundError(f"Missing reference CSV: {ref_csv}")

            easy_series = _load_metric_csv(
                easy_csv,
                {"openpose25": "easymocap"},
                frame_shift=preset.frame_shift,
            )
            ref_series = _load_metric_csv(
                ref_csv,
                {
                    "posed": "posed",
                    "fused": "fused",
                    "learnable": "learnable",
                    "optimized": "optimized",
                },
            )
            merged = _merge_series(easy_series, ref_series)

            for priority in PRIORITIES:
                raw_output = (
                    preset.output_dir
                    / preset.name
                    / metric
                    / f"{metric}_{out_cam}_{priority}.png"
                )
                delta_output = (
                    preset.output_dir
                    / preset.name
                    / metric
                    / f"{metric}_{out_cam}_{priority}_delta.png"
                )
                _plot_raw(ref_csv, merged, priority, raw_output)
                _plot_delta(ref_csv, merged, priority, delta_output)
                print(f"[Saved] {raw_output}")
                print(f"[Saved] {delta_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EasyMocap-mapped evaluation CSVs with raw and delta views."
    )
    parser.add_argument(
        "--preset",
        choices=["all", *sorted(PRESETS)],
        default="all",
        help="Which dataset mapping preset to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional root directory for PNG outputs. Defaults to each preset's built-in output dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preset == "all":
        presets = list(PRESETS.values())
    else:
        presets = [PRESETS[args.preset]]

    for preset in presets:
        if args.output_dir is not None:
            preset = DatasetPreset(
                name=preset.name,
                easy_dir=preset.easy_dir,
                reference_dir=preset.reference_dir,
                output_dir=args.output_dir,
                frame_shift=preset.frame_shift,
                cam_pairs=preset.cam_pairs,
            )
        _plot_dataset(preset)


if __name__ == "__main__":
    main()
