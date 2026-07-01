from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

try:
    import joblib
except Exception:  # pragma: no cover - joblib is expected in the project env
    joblib = None


DEFAULT_GROUPS: dict[str, list[int]] = {
    "S1/Seq1": [0, 1, 2, 5, 6],
    "S1/Seq2": [2, 6, 8],
    "S3/Seq1": [0, 1, 2, 6, 8],
    "S3/Seq2": [6, 8],
}


def _extract_person_payload(wham_data: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(wham_data, Mapping):
        if 0 in wham_data:
            value = wham_data[0]
            if isinstance(value, Mapping):
                return value
        if "0" in wham_data:
            value = wham_data["0"]
            if isinstance(value, Mapping):
                return value
        for value in wham_data.values():
            if isinstance(value, Mapping):
                return value
    if isinstance(wham_data, list):
        for value in wham_data:
            if isinstance(value, Mapping):
                return value
    return None


def _load_wham_payload(path: Path) -> Any:
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with path.open("rb") as f:
        return pickle.load(f)


def _load_frame_ids(pkl_path: Path) -> list[int]:
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing WHAM PKL: {pkl_path}")

    data = _load_wham_payload(pkl_path)
    person = _extract_person_payload(data)
    if person is None:
        raise ValueError(f"Cannot find person payload in {pkl_path}")

    tracking = person.get("tracking_results_for_reproj")
    if not isinstance(tracking, Mapping):
        raise ValueError(f"Missing tracking_results_for_reproj in {pkl_path}")

    frame_ids = tracking.get("frame_id")
    if frame_ids is None:
        raise ValueError(f"Missing frame_id in {pkl_path}")

    return [int(v) for v in frame_ids]


def _summarize_frame_ids(frame_ids: list[int]) -> dict[str, Any]:
    if not frame_ids:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "preview_head": [],
            "preview_tail": [],
            "gaps": [],
        }

    gaps = []
    for prev, curr in zip(frame_ids, frame_ids[1:]):
        if curr != prev + 1:
            gaps.append([prev, curr, curr - prev])

    return {
        "count": len(frame_ids),
        "first": frame_ids[0],
        "last": frame_ids[-1],
        "preview_head": frame_ids[:10],
        "preview_tail": frame_ids[-10:],
        "gaps": gaps[:20],
    }


def _scan_group(root: Path, group_rel: str, cameras: list[int]) -> dict[str, Any]:
    group_dir = root / group_rel
    cameras_info: dict[str, Any] = {}
    for camera_id in cameras:
        pkl_path = group_dir / f"video_{camera_id}" / "wham_opencap.pkl"
        frame_ids = _load_frame_ids(pkl_path)
        cameras_info[f"video_{camera_id}"] = {
            "pkl_path": str(pkl_path),
            "frame_ids": frame_ids,
            "summary": _summarize_frame_ids(frame_ids),
        }

    common = set.intersection(*(set(info["frame_ids"]) for info in cameras_info.values())) if cameras_info else set()
    union = set.union(*(set(info["frame_ids"]) for info in cameras_info.values())) if cameras_info else set()
    return {
        "group": group_rel,
        "cameras": cameras_info,
        "shared_frame_ids": sorted(common),
        "union_frame_ids": sorted(union),
    }


def _parse_groups(values: Optional[list[str]]) -> dict[str, list[int]]:
    if not values:
        return DEFAULT_GROUPS

    groups: dict[str, list[int]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid group spec '{item}'. Use 'path=0,1,2'.")
        rel_path, cam_spec = item.split("=", 1)
        rel_path = rel_path.strip().rstrip("/\\")
        cams = [int(part.strip()) for part in cam_spec.split(",") if part.strip()]
        if not cams:
            raise ValueError(f"Invalid camera list in '{item}'.")
        groups[rel_path] = cams
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan WHAM PKLs and export real frame IDs per camera group.")
    parser.add_argument("--root", type=Path, default=Path("input/tt"), help="Root directory containing Seq folders.")
    parser.add_argument(
        "--group",
        action="append",
        help="Override/add a group in the form 'S1/Seq1=0,1,2'. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wham_frame_ids_report.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument("--no-output", action="store_true", help="Do not write the JSON report to disk.")
    args = parser.parse_args()

    groups = _parse_groups(args.group)
    report = {
        "root": str(args.root),
        "groups": [],
    }

    for group_rel, cameras in groups.items():
        group_report = _scan_group(args.root, group_rel, cameras)
        report["groups"].append(group_report)

    for group_report in report["groups"]:
        print(f"[Group] {group_report['group']}")
        for camera_name, camera_report in group_report["cameras"].items():
            summary = camera_report["summary"]
            print(
                f"  - {camera_name}: count={summary['count']}, "
                f"first={summary['first']}, last={summary['last']}, gaps={len(summary['gaps'])}"
            )
            print(f"    head={summary['preview_head']}")
            print(f"    tail={summary['preview_tail']}")
        print(f"  shared_count={len(group_report['shared_frame_ids'])}")
        print(f"  union_count={len(group_report['union_frame_ids'])}")

    if not args.no_output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[Saved] {args.output}")


if __name__ == "__main__":
    main()
