# Edge Cases

## Consolidated Root Cache Notes

- Root-level `edge_cases.md` was consolidated here:
  - non-integer calibration suffixes are pushed to the end of sorted calibration candidates
  - Skeletool parser should tolerate missing `intrinsicMat`, `extrinsicMat`, `W`, and `H`
  - when more than two calibration files exist, only the two lowest IDs are mapped to `cam1` and `cam2`

## Config And Stage Selection

- `pipeline.py` always runs preprocess first even when another stage is selected.
- `preprocess` is now a valid explicit stage value in both CLI choices and config validation.

## Camera Profile Schema

- New camera profile translation key is `tvec`; HBH readers still fallback to legacy `xyz` for backward compatibility.
- Intrinsics source is configurable (`intri_cam` vs `intri_esti`); consumers must read selected source consistently.

## HBH Method Outputs

- HBH may run one method or all methods:
  - `primary_method: all` => all method folders generated
  - `primary_method: <method>` => only one method folder generated
- Evaluation and visualization should handle either case based on discovered `method_*` folders.

## Frame Alignment

- HBH evaluation matches frames by numeric id from filenames (`hbh_data_<id>` vs `ground_truth_<id>`).
- Missing/filtered frames (confidence gate) reduce overlap; resulting CSV frame counts can vary by run.

## Learnable Integration

- Learnable stage expects fused keypoints under `fused_output_dir/keypoints3d/fused_data_*.json`; missing `fused_output_dir/metadata` must not block execution.
- Upstream Learnable-SMPLify expects assets under `models/smpl/*`; adapter now copies from configured project paths when these files are absent.
- If only one SMPL pickle is available (`SMPL_NEUTRAL.pkl`), adapter may mirror it to male/female filenames for runtime compatibility.

## Optimization Standalone Integration

- `optimization_pipeline` is currently standalone only; no `main.py` or `pipeline.py` stage dispatch exists yet for these phase scripts.
- Standalone phase scripts are intentionally self-contained and do not import sibling modules; helper logic is duplicated on purpose to preserve file independence.
- Phase scripts accept either direct payload dicts or wrapper dicts such as `{'params': ...}` / `{'cameras': ...}` / `{'keypoints': ...}` when loading input files.
- Frame-broadcast behavior is supported for common singleton inputs:
  - `params['shapes']` may be provided as `(10,)` or `(1, 10)` and is expanded across frames.
  - `params['Rh']`, `params['Th']`, or `params['poses']` may be single-frame and will be repeated when safe.
  - camera `K`, `R`, `T` may be constant per sequence or framewise arrays.
- Phase 2, 5, and 6 all assume the number of frames is defined by the loaded 2D keypoint sequence; mismatched frame counts in params/cameras are only tolerated when they are broadcastable from a singleton.
