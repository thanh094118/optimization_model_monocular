# Known Issues

## Consolidated Root Cache Notes

- Root-level `known_issues.md` was consolidated here:
  - if fewer than 2 `.calibration` files are found, fallback to estimated intrinsics may hide calibration quality problems unless logs are monitored
  - estimated intrinsics depend on OpenCV successfully reading the MP4 dimensions; wrong/corrupt videos can prevent intrinsic generation for that camera

## Environment

- System `/usr/bin/python` in this workspace may miss required packages (e.g., NumPy). Use project env Python (`/home/thanh/miniconda3/envs/easymocap/bin/python`) for checks/runs.
- `ffprobe` in current environment has shown runtime symbol issues (`avio_protocol_get_class`) in some runs; this can block preprocess video extraction steps.
- Ultralytics custom checkpoints may fail to load if model class is not present in installed package version (e.g., `Pose26` mismatch).
- Learnable-SMPLify upstream source is GPU-biased and may break on CPU without adapter patches (`x.get_device()` / `.cuda()` usage in backbone/transforms).
- Standalone optimization phases 2, 5, and 6 require `smplx` at runtime; the default shell `python` used during inspection currently does not import `smplx`, so those scripts must be run from the project environment.

## HBH-Specific

- HBH visualization can fail to initialize writer when output canvas is too large for MPEG-4; mitigation is implemented via auto-scaling and writer-open checks.
- If `hbh_data_*.json` were generated before key2D/debug schema additions, visualization debug overlays may appear incomplete; rerun HBH to regenerate outputs.

## Optimization Standalone

- The new standalone optimization scripts currently validate only by syntax; they have not yet been exercised end-to-end on real phase input payloads.
- Phase config defaults assume intermediate pickle files under `optimization_pipeline/output/`; missing prepared inputs under `optimization_pipeline/data/` will cause runtime failure until real payloads are generated.
