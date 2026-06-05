# Current Status

Last updated: 2026-06-05 14:46 +07

## WHAM 2D Confidence Extraction And Fusion Confidence Blend

- Added `preprocess_pipeline/extract_2d.py`.
- The new preprocess extractor reads WHAM/OpenCap PKLs with `joblib.load()` because the files are `collections.defaultdict` payloads, not plain `pickle.dump()` files.
- It extracts `tracking_results_for_reproj` fields:
  - `frame_id`
  - `keypoints`
  - `init_betas`
- It maps WHAM 133-keypoint rows into the project schema with `[x, y, confidence]` values:
  - neck, shoulders, elbows, hands, hips, knees, ankles, feet
- It updates camera profile JSONs:
  - `output/preprocess_results/data_cam1.json`
  - `output/preprocess_results/data_cam2.json`
- New camera profile fields:
  - `2D_camera_cam1` / `2D_camera_cam2`
  - `shape`
  - `init_betas`
- Fusion now loads these 2D camera confidence profiles and aligns confidence by `metadata.source_frame_indices` from pose metadata, with fallback to `frame_idx - 1` / `frame_idx`.
- `fusion_pipeline/detection.py` now computes final `H1_all`/`H2_all` by harmonic-blending:
  - old geometric/visibility confidence
  - WHAM 2D keypoint confidence
- `K1` and `K2` are computed from the new blended `H1_all`/`H2_all`.

## Verification

- Passed:
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -m compileall preprocess_pipeline fusion_pipeline`
  - direct `export_tracking_2d_to_camera_profiles(load_config("configs/pipeline.yml"))`
  - camera profile schema smoke check for `2D_camera_cam1`, `2D_camera_cam2`, and shape length `(1, 10)`
  - fusion confidence lookup smoke check from camera profiles
  - detection blend smoke check showing low camera1 2D confidence and high camera2 2D confidence moves a joint into `K2`
  - full compile checklist for main modules
  - config validation
  - import smoke checklist

Last updated: 2026-06-05 12:57 +07

## Fusion Debug Evaluation And Visualization

- Evaluation now discovers fusion debug outputs under:
  - `output/fused_results/debug1`
  - `output/fused_results/debug2`
- When those directories exist, evaluation treats them as modules:
  - `fusion_debug1`
  - `fusion_debug2`
- Verified evaluation generated:
  - `pa_mpjpe_fusion_debug1_cam1.csv`
  - `pa_mpjpe_fusion_debug1_cam2.csv`
  - `pa_mpjpe_fusion_debug2_cam1.csv`
  - `pa_mpjpe_fusion_debug2_cam2.csv`
- Visualization project videos remain unchanged with the original three columns:
  - pose
  - fusion
  - learnable
- Fusion debug projection is exported as a separate video when debug JSON exists:
  - `project_camera1_fusion_debug1_fusion_debug2_<timestamp>.mp4`
  - `project_camera2_fusion_debug1_fusion_debug2_<timestamp>.mp4`
- Project 2D visualization now honors `visualization.max_frames`.

## Verification

- Passed:
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -m compileall evaluation_pipeline visualization_pipeline`
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -c 'from evaluation_pipeline.stage import run_evaluation; from visualization_pipeline.stage import run_visualization; print("eval/vis imports ok")'`
  - direct `run_evaluation(load_config("configs/pipeline.yml"))`
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -c 'import main, pipeline; from config_loader import load_config; from evaluation_pipeline.stage import run_evaluation; from visualization_pipeline.stage import run_visualization; print("imports ok")'`
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -c 'from config_loader import load_config; load_config("configs/pipeline.yml"); print("config ok")'`
- Render smoke for one project frame reached ffmpeg but failed due to environment ffmpeg symbol error:
  - `ffmpeg: undefined symbol: avio_print_string_array, version LIBAVFORMAT_58`
  - This is an environment/runtime ffmpeg issue, not a Python import/logic error.

Last updated: 2026-06-05 12:35 +07

## Fusion Core Removal And Detection Ownership

- Removed `fusion_pipeline/core.py`.
- Added `fusion_pipeline/pipeline.py` as the frame-level fusion orchestrator.
- Updated `fusion_pipeline/executor.py` to import:
  - `run_phase3_pipeline` from `fusion_pipeline.pipeline`
  - `load_torso_mask` and `make_raw_judgement_fallback` from `fusion_pipeline.detection`
- Moved occlusion visibility logic out of `fusion_pipeline/geometry.py` and into `fusion_pipeline/detection.py`.
- `fusion_pipeline/geometry.py` now only contains low-level geometry helpers for this stage:
  - `_as_xyz`
  - `get_orientation_flag`
  - `ROTATION_PARENT_JOINTS`

## Consolidated Root Cache Notes

- Root-level `current_status.md` was consolidated into this file. It recorded preprocess calibration cleanup:
  - removed redundant preprocess config keys from `pipeline.yml`
  - dynamic loading of `camera_*.calibration`
  - fallback to estimated intrinsics when calibration is missing/incomplete
- Root-level `a.md` was consolidated here as an operational reminder:
  - reread `AGENTS.md` and `conversation_cache` before continuing
  - persist project state before ending substantial sessions

Last updated: 2026-06-05 12:23 +07

## Fusion Refactor And M-Set Semantics

- Updated fusion rotation-mismatch detection:
  - `get_orientation_flag()` no longer classifies every joint by `dot(pos - torso_center, forward_vec)`.
  - It now computes the sign from each configured joint's parent vector:
    - hand/wrist from shoulder to hand/wrist
    - shoulder from hip to shoulder
    - knee from hip to knee
    - ankle from knee to ankle
  - Non-target joints return `0`, so `M` now represents cross-camera limb rotation-direction disagreement, not generic before/after position relative to the torso plane.
- Split `fusion_pipeline/core.py` into clearer functional modules:
  - `detection.py`: cross-view error grouping, harmonic precision, fallback judgement
  - `correction.py`: RANSAC/Umeyama similarity and large keypoint replacement/correction
  - `optimization.py`: stats/losses, bone constraints, SLSQP fine-tuning
  - `core.py`: frame-level orchestration and compatibility re-exports

## Verification

- Passed:
  - `python -m compileall fusion_pipeline`
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -c 'from fusion_pipeline.stage import run_fusion; from fusion_pipeline.core import run_phase3_pipeline, make_raw_judgement_fallback, optimize_f_points, ransac_umeyama; from fusion_pipeline.geometry import get_orientation_flag; print("fusion imports ok")'`
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -m compileall main.py config_loader.py pipeline.py compat.py json_io.py keypoints_map.py preprocess_pipeline pose_pipeline fusion_pipeline learnable_pipeline evaluation_pipeline visualization_pipeline refinement_pipeline`
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -c 'import main, pipeline; from config_loader import load_config; from pose_pipeline.stage import run_pose_export; from fusion_pipeline.stage import run_fusion; from learnable_pipeline.stage import run_learnable_smplify; from evaluation_pipeline.stage import run_evaluation; from visualization_pipeline.stage import run_visualization; print("imports ok")'`
  - `/home/thanh/miniconda3/envs/easymocap/bin/python -c 'from config_loader import load_config; load_config("configs/pipeline.yml"); print("config ok")'`
- Default `python` import smoke still fails because that interpreter lacks NumPy; project env Python works.
- Full fusion stage was not run because it depends on local pose/WHAM runtime data and may clean generated outputs.

Last updated: 2026-06-03 00:00 +07

## Completed In This Session

- Removed obsolete stage names from the active validator:
  - `pose_fusion`
  - `postprocess`
  - `learnable_visualization`
- Cleaned up visualization output naming so project videos no longer use the legacy `pose_fusion` / `learnable` combined label.
- Verified repository search no longer finds runtime references to the removed stage names outside cached notes.

- Removed `hbh_pipeline` from the active system:
  - deleted the `hbh_pipeline/` package
  - removed the `hbh` stage from `main.py` CLI choices
  - removed the `hbh` execution branch from `pipeline.py`
  - removed `hbh` from `config_loader.ALLOWED_STAGES`
  - removed `paths.hbh_output_dir` and the `hbh:` config block from `configs/pipeline.yml`
- Verified no repository code paths still reference `hbh_pipeline` or `runtime.stage == "hbh"`.
- Ran a compile/syntax smoke check after the removal:
  - `python -m compileall main.py config_loader.py pipeline.py compat.py json_io.py keypoints_map.py preprocess_pipeline pose_pipeline fusion_pipeline learnable_pipeline evaluation_pipeline visualization_pipeline refinement_pipeline`

Last updated: 2026-06-02 10:02 +07

## Completed In This Session

- Refactored standalone `optimization_pipeline` entrypoints for phases 2-6:
  - added shared config file `optimization_pipeline/configs.yml`
  - added five self-contained scripts with no local-module imports:
    - `phase2_init_translation.py`
    - `phase3_smooth_poses.py`
    - `phase4_mean_shapes.py`
    - `phase5_init_rt.py`
    - `phase6_refine_poses.py`
- Config-driven standalone phase I/O is now defined in `optimization_pipeline/configs.yml`:
  - each phase reads its own `inputs`, `model`, optimization parameters, and `output`
  - default intermediate payloads are chained through `optimization_pipeline/output/*.pkl`
- Standalone phase behavior implemented:
  - phase 2 loads params/keypoints/cameras + SMPL/J-regressor, estimates `Th`
  - phase 3 smooths `params['poses']`
  - phase 4 averages `params['shapes']`
  - phase 5 optimizes `Th` and `Rh` with reprojection + temporal smoothness
  - phase 6 optimizes `poses`, `Rh`, `Th` with robust reprojection + temporal smoothness + init-anchor + GMM prior
- Input robustness added for standalone scripts:
  - params can be provided either batched or single-frame / single-shape and will be expanded to frame count when safe
  - camera payload accepts framewise or constant `K/R/T`

- Re-read repository operating docs before continuing:
  - `AGENTS.md`
  - `conversation_cache/current_status.md`
  - `conversation_cache/decisions.md`
  - `conversation_cache/todo.md`
  - `conversation_cache/known_issues.md`
  - `conversation_cache/edge_cases.md`
  - `conversation_cache/datasets.md`
- Reviewed newly added `optimization_pipeline/` and `optimization_pipeline/hrnet_pare_finetune.yml` to map active optimization logic:
  - Multi-stage final optimization flow in YAML:
    1. load SMPL model
    2. initialize translation from 2D keypoints
    3. temporal smoothing on initial pose
    4. mean shape sharing
    5. `init_RT` optimization on `[Th, Rh]` with reprojection + smooth losses
    6. `refine_poses` optimization (repeat=2) on `[poses, Rh, Th]` with reprojection + smooth + init-anchor + GMM prior losses
  - Confirmed optimizer behavior from code:
    - LBFGS (`strong_wolfe`) plus early-stop by relative loss change and NaN/Inf guard.
  - Confirmed loss implementations used by YAML:
    - `Keypoints2D` (confidence-weighted reprojection),
    - `Smooth` (multi-window second-order temporal finite differences with `Linear/Depth` modes),
    - `Init` (anchor to initialization),
    - `GMMPrior` (CMU mixture prior on pose vector).
  - Integration note from current project state:
    - `main.py` CLI and `pipeline.py` runtime path still do not include an explicit `optimization` stage hook.

- Added dedicated `preprocess` stage option to CLI/config validation:
  - `main.py` stage choices include `preprocess`
  - `config_loader.ALLOWED_STAGES` includes `preprocess`
- Updated preprocess camera export schema and intrinsics handling:
  - `data_cam*.json` now stores `intrinsics_estimation` in addition to `intrinsics_cam`
  - Added YAML switch `preprocess.calibration.intrinsics_source: intri_cam | intri_esti`
- Renamed translation field in camera profile output:
  - `xyz` -> `tvec` (with HBH fallback support for legacy `xyz`)
- Corrected HBH projection model for this calibration format:
  - Uses `[R | tvec]` (not `-R@t`)
- Removed `hbh_pipeline/execute.py` and migrated required logic into `hbh_pipeline/executor.py`
  - detector wrapper
  - COCO->H36M conversion
  - triangulation methods and method registry
- HBH flow now includes by default:
  1. method-wise 3D generation
  2. method-wise PA-MPJPE evaluation CSVs
  3. 5-column projection visualization video
- HBH output structure:
  - `output/hbh_results/method_<method>/hbh_data_<frame>.json`
  - `output/hbh_results/pa_mpjpe_hbh_<method>.csv`
  - `output/hbh_results/hbh_project_5col.mp4`
- HBH visualization improvements:
  - one multi-column projection video only (no compare video)
  - method-specific title and PA-MPJPE text
  - enlarged header text and moved to top panel (not overlaying frame content)
  - white text with black outline for readability

## Verification

- Compile checks passed across modified modules, including:
  - `hbh_pipeline/executor.py`
  - `hbh_pipeline/visualization.py`
  - `hbh_pipeline/evaluation.py`
  - `preprocess_pipeline/calib.py`
  - `main.py`, `config_loader.py`
- Standalone optimization phase scripts compile check passed:
  - `python -m compileall optimization_pipeline/phase2_init_translation.py optimization_pipeline/phase3_smooth_poses.py optimization_pipeline/phase4_mean_shapes.py optimization_pipeline/phase5_init_rt.py optimization_pipeline/phase6_refine_poses.py`

## Remaining Runtime Validation

- The new standalone optimization scripts have not been executed end-to-end on real phase input files yet.
- Runtime for phases 2, 5, and 6 still depends on an environment where `smplx` is installed and the configured SMPL/J-regressor paths exist.
- End-to-end HBH run should be re-executed after the latest text/layout and geometry changes to regenerate outputs and verify final visual quality.

## Learnable-SMPLify Logic Verification (This Session)

- Reviewed `learnable_pipeline/executor.py` against bundled `learnable_pipeline/Learnable-SMPLify/src` code:
  - `module/net_body25.py`
  - `inference.py`
  - `common/keypoint_geo.py`
  - `config/net.yaml`
- Verified current integration follows official NetBody25 formulation:
  - Builds `J_init_body25` from WHAM SMPL pose/trans/betas via SMPL + `J_regressor_body25.npy`
  - Builds `J_target_body25` from fused joints and fills missing joints from `J_init_body25`
  - Uses human-centric normalization with `R,T` estimated from init joints and reused for target joints
  - Runs `net.predict(...)` with the same feature pathway (start pose + betas + keypoint-pair delta cues)
  - Exports refined joints in project 15-joint schema.
- Identified one key logic mismatch vs sequential inference design:
  - Current code performs per-frame independent inference (same-frame init->target) rather than sequence transition inference (`t-s -> t`) used by official `inference.py`.
- Tooling limitation:
  - Could not inspect bundled PDF content because `pdftotext` is unavailable and no PDF parser package is installed in current environment.

## Learnable-SMPLify Compatibility Fix (This Session)

- Updated `learnable_pipeline/executor.py` to match current pipeline config shape while preserving the existing output schema.
- Added backward-compatible config defaults for projects that still use `learnable` block:
  - auto default `repo_src` to `learnable_pipeline/Learnable-SMPLify/src`
  - auto default `net_config` to `learnable_pipeline/Learnable-SMPLify/src/config/net.yaml`
  - auto default `smpl_family_dir` to `models`
  - if missing, `checkpoint` falls back to `learnable.checkpoint`, then `models/best_ckpt.pth.tar`
- Removed duplicated per-camera `inference_mode` re-read in the camera loop.
- Verified syntax by compile check:
  - `python -m compileall learnable_pipeline/executor.py` passed.

## Learnable Runtime + Cleanup (This Session)

- Fixed CPU runtime incompatibility from upstream Learnable-SMPLify hardcoded CUDA calls by patching at import-time:
  - `Unit_GCN.forward` now uses `to(x.device)` for adjacency tensor.
  - `rot6d_to_axis_angle` no longer calls `.cuda()` and allocates tensors on the input device.
- Added automatic model asset path compatibility to match existing `pipeline.yml`:
  - If missing, copy `paths.j_regressor_body25` to `models/smpl/J_regressor_body25.npy`.
  - If missing, copy `paths.smpl_model` (and optional explicit male/female paths) into `models/smpl/SMPL_*.pkl`.
- Updated learnable input contract:
  - removed dependency on `fused_output_dir/metadata/fused_data_*.json`.
  - now reads only `fused_output_dir/keypoints3d/fused_data_*.json` for inference inputs.
- Pruned bundled `learnable_pipeline/Learnable-SMPLify` to inference-required files:
  - removed docs/assets/train/eval/dataset entrypoints.
  - kept `src/common`, `src/module`, and `src/config/net.yaml`.

## Offset Calculation Cleanup (This Session)
- Removed `offset_colab.py` logic from `preprocess_pipeline`.
- Updated `preprocess_pipeline` to exclusively and by default use `offset_paper` for camera sync offsets.
- Removed `offset_colab` variables and output keys from `data_cam*.json` exports in `calib.py`.
- Removed `colab` option from `configs/pipeline.yml` and `preprocess_pipeline/config.py`.

## Preprocess Calibration Refactor (This Session)
- Removed `calib_cameraIntrinsics.pickle.py` by integrating its logic directly into `preprocess_pipeline/calib.py`.
- Replaced monolithic `camera.calibration` input with individual per-camera inputs (`camera_{cam_id}.calibration`).
- Automatically infer `intrinsics_source` ("intri_cam" or "intri_esti") based on the existence of the camera calibration file.
- `export_camera_jsons` now simultaneously outputs `data_{cam_id}.json` and `cameraIntrinsics_{cam_id}.pickle`.
- Removed `camera_name_map`, `intrinsics_source` and `file` parameters from `configs/pipeline.yml` calibration block.
