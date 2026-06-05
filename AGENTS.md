# AGENTS.md

Operational instructions for autonomous coding agents working in this repository.

## Repository Shape
- This is a Python 3.9 motion-processing pipeline driven by `main.py`.
- `main.py` loads YAML with `config_loader.load_config`, applies `--stage` overrides, and calls `pipeline.run_pipeline`.
- `pipeline.py` always runs `preprocess` first, then conditionally runs `pose`, `fusion`, `learnable`, `evaluation`, and `visualization`; `refinement` is a separate stage path.
- Runtime configuration lives primarily in `configs/pipeline.yml`; parameter tables live in `configs/parameters.yaml`; SMPL joint mapping lives in `configs/keypoints_map.yml`.
- Each pipeline package follows the same shape: `stage.py` re-exports the public runner, `executor.py` contains implementation, `config.py` contains constants, and `logs.py` contains log helpers.
- Shared helpers are in `json_io.py`, `keypoints_map.py`, `compat.py`, and `config_loader.py`.
- `slahmr/` and `refinement_pipeline/third_party_modified/ipman/` are vendored or external-derived code. Avoid changing them unless the task explicitly targets those integrations.

## Stage Responsibilities
- `preprocess_pipeline`: extracts frames from MP4 inputs with ffmpeg/ffprobe, computes camera offset with the `paper` method, writes offset text files, and mutates `config["runtime"]` with `selected_offset` and `offset_method`.
- `pose_pipeline`: reads WHAM/OpenCap PKLs, uses SMPL to export per-frame 3D joints under `output/pose_results/keypoints3d` and metadata under `output/pose_results/metadata`.
- `fusion_pipeline`: loads pose JSON plus WHAM mesh vertices, performs occlusion-aware two-camera fusion and constrained optimization, and writes split `keypoints3d`/`metadata` outputs.
- `learnable_pipeline`: fits SMPL pose/translation to fused targets with torch/smplx and writes learnable keypoints plus metadata.
- `evaluation_pipeline`: compares pose/fusion/learnable outputs to `input/gtruth_results` and writes PA-MPJPE CSVs plus `summary.json`.
- `visualization_pipeline`: renders comparison and projection MP4s with matplotlib, OpenCV, and ffmpeg.
- `refinement_pipeline`: runs the standalone optimization flow using WHAM data, subject parameters, camera intrinsics, torch, and modified third-party utilities.

## Coding Conventions
- Prefer the existing package pattern: public stage function in `stage.py`, implementation in `executor.py`, constants in `config.py`, logging text in `logs.py`.
- Use `pathlib.Path` for new path handling unless integrating with existing code that already uses `os.path`.
- Use `json_io.read_json` and `json_io.write_json` for pipeline JSON so NumPy scalars and arrays serialize consistently.
- Preserve the split output schema: `keypoints3d/` contains camera joint dictionaries, `metadata/` contains run details. Keep frame filenames 1-based and prefix-compatible (`pose_data_`, `fused_data_`, `learnable_frame_`).
- Keep keypoint names stable across pose, fusion, learnable, evaluation, and visualization. Update mapping, aliases, and consumers together.
- Add new config keys to `configs/pipeline.yml` and update `config_loader.validate_config` when they are required.
- If adding or renaming stages, update both `main.py` CLI choices and `config_loader.ALLOWED_STAGES`.
- Keep compatibility patches centralized in `compat.py`; do not scatter NumPy/chumpy/Python-version monkey patches through stage code.
- Keep changes scoped. Do not refactor vendored code, generated outputs, or unrelated pipeline stages as part of a local fix.

## Workflow Rules
- Start each task by reading `AGENTS.md` and the files relevant to the requested stage.
- Read `conversation_cache/current_status.md`, `conversation_cache/decisions.md`, and `conversation_cache/todo.md` when present; update them before ending any substantial session.
- Put only stable, long-term instructions in `AGENTS.md`. Put transient progress, decisions, blockers, and next steps in `conversation_cache/`.
- Treat source code and YAML as authoritative when cache notes are stale; update the cache rather than preserving stale memory.
- Do not run `setup.sh` automatically. It installs packages and downloads external data.
- Do not run the full pipeline casually. Default config has `runtime.clean_output: true`, and stages delete matching files in their output directories before writing new results.
- Use a temporary copied config with `runtime.clean_output: false` for exploratory runs when preserving current outputs matters.
- Before changing data schemas, inspect all downstream consumers in later stages and visualization/evaluation.
- Before final response, report commands run and any checks that were skipped or could not run.

## Persistence Workflow
- Before coding, read `conversation_cache/current_status.md` and `conversation_cache/decisions.md`; also check `todo.md`, `known_issues.md`, `edge_cases.md`, and `datasets.md` when the task touches their scope.
- After coding, update `conversation_cache/current_status.md`, append important architectural decisions to `conversation_cache/decisions.md`, and update `conversation_cache/todo.md`.
- If session context becomes inconsistent, reread `AGENTS.md`, reread `conversation_cache/*`, and verify the current architecture before modifying code.
- Keep `AGENTS.md` stable and operational. Put session logs, temporary progress, blockers, TODOs, and session summaries in `conversation_cache/`.

## Memory File Update Format
- `conversation_cache/current_status.md`: latest session summary, current task state, blockers, verification results, and working-tree notes.
- `conversation_cache/decisions.md`: durable architectural decisions and the reasoning behind them.
- `conversation_cache/todo.md`: active, deferred, and completed task checklist items.
- `conversation_cache/known_issues.md`: recurring bugs, fragile behavior, environment problems, and known limitations.
- `conversation_cache/edge_cases.md`: parser, schema, stage-ordering, and pipeline edge cases agents should preserve.
- `conversation_cache/datasets.md`: dataset paths, input/output assumptions, expected file formats, model/checkpoint locations, and benchmark notes.

## Testing And Checks
- Use the project Python environment with `requirements.txt` installed. The commands below assume `python` resolves to that environment; if imports fail with missing dependencies, verify the interpreter before treating it as a code failure.
- Syntax check:
  ```bash
  python -m compileall main.py config_loader.py pipeline.py compat.py json_io.py keypoints_map.py preprocess_pipeline pose_pipeline fusion_pipeline learnable_pipeline evaluation_pipeline visualization_pipeline refinement_pipeline
  ```
- Config validation check:
  ```bash
  python -c 'from config_loader import load_config; load_config("configs/pipeline.yml"); print("config ok")'
  ```
- Import smoke check:
  ```bash
  python -c 'import main, pipeline; from config_loader import load_config; from pose_pipeline.stage import run_pose_export; from fusion_pipeline.stage import run_fusion; from learnable_pipeline.stage import run_learnable_smplify; from evaluation_pipeline.stage import run_evaluation; from visualization_pipeline.stage import run_visualization; print("imports ok")'
  ```
- Manual stage run:
  ```bash
  python main.py --config configs/pipeline.yml --stage pose
  ```
- Full demo run:
  ```bash
  python main.py --config configs/pipeline.yml
  ```
- Stage and full runs require local demo data, SMPL model files, ffmpeg/ffprobe, and installed Python dependencies from `requirements.txt`.
- In restricted sandboxes, matplotlib may need a writable cache directory such as `MPLCONFIGDIR=/tmp/matplotlib`.

## Dangerous Areas
- `output/` is generated and stage cleanup can delete existing JSON, CSV, image, and MP4 outputs.
- `models/` contains large downloaded model/checkpoint files. Do not modify, re-download, or commit replacements unless explicitly requested.
- `input/*.pkl`, `input/*.pickle`, and MP4 inputs are large runtime data. `input/gtruth_results/` is tracked fixture data and should not be regenerated without a specific evaluation-data task.
- `preprocess_pipeline.executor` shells out to ffmpeg/ffprobe. Quote all user-controlled paths and avoid widening shell usage.
- `fusion_pipeline.core` contains numerical optimization and fallback behavior. Small changes can alter downstream learnable/evaluation results.
- `learnable_pipeline` and `refinement_pipeline` can be slow and GPU-sensitive. Preserve CPU fallback behavior.
- `visualization_pipeline` depends on generated frame images, ffmpeg, OpenCV, and matplotlib's non-interactive backend.
- `slahmr/` and `refinement_pipeline/third_party_modified/ipman/` are external-derived code; isolate local adapter changes outside them where possible.
- `.gitignore` excludes important generated/runtime paths (`models/`, `output/`, PKLs, MP4s, caches). Check `git status --short` before and after work.

## Language

- Respond to the user in Vietnamese by default.
- Explain technical concepts in Vietnamese.