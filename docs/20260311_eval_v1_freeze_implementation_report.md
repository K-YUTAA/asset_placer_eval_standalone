# 20260311 eval_v1 freeze implementation report

## Purpose

This report summarizes the implementation work that moved the evaluation pipeline from an "almost frozen" state to an operationally frozen `eval_v1` state.

The goal was not to change evaluation values, but to ensure that the already agreed evaluation values are the ones that actually get used everywhere.

## Scope

The freeze work covered the following areas.

1. Remove `default_eval.json` from active experiment conduits
2. Make `eval_metrics.py` capable of strict spec execution
3. Clarify legacy `start_xy` / `goal_xy` semantics
4. Store evaluation config identity in result manifests
5. Tie proposed refine runs back to `proposed_beam_v1.json`

## Implemented changes

### 1. `default_eval.json` was removed from active conduits

The legacy file was renamed:

- old: `experiments/configs/eval/default_eval.json`
- new: `experiments/configs/eval/default_eval_legacy.json`

Active defaults were switched to:

- `experiments/src/run_generate_and_eval.py`
- `experiments/src/run_pipeline_from_json.py`
- pipeline config JSONs under `experiments/configs/pipeline/`

All active experiment conduits now default to:

- `experiments/configs/eval/eval_v1.json`

### 2. `eval_v1.json` now enables strict evaluation mode

`experiments/configs/eval/eval_v1.json` now includes:

- `"strict_eval_spec": true`

This makes the intended frozen config explicit at the config-file level rather than relying on convention only.

### 3. `eval_metrics.py` now supports strict spec validation

`experiments/src/eval_metrics.py` was updated so that when a config file contains:

- `strict_eval_spec = true`

it behaves as follows.

- required keys must exist
- unsupported task modes raise an error
- hidden fallback defaults are not used
- `start_xy` is required only when `task.start.mode == "manual"`
- `goal_xy` is required only when `task.goal.mode == "manual"`

This means a partially specified config can no longer silently run under the frozen-spec path.

### 3.5. The canonical `Adopt_core` clearance rule is now explicit

The active frozen implementation uses:

- `adopt.clearance_metric = "clr_feasible"`
- `tau_clr_feasible = 0.10`

So, in frozen `eval_v1`, `Adopt_core` is evaluated against `clr_feasible`, not against a legacy `clr_min_m = 0.25` proposal.

`clr_min_astar` is still computed and saved, but it is not the clearance field used for the frozen adoption decision.

### 4. Legacy `start_xy` / `goal_xy` semantics were documented

The current evaluator behavior is now explicitly documented.

- `start_xy` / `goal_xy` are legacy compatibility keys
- current experiments derive task points from `task.start.*` and `task.goal.*`
- non-manual task modes do not use legacy XY values as the primary definition

This was documented in:

- `docs/20260309_eval_v1_freeze_checklist.md`

### 5. Trial manifests now save eval config identity

`experiments/src/run_trial.py` was updated to store the following in trial-level outputs.

- `eval_config_path`
- `eval_config_name`
- `eval_config_sha256`
- `resolved_eval_config`
- `resolved_method_hash`
- `method_config_path`
- `method_config_sha256`

It also appends the following to trial CSV rows.

- `eval_config_path`
- `eval_config_name`
- `eval_hash`
- `method_hash`

For `method == "proposed"`, the manifest also ties the run back to:

- `experiments/configs/refine/proposed_beam_v1.json`

### 6. Generate-and-eval runs now save eval identity

`experiments/src/run_generate_and_eval.py` now writes:

- `run_manifest.json`

with:

- `eval_config_path`
- `eval_config_name`
- `eval_config_sha256`

### 7. Batch manifests now save eval identity

`experiments/src/run_pipeline_from_json.py` now stores:

- per-case `eval_config_path`
- per-case `eval_config_name`
- per-case `eval_hash`
- top-level `eval_hashes`

This makes it possible to recover which evaluation spec was used for any batch result.

## Verification performed

Verification was intentionally limited to local, non-API checks.

### A. strict mode success check

Command:

```bash
cd /Users/yuuta/Research/asset_placer_eval_standalone
uv run python experiments/src/eval_metrics.py \
  --layout experiments/runs/batch_v2_gpt_high_e2e_rerun_20260302/komazawakoen_B1_30_5.1x5.88_v2/layout_generated.json \
  --config experiments/configs/eval/eval_v1.json \
  --out /tmp/eval_v1_strict_ok_metrics.json \
  --debug_dir /tmp/eval_v1_strict_ok_debug
```

Result:

- completed successfully
- produced metrics and debug outputs

### B. strict mode failure check

A temporary config was created with `strict_eval_spec=true` but without `tau_V`.

Result:

- evaluation failed immediately with:
  - `ValueError: strict_eval_spec requires missing key(s) ... ['tau_V']`

This confirms that strict mode is not falling back to hidden defaults.

### C. trial manifest hash check

`run_trial.py` was executed without any API call by using an existing `layout_generated.json` through `layout_input`.

Result:

- `trial_manifest.json` contained:
  - `eval_config_path`
  - `eval_config_name`
  - `eval_config_sha256`
  - `resolved_method_hash`
- observed eval hash:
  - `f9db2ab39a726b4055783e7dd3b706d3f6806814c8cfaee17aa993f270d54aa1`

### D. batch manifest hash check

`run_pipeline_from_json.py` was checked using a dry-run style invocation against an existing case directory.

Result:

- `batch_manifest.json` contained:
  - top-level `eval_hashes`
  - per-case `eval_config_path`
  - per-case `eval_config_name`
  - per-case `eval_hash`

## Files changed

### Code

- `experiments/src/eval_metrics.py`
- `experiments/src/run_trial.py`
- `experiments/src/run_generate_and_eval.py`
- `experiments/src/run_pipeline_from_json.py`

### Config

- `experiments/configs/eval/eval_v1.json`
- `experiments/configs/eval/default_eval_legacy.json`
- pipeline configs under `experiments/configs/pipeline/`

### Docs

- `docs/20260309_eval_v1_freeze_checklist.md`
- `docs/20260308_two_week_implementation_plan.md`
- `docs/20260311_eval_v1_freeze_implementation_report.md`

## Operational conclusion

At this point, `eval_v1` is operationally frozen for the current experiment pipeline.

That means the following are now true.

- active experiment entry points default to `eval_v1.json`
- strict spec execution is available and enabled in `eval_v1.json`
- hidden fallback defaults are blocked on the strict path
- result manifests can identify the exact evaluation config by path and SHA256
- proposed refine trials can be traced back to `proposed_beam_v1.json`
- the canonical `Adopt_core` clearance criterion is `clr_feasible >= 0.10`

## Remaining boundary

The remaining rule is operational rather than technical.

- do not edit `eval_v1.json` or `proposed_beam_v1.json` unless the user explicitly requests a spec update

That is now a process constraint, not an implementation gap.
