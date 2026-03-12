# eval_v1 freeze checklist

## Purpose

This document defines what "eval_v1 is fully frozen" means in this repository.
The goal is to ensure that every experiment run uses the same evaluation spec, with no hidden defaults and no accidental fallback to legacy config files.

## Frozen artifacts

### A. Evaluation spec

- Path: `experiments/configs/eval/eval_v1.json`
- Role: defines what counts as good / bad in evaluation
- Freeze rule: do not change thresholds, task rules, or OOE settings unless the user explicitly requests a spec update

### B. Proposed method spec

- Path: `experiments/configs/refine/proposed_beam_v1.json`
- Role: defines how the proposed refine method searches and improves layouts
- Freeze rule: treat as part of the comparison protocol; changes require explicit spec update

## Current status

As of 2026-03-11, the freeze implementation is in place.

- experiment entry scripts default to `eval_v1.json`
- `default_eval.json` has been removed from active experiment conduits
- strict evaluation mode is enabled in `eval_v1.json`
- trial-level manifests store eval path / name / SHA256
- batch-level manifests store per-case `eval_hash` and top-level `eval_hashes`
- proposed refine trials can be tied to `proposed_beam_v1.json` through `method_config_path` / `method_config_sha256`

## Completion criteria for full freeze

A run is considered to use the frozen spec only when all of the following are true.

1. All entry scripts default to `experiments/configs/eval/eval_v1.json`
2. `default_eval.json` is removed from experiment conduits
3. `eval_metrics.py` can run in `strict_eval_spec=true` mode with no hidden fallback defaults
4. Legacy keys `start_xy` / `goal_xy` have explicit semantics
5. Each trial / run manifest stores the evaluation config path and SHA256 hash
6. Proposed refine runs can also be tied back to `proposed_beam_v1.json` and/or a resolved method hash

## Current strict semantics

### strict mode

`experiments/src/eval_metrics.py` supports `strict_eval_spec=true`.
In this mode:

- required keys must exist
- hidden defaults must not be relied on
- unsupported task modes raise an error
- `start_xy` is required only when `task.start.mode == "manual"`
- `goal_xy` is required only when `task.goal.mode == "manual"`

### Legacy keys

- `start_xy`, `goal_xy` are legacy compatibility keys
- they are not the primary start/goal definition for the current experiments
- actual task points are derived from:
  - `task.start.*`
  - `task.goal.*`
- when `task.start.mode != "manual"`, `start_xy` is ignored by the evaluator logic
- when `task.goal.mode != "manual"`, `goal_xy` is ignored by the evaluator logic

## Required keys in strict mode

### Geometry / Occupancy

- `strict_eval_spec`
- `grid_resolution_m`
- `robot_radius_m`
- `clr_feasible_max_m`
- `clr_feasible_tol_m`
- `clr_feasible_max_iters`

### Task

- `task.start.mode`
- `task.start.in_offset_m`
- `task.start.door_selector.strategy`
- `task.goal.mode`
- `task.goal.offset_m`
- `task.goal.bed_selector.strategy`
- `task.goal.choose`
- `task.snap.max_radius_cells`

### Thresholds

- `tau_R`
- `tau_clr`
- `tau_clr_feasible`
- `tau_clr_astar`
- `tau_V`
- `tau_Delta`
- `lambda_rot`

### Entry observability

- `entry_observability.enabled`
- `entry_observability.mode`
- `entry_observability.sensor_height_m`
- `entry_observability.num_rays`
- `entry_observability.max_range_m`
- `entry_observability.tau_p`
- `entry_observability.tau_v`

### Adopt

- `adopt.report_both`
- `adopt.clearance_metric`
- `adopt.entry_gate.enabled`
- `adopt.entry_gate.metric`
- `adopt.entry_gate.min_value`

## Frozen parameters in eval_v1.json

### Geometry / occupancy

- `grid_resolution_m = 0.05`
- `robot_radius_m = 0.28`
- `clr_feasible_max_m = 2.0`
- `clr_feasible_tol_m = 0.01`
- `clr_feasible_max_iters = 14`
- `occupancy_exclude_categories = ["floor"]`

### Task

- `task.start.mode = entrance_slidingdoor_center`
- `task.start.in_offset_m = 0.50`
- `task.start.door_selector.strategy = largest_opening`
- `task.goal.mode = bedside`
- `task.goal.offset_m = 0.60`
- `task.goal.bed_selector.strategy = first`
- `task.goal.choose = closest_to_room_centroid`
- `task.snap.max_radius_cells = 30`

### Thresholds

- `tau_R = 0.90`
- `tau_clr = 0.10`
- `tau_clr_feasible = 0.10`
- `tau_clr_astar = 0.10`
- `tau_V = 0.60`
- `tau_Delta = 0.10`
- `lambda_rot = 0.50`

### Canonical adopt rule

- `adopt.clearance_metric = "clr_feasible"`
- `Adopt_core = 1` iff `R_reach >= tau_R`, `clr_feasible >= tau_clr_feasible`, `C_vis >= tau_V`, and `Delta_layout <= tau_Delta`
- `clr_min_astar` remains a recorded metric, but it is not the active clearance criterion for `Adopt_core` in frozen `eval_v1`

### Entry observability

- `enabled = true`
- `mode = both`
- `sensor_height_m = 0.60`
- `num_rays = 720`
- `max_range_m = 10.0`
- `tau_p = 0.05`
- `tau_v = 0.30`
- `entry_gate.metric = OOE_R_rec_entry_surf`
- `entry_gate.min_value = 0.70`

## Manifest requirements

### Trial-level

Each trial manifest should store at least:

- `eval_config_path`
- `eval_config_name`
- `eval_config_sha256`
- `resolved_eval_config`
- `resolved_method_hash`
- `method_config_path` if applicable
- `method_config_sha256` if applicable

For `method == "proposed"`, `method_config_path` should point to:

- `experiments/configs/refine/proposed_beam_v1.json`

and `method_config_sha256` should store the SHA256 of that file.

### Batch-level

Each batch manifest should store at least:

- per-case `eval_config_path`
- per-case `eval_hash`
- top-level list of `eval_hashes`

Batch manifests are not required to duplicate `proposed_beam_v1.json` hashes unless refine execution is part of that batch runner.

## Legacy config handling

- old file: `experiments/configs/eval/default_eval.json`
- current status: renamed to `experiments/configs/eval/default_eval_legacy.json`
- rule: legacy config must not be referenced from current experiment conduits

## Practical check commands

```bash
cd /Users/yuuta/Research/asset_placer_eval_standalone
rg "default_eval.json" experiments/
rg "strict_eval_spec" experiments/src/eval_metrics.py experiments/configs/eval/eval_v1.json
```

The first command should return no active experiment conduit references.
