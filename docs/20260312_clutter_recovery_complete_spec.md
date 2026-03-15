# 2026-03-12 Clutter Recovery Complete Spec

## Scope

This document is the complete specification for clutter-assisted recovery around `clutter` objects.
It supersedes the narrower notes in:

- `docs/20260312_clutter_assisted_recovery_extension_plan.md`
- `docs/20260312_m0_m1_x1_x2_clutter_recovery_policy.md`

Main comparison remains frozen. This document only specifies the extension protocols that operate on stress-added clutter.

## Frozen artifacts that remain unchanged

- Evaluator: `experiments/configs/eval/eval_v1.json`
- Main proposed method protocol: `experiments/configs/refine/proposed_beam_v1.json`
- Main comparison protocols remain layout-only.

## Protocols

### Protocol M0
- No refinement.
- Input stress layout is evaluated as-is.

### Protocol M1
- Layout-only furniture refinement.
- Only movable furniture is updated.
- Stress-added clutter remains fixed.

### Protocol X1
- Clutter-assisted recovery.
- Only stress-added clutter is movable.
- Existing furniture is fixed.

### Protocol X2
- Sequential recovery.
- First run `M1`, then run `X1` on the refined layout.

## Clutter definition

A clutter object is an externally added obstacle introduced by stress generation.
It is not part of the original room layout.

Required metadata on clutter objects:
- `origin = "stress_added"`
- `object_role = "clutter"`
- `movable_in_main = false`
- `movable_in_clutter_recovery = true`

## Clutter recovery v1.1 behavior

### Allowed operations
- Translation inside the room
- Rotation only for rectangular clutter

### Disallowed operations
- Deletion
- Moving clutter outside the room
- Free-angle rotation
- Moving original furniture during `X1`

## Rotation rule for clutter

### Rectangular clutter
Rectangular clutter may be reoriented only to room-axis orthogonal directions:
- `theta_room + 0°`
- `theta_room + 90°`
- `theta_room + 180°`
- `theta_room + 270°`

No off-axis exceptions are used for clutter.

### Square clutter
Square clutter is treated as rotation-invariant and keeps its original yaw.
A clutter object is treated as square when the size difference between length and width is within `square_tolerance_m`.

## Objective

Clutter recovery does not try to preserve the original clutter pose.
The primary objective is improvement of evaluation metrics under valid occupancy constraints.

Priority order:
1. `validity`
2. `Adopt_entry`
3. `Adopt_core`
4. `clr_feasible`
5. `C_vis_start`
6. `OOE_R_rec_entry_surf`
7. `R_reach`
8. `C_vis`
9. `delta_layout_clutter` as a weak tie-breaker

## Feasibility checks

All candidate clutter moves are validated using the same occupancy-grounded rules as the evaluator:
- remain inside room boundary
- no wall crossing
- no overlap with existing objects beyond threshold
- preserve door keep-out radius
- preserve valid start/goal geometry

## Logging

The extension must report clutter movement separately from furniture movement.
Required fields include:
- `recovery_protocol`
- `delta_layout_furniture`
- `delta_layout_clutter`
- `moved_furniture_count`
- `moved_clutter_count`
- `sum_clutter_displacement_m`
- `max_clutter_displacement_m`

## Intended analysis

Main text should keep the frozen layout-only comparison.
Clutter-assisted recovery is an extension analysis, primarily interpreted on:
- `clutter`
- `compound`

`base` and `usage_shift` remain sanity checks.

## Implementation files

- `experiments/src/refine_clutter_recovery.py`
- `experiments/src/run_trial.py`
- `experiments/configs/refine/clutter_assisted_v1.json`

