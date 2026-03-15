# 2026-03-12 Clutter rotation extension results

## Scope

This document fixes the result snapshot for the clutter-rotation extension.
Main frozen comparison is unchanged. The following results belong to the extension only.

- Main frozen evaluator: `experiments/configs/eval/eval_v1.json`
- Main frozen proposed protocol: `experiments/configs/refine/proposed_beam_v1.json`
- Extension config: `experiments/configs/refine/clutter_assisted_v1.json`

## Extension behavior

Only stress-added `clutter` is movable in `X1` / `X2`.
This update adds wall-aligned rotation for rectangular clutter:

- rectangular clutter: room-axis orthogonal yaw candidates only
- square clutter: no rotation
- no free-angle rotation
- no furniture movement in `X1`

## Compared protocols

- `M0`: no refine
- `M1`: furniture refine only
- `X1`: clutter refine only
- `X2`: furniture refine then clutter refine

## Input set

- stress root: `experiments/runs/stress_v2_natural_from_latest_design_frozen_20260312`
- active variants in this extension result:
  - `clutter`
  - `compound`

## Output roots

- shard runs: `experiments/runs/clutter_rotation_shards`
- merged compare root: `experiments/runs/clutter_rotation_compare_from_latest_design_frozen_20260312`

## Summary table

Source: `experiments/runs/clutter_rotation_shards/clutter_rotation_summary.csv`

| method | variant | protocol | n | core adopt | entry adopt | mean clr_feasible | mean C_vis_start | mean OOE entry surf | mean delta clutter | moved clutter |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic | clutter | M1 | 5 | 0 | 0 | 0.0149 | 0.3839 | 0.4528 | 0.0000 | 0 |
| heuristic | clutter | X1 | 5 | 2 | 0 | 0.0879 | 0.5788 | 0.6722 | 3.7281 | 5 |
| heuristic | clutter | X2 | 5 | 2 | 0 | 0.0879 | 0.5781 | 0.6722 | 3.7281 | 5 |
| heuristic | compound | M1 | 5 | 1 | 0 | 0.0334 | 0.4002 | 0.4972 | 0.0000 | 0 |
| heuristic | compound | X1 | 5 | 2 | 0 | 0.0879 | 0.5758 | 0.6722 | 3.6065 | 5 |
| heuristic | compound | X2 | 5 | 2 | 0 | 0.0879 | 0.5926 | 0.6944 | 3.6222 | 5 |
| proposed | clutter | M1 | 5 | 0 | 0 | 0.0243 | 0.3666 | 0.5028 | 0.0000 | 0 |
| proposed | clutter | X1 | 5 | 2 | 0 | 0.0879 | 0.5794 | 0.6722 | 3.8237 | 5 |
| proposed | clutter | X2 | 5 | 3 | 0 | 0.1073 | 0.5975 | 0.6500 | 3.7672 | 5 |
| proposed | compound | M1 | 5 | 1 | 0 | 0.0304 | 0.3693 | 0.5000 | 0.0000 | 0 |
| proposed | compound | X1 | 5 | 2 | 0 | 0.0879 | 0.5789 | 0.6722 | 4.0730 | 5 |
| proposed | compound | X2 | 5 | 2 | 0 | 0.0979 | 0.5978 | 0.6694 | 3.9843 | 5 |

## Interpretation

### Clutter variant

- `M1` alone does not recover any clutter case for either method.
- `X1` improves both methods to `2/5` core recovery.
- `X2` improves `proposed` further to `3/5` core recovery.
- `entry adopt` remains `0/5` for all settings.

This means clutter relocation is directly responsible for the recovered `Adopt_core` cases. Furniture-only refinement is not sufficient on these clutter-driven failures.

### Compound variant

- `M1` already recovers `1/5` for both methods.
- `X1` and `X2` both improve recovery to `2/5`.
- `X2` raises `C_vis_start` and `OOE_R_rec_entry_surf` more than `M1`.
- `entry adopt` still remains `0/5`.

This means compound failures are harder. Clutter relocation helps, but it is still not enough to pass the entry gate.

## Visualization artifacts

### Summary images

- heuristic / clutter: `experiments/runs/clutter_rotation_compare_from_latest_design_frozen_20260312/protocol_compare_summary_heuristic_clutter.png`
- heuristic / compound: `experiments/runs/clutter_rotation_compare_from_latest_design_frozen_20260312/protocol_compare_summary_heuristic_compound.png`
- proposed / clutter: `experiments/runs/clutter_rotation_compare_from_latest_design_frozen_20260312/protocol_compare_summary_proposed_clutter.png`
- proposed / compound: `experiments/runs/clutter_rotation_compare_from_latest_design_frozen_20260312/protocol_compare_summary_proposed_compound.png`

### Visual report

- `experiments/runs/clutter_rotation_compare_from_latest_design_frozen_20260312/protocol_compare_visual_report.md`

### Render manifest

- `experiments/runs/clutter_rotation_compare_from_latest_design_frozen_20260312/protocol_compare_render_manifest.json`

## Implementation note

`run_m0_m1_x1_x2_compare.py` was made robust to filtered runs that omit `M0`. When baseline rows are absent, summary fallback now reads the stress-case `metrics.json` from each variant directory.

## Current conclusion

This extension is useful and measurable.

- clutter rotation plus relocation helps `Adopt_core`
- the effect is strongest on `clutter`
- `proposed + X2` is currently the strongest setting in this extension
- the extension still does not solve `Adopt_entry`

So this belongs in appendix / extension analysis, not in the frozen main comparison.
