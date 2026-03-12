# 2026-03-12 frozen latest-design refine compare

## Scope
- Frozen upstream generation run: `experiments/runs/batch_v2_gpt_high_latest_design_frozen_20260312`
- Frozen natural stress run: `experiments/runs/stress_v2_natural_from_latest_design_frozen_20260312`
- Refine compare run root: `experiments/runs/refine_compare_from_latest_design_frozen_20260312`
- Cases: 5 base layouts
- Stress variants per case: `base`, `usage_shift`, `clutter`, `compound`
- Refine methods: `heuristic`, `proposed`
- Total refine trials: 40

## Frozen specs used
- Latest design config: `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`
  - sha256: `9c6baaece9c6a58c4222bf3b7434a180cceb941adf9cc61b40e779995b4b31a0`
- Eval config: `experiments/configs/eval/eval_v1.json`
  - sha256: `f9db2ab39a726b4055783e7dd3b706d3f6806814c8cfaee17aa993f270d54aa1`
- Stress config: `experiments/configs/stress/stress_v2_natural.json`
  - sha256: `845f491480fc2a6c0dec1660395edaa3680abc443c79697c3a84e2a38b6aeb0d`
- Proposed config: `experiments/configs/refine/proposed_beam_v1.json`
  - sha256: `677cf625b600571c5e205c6327b7bbe242db978f35f3e7a8e373f00c9cebe416`
- Prompt snapshot dir: `prompts/latest_design_frozen_20260312`

## Upstream generation conditions
- Step1 model: `gpt-5.2`
- Reasoning effort: `high`
- Text verbosity: `high`
- Image detail: `high`
- Spatial understanding: enabled
- Gemini model: `gemini-3-flash-preview`
- Gemini temperature: `0.6`
- Gemini label language: `English`
- Gemini task: `boxes`

## Stress generation conditions
- Generator family: `stress_v2_natural`
- Variants: `base`, `usage_shift`, `clutter`, `compound`
- Eval during generation: `eval_v1.json`
- QA result: `exists 20/20`, `complete 20/20`, `explainable 20/20`

## Refine conditions
### heuristic
```json
{
  "refine_max_iterations": 30,
  "refine_step_m": 0.1,
  "refine_rot_deg": 15.0,
  "refine_max_changed_objects": 3
}
```

### proposed
```json
{
  "refine_step_m": 0.1,
  "refine_rot_deg": 15.0,
  "refine_max_changed_objects": 3,
  "refine_beam_width": 5,
  "refine_depth": 3,
  "refine_candidate_objects_per_state": 2,
  "refine_eval_budget": 780,
  "refine_ooe_primary": "OOE_R_rec_entry_surf",
  "refine_use_lexicographic": true,
  "refine_allow_intermediate_regression": true,
  "refine_door_keepout_radius_m": 0.0,
  "refine_overlap_ratio_max": 0.05,
  "refine_delta_weight": 0.3
}
```

## Output roots
- Heuristic run root: `experiments/runs/refine_compare_from_latest_design_frozen_20260312/heuristic_run`
- Proposed run root: `experiments/runs/refine_compare_from_latest_design_frozen_20260312/proposed_run`
- Casewise CSV: `experiments/runs/refine_compare_from_latest_design_frozen_20260312/refine_compare_casewise.csv`
- Method summary CSV: `experiments/runs/refine_compare_from_latest_design_frozen_20260312/refine_compare_summary_by_method.csv`
- Method x variant summary CSV: `experiments/runs/refine_compare_from_latest_design_frozen_20260312/refine_compare_summary_by_method_variant.csv`

## Results by method
| method | n | core_recovered | entry_recovered | adopt_core_gain_sum | adopt_entry_gain_sum | mean_d_clr_feasible | mean_d_C_vis_start | mean_d_OOE_entry | mean_Delta_layout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 20 | 2 | 2 | 2 | 2 | +0.0163 | +0.0121 | -0.0014 | 0.0005 |
| proposed | 20 | 2 | 0 | 2 | 0 | +0.0202 | +0.0219 | +0.0118 | 0.0020 |

## Results by method x variant
| method | variant | n | core_recovered | entry_recovered | mean_d_clr_feasible | mean_d_C_vis_start | mean_d_OOE_entry | mean_Delta_layout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | base | 5 | 1 | 1 | +0.0295 | +0.0137 | +0.0000 | 0.0006 |
| heuristic | usage_shift | 5 | 1 | 1 | +0.0309 | +0.0083 | +0.0000 | 0.0008 |
| heuristic | clutter | 5 | 0 | 0 | +0.0009 | -0.0337 | -0.0500 | 0.0000 |
| heuristic | compound | 5 | 0 | 0 | +0.0040 | +0.0603 | +0.0444 | 0.0007 |
| proposed | base | 5 | 1 | 0 | +0.0303 | +0.0242 | -0.0250 | 0.0022 |
| proposed | usage_shift | 5 | 1 | 0 | +0.0402 | +0.0288 | -0.0250 | 0.0022 |
| proposed | clutter | 5 | 0 | 0 | +0.0103 | +0.0050 | +0.0500 | 0.0019 |
| proposed | compound | 5 | 0 | 0 | +0.0000 | +0.0297 | +0.0472 | 0.0019 |

## Notes
- `delta_*` values are measured against the stress input metrics in `experiments/runs/stress_v2_natural_from_latest_design_frozen_20260312`.
- `core_recovered` counts cases with `Adopt_core: 0 -> 1`.
- `entry_recovered` counts cases with `Adopt_entry: 0 -> 1`.
- This report fixes the comparison conditions to the frozen latest-design upstream run, frozen `eval_v1`, frozen `stress_v2_natural`, and frozen `proposed_beam_v1`.
