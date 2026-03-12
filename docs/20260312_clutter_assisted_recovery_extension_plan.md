# Clutter-Assisted Recovery Extension Plan

## Goal

`main` の frozen comparison を壊さずに、`stress` で追加された `clutter` だけを動かせる recovery protocol を extension として追加する。

## Protocols

### Protocol M: Layout-only refinement

- 現行 main の正本
- 対象: 既存の可動家具のみ
- 比較条件:
  - `Original`
  - `Heuristic`
  - `Proposed`
- frozen artifacts:
  - `experiments/configs/eval/eval_v1.json`
  - `experiments/configs/refine/proposed_beam_v1.json`

### Protocol C: Clutter-assisted recovery

- extension 扱い
- 入力: Protocol M と同じ stress layout
- 対象: `stress` で追加された `clutter` だけ
- evaluator / task / thresholds / plotting は Protocol M と同一
- 違いは「何を動かしてよいか」だけ

## Scope of v1

v1 では次だけを許す。

- `clutter` の局所平行移動
- 回転なし
- 削除なし
- 室外退避なし
- furniture と clutter の同時可動なし

## Object metadata

各 object で次を追跡できるようにする。

- `origin`
  - `original`
  - `stress_added`
- `stress_kind`
  - `base`
  - `usage_shift`
  - `clutter`
  - `compound`
  - `targeted_bottleneck`
  - `targeted_occlusion`
- `movable_in_main`
- `movable_in_clutter_recovery`
- `object_role`
- `source_case_id`
- `stress_case_id`

## Logging requirements

`Delta_layout` は 1 本に潰さず、少なくとも次を分離して保存する。

- `delta_layout_furniture`
- `delta_layout_clutter`
- `moved_furniture_count`
- `moved_clutter_count`
- `sum_clutter_displacement_m`
- `max_clutter_displacement_m`
- `sum_furniture_displacement_m`

## Implementation strategy

1. `run_trial.py` に `recovery_protocol` を導入する
   - default: `layout_only`
2. `refine_heuristic.py`
   - protocol ごとに候補 object 集合を切り替える
3. `refine_proposed_beam.py`
   - protocol ごとに候補 object 集合を切り替える
   - `clutter_assisted` では `clutter` を除外しない
4. `generate_stress_cases.py`
   - `stress_added clutter` を object metadata で識別可能にする
5. `run_trial.py`
   - protocol ごとの displacement summary を manifest / CSV に保存する

## Experimental positioning

- main text:
  - Protocol M
- appendix / extension:
  - Protocol C

## Rationale

- main の frozen comparison を維持できる
- `clutter` 可動化の効果だけを単独で測れる
- 後で `threshold-repair + do-no-harm` を足しても、何が効いたか分離できる
