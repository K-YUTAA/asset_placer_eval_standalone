# M0 / M1 / X1 / X2 clutter recovery implementation policy

## Goal

`main` の frozen comparison を維持したまま、`clutter` を動かせる extension を追加する。

今回の目的は 2 つです。

- `layout-only refinement` と `clutter-assisted recovery` を明確に分ける
- `clutter` は元位置保持よりも、評価指標を改善する位置への再配置を優先する

## Protocol definition

### `M0`: no refine

- stress 入力をそのまま評価する
- 既存の `stress_v2_natural` 出力を基準にする

### `M1`: furniture refine

- 現行 `main`
- movable furniture のみ可動
- stress で追加された `clutter` は固定
- method は既存の `heuristic` / `proposed`

### `X1`: clutter refine

- stress で追加された `clutter` のみ可動
- furniture は固定
- 評価器は `eval_v1.json` をそのまま使う
- objective は `clutter` の元位置維持ではなく、指標改善を優先する

### `X2`: furniture -> clutter refine

- まず `M1`
- その出力に対して `X1`
- `M1` と `X1` の効果を分離できるよう、2段階を明示的に扱う

## What stays frozen

- `experiments/configs/eval/eval_v1.json`
- `experiments/configs/refine/proposed_beam_v1.json`

今回変更するのは extension 側だけです。

- `main` の furniture refine
- `Original / Heuristic / Proposed` の frozen comparison

は変えません。

## Why clutter is treated differently

`clutter` は stress で後付けされた外部障害物です。

- furniture:
  - 元レイアウトからの変更量を小さく保つ意味がある
- clutter:
  - 元位置保持の意味が薄い
  - 通れる範囲で、指標を最も改善する位置にスライドできる方が自然

したがって `X1 / X2` では、

- `delta_layout_clutter` は tie-breaker
- 主目的は
  - `Adopt_entry`
  - `Adopt_core`
  - `clr_feasible`
  - `C_vis_start`
  - `OOE_R_rec_entry_surf`
  - `R_reach`

の改善とする。

## X1 scoring policy

`clutter` 再配置では次の優先順位を使う。

1. `validity == 1`
2. `Adopt_entry == 1`
3. `Adopt_core == 1`
4. `clr_feasible` 最大化
5. `C_vis_start` 最大化
6. `OOE_R_rec_entry_surf` 最大化
7. `R_reach` 最大化
8. `C_vis` 最大化
9. `delta_layout_clutter` 最小化

`delta_layout_clutter` は主目的ではなく、同程度の候補が複数あるときの tie-breaker に落とす。

## X1 candidate generation

`clutter` は近傍 26 候補だけでは弱いので、同一 room 内の候補点を広く見る。

### Candidate constraints

- same room
- wall crossing なし
- door keep-out を守る
- object overlap 上限を守る
- evaluator と同じ occupancy 上で `validity == 1`

### Candidate positions

- room bounding box 上の格子点を候補にする
- yaw は初版では固定
- current position 近傍だけでなく、room 全体の free space を候補に含める
- ただし候補数は上限を設ける

### Quick ranking for candidate pruning

格子点は多いので、評価前に quick score で上位だけ残す。

- start-goal 線分から遠い
- start 点から遠い
- bottleneck から遠い
- room の端に寄っている

候補を優先する。

## Method behavior

### `heuristic-C`

- `clutter` のみ対象
- 各反復で、全 `clutter x candidate_position` を見て最良の 1 手だけ採用
- 反復回数は小さく保つ

### `proposed-C`

- `clutter` のみ対象
- beam search で複数候補を保持
- ただし objective は `main` の `proposed` とは分離
- `clutter` 再配置用の score / candidate generation を使う

## Logging

`X1 / X2` では変更量を分離して保存する。

- `delta_layout_furniture`
- `delta_layout_clutter`
- `moved_furniture_count`
- `moved_clutter_count`
- `sum_furniture_displacement_m`
- `sum_clutter_displacement_m`
- `max_clutter_displacement_m`

`compound` でも `Delta_total` は使わず、意味の違う変更量を分離する。

## Config files

### Main

- `experiments/configs/eval/eval_v1.json`
- `experiments/configs/refine/proposed_beam_v1.json`

### Extension

- `experiments/configs/refine/clutter_assisted_v1.json`

`clutter_assisted_v1.json` は extension protocol のみを定義する。

## Implementation scope for this step

この実装で入れるのは次まで。

- `X1` の clutter-specific scoring
- `X1` の global candidate positions
- `run_trial.py` の protocol dispatch
- `clutter_assisted_v1.json` の追加

この実装ではまだ入れない。

- `threshold-repair + do-no-harm`
- furniture と clutter の hybrid simultaneous optimization
- X2 専用 runner の全面整備

`X2` 自体は `M1` の出力に `X1` を重ねる chained execution として扱う。

## Expected outcome

- `main` の frozen comparison は不変
- `X1` では `clutter / compound` の回復率を別軸で見られる
- `delta_layout_clutter` を小さく保つことより、回復・可視性改善を優先できる

