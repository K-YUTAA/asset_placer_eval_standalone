# 2026-03-13 Layout Refine Plausibility Fix Plan

## 目的

現行の `heuristic` / `proposed` に対して、次の 2 点を改善する。

1. `sink` などが不自然な角度に回転する問題
2. 壁食い込み・家具同士の見た目上の重複が残る問題

本ドキュメントは、これらを **frozen main protocol を壊さずに改善候補として切り出す** ための実装方針を整理する。

## 前提

### frozen のまま保持するもの

- `experiments/configs/eval/eval_v1.json`
- `experiments/configs/refine/proposed_beam_v1.json`
- `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`

### 今回の位置づけ

- **main result の上書きではない**
- `method spec update candidate` として扱う
- 既存の frozen 比較結果は保持する

## 現状の問題整理

### 1. 不自然回転

- 観測した比較 run では、家具の軸整列 prior が投入されていないケースがある
- prior 未投入時は `current_yaw ± rot_deg` がそのまま候補になる
- そのため、`sink`, `storage`, `tv_cabinet` などが visibility / OOE 改善だけを目的に斜め回転しやすい

### 2. 重複・壁食い込み

- `heuristic`
  - room polygon 内判定と evaluator validity には依存している
  - ただし候補評価中に明示的な overlap hard check がない
- `proposed`
  - overlap check はある
  - ただし `AABB overlap ratio <= 0.05` は緩い
- 両手法とも、現状は `wall margin` を持たず、`corners inside room polygon` ベース
- そのため、plot 上で壁厚込みだと食い込みや接触が目立つ

## 今回やること

## A. 家具回転の自然さ制約を入れる

### 方針

- 家具は原則として、部屋軸 / 壁方向に **平行または直交** を基本にする
- ただし、評価指標が大きく改善する場合のみ、少しの off-axis を許す

### 適用対象

- `M1` の furniture refine
- `X2` の furniture phase

### 非適用対象

- `X1` の clutter-only recovery
- `clutter` 自体の回転は別仕様で扱う

### 初期ルール

- 通常候補:
  - `theta_room + 0°`
  - `theta_room + 90°`
  - `theta_room + 180°`
  - `theta_room + 270°`
- 例外候補:
  - 直交方向の最近傍から `±10°`, `±15°`
- 例外採択条件:
  - `Adopt_core: 0 -> 1`
  - または `Adopt_entry: 0 -> 1`
  - または
    - `Δclr_feasible >= 0.03`
    - `ΔC_vis_start >= 0.05`
    - `ΔOOE_R_rec_entry_surf >= 0.10`

### category 別の考え方

- 強く拘束する:
  - `sink`
  - `storage`
  - `tv_cabinet`
  - `cabinet`
  - `bed`
  - `sofa`
- 将来的に弱めてもよい:
  - `chair`
- 向き意味が弱い:
  - 円形 `table`

## B. overlap hard constraint を追加・強化する

### heuristic

追加する。

- 家具同士の overlap hard check
- door keepout hard check
- wall margin hard check

最低でも `proposed` と同等以上の reject 条件にする。

### proposed

既存の overlap check を厳しくする。

- `AABB overlap ratio`
  - 現状 `0.05`
  - 候補は `0.01` まで引き下げる
- 将来候補:
  - OBB ベースのより厳密な判定

## C. wall margin を両手法に追加する

### 方針

- 「room polygon 内ならよい」では不足
- 家具 OBB が内壁から一定距離以上離れることを要求する

### 初期候補値

- `wall_margin_m = 0.03 ~ 0.05`

## D. clutter 側は別扱いを維持する

今回の主問題は furniture phase 側なので、`clutter_assisted` extension の位置づけは維持する。

- `X1`
  - clutter だけ可動
- `X2`
  - furniture refine 後に clutter refine

ただし、今回の `sink` 回転問題の直接原因は furniture phase なので、まずは `M1 / X2` を優先して直す。

## やらないこと

今回はやらない。

- `eval_v1.json` の変更
- frozen main result の上書き
- `threshold-repair + do-no-harm` への全面再設計
- clutter cleanup と furniture refine の同時最適化
- 3D / USD 配置ロジックの変更

## 実装順

1. `layout_axis_alignment_prior` を `M1 / X2` rerun に実際に投入する
2. `heuristic` に overlap / wall margin hard constraint を追加する
3. `proposed` の overlap / wall margin を厳しくする
4. 必要なら `sink / storage / tv_cabinet` をさらに強く拘束する
5. `clutter / compound` を中心に rerun する

## rerun 方針

### 優先ケース

- `clutter`
- `compound`

### protocol

- `M1`
- `X2`

### method

- `heuristic`
- `proposed`

### 理由

- prior の効果が出るのは furniture phase
- `X1` は clutter-only なので今回の `sink` 問題には直接効かない
- `base / usage_shift` は確認用に留めてよい

## 評価観点

### 維持したいもの

- `Adopt_core`
- `Adopt_entry`
- `clr_feasible`
- `C_vis_start`
- `OOE_R_rec_entry_surf`

### 改善したいもの

- 斜め配置の減少
- 壁食い込みの消失
- 家具重複の消失

### 追加で見るもの

- `delta_layout_furniture`
- `delta_layout_clutter`
- category 別の yaw 分布

## 成功条件

最低限の成功条件は次。

1. `sink`, `storage`, `tv_cabinet` の明らかな斜め回転が目視で大きく減る
2. `heuristic` で見えていた家具重複が hard reject される
3. 壁食い込みが目視で大きく減る
4. `Adopt_core` / `Adopt_entry` が極端に悪化しない

## 出力物

- rerun root
- `trial_manifest.json` に prior config path/hash を保存
- before/after の `plot_with_bg_compare`
- summary 画像
- 結果 MD

## 論文上の位置づけ

- main text:
  - frozen main comparison を維持
- appendix / extension:
  - layout plausibility fix candidate
  - axis-aligned prior + overlap/wall guard の効果

## 要点

今回の変更は、

- evaluator の凍結仕様を触らず
- main の frozen comparison も壊さず
- method 側の改善候補として
- 「不自然回転」と「見た目破綻」を減らす

ためのものです。
