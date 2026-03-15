# 20260316 Proposed v2 Threshold-Repair / Do-No-Harm Implementation Plan

## Goal

現行の `proposed_beam_v1.json` は frozen main comparison の正本として保持しつつ、別仕様の `proposed_v2` を追加実装し、`threshold-repair + do-no-harm` の効果を検証する。

この文書の目的は次の 2 点である。

1. `proposed_v2` の実装範囲を固定する
2. すべての変更を main frozen protocol から分離し、元に戻せる形で進める

## Reversibility Policy

今回の実装は、必ず元に戻せる形で行う。

原則:

- frozen artifact は上書きしない
  - `experiments/configs/eval/eval_v1.json`
  - `experiments/configs/refine/proposed_beam_v1.json`
  - `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`
- `proposed_v2` は新しい config 名で追加する
- 実行出力 root は新規ディレクトリに分ける
- `run_trial.py` / runner 側では `method_config_path` と hash を必ず保存する
- 既存の `heuristic` / `proposed_v1` の結果は消さない

実装後に元へ戻す方法:

- `proposed_v2` 用 config を指定しなければ、従来の `proposed_beam_v1.json` の挙動に戻る
- compare runner も `v2` 用 out root を分離するため、既存結果を汚染しない

## Scope

今回触る対象は `proposed` のみとする。

含む:

- `threshold-repair` 目的関数
- `do-no-harm` 制約
- 新しい `proposed_v2` config
- `proposed_v2` 実験 runner / 出力整理

含まない:

- `eval_v1.json` の変更
- `heuristic` のアルゴリズム変更
- `stress_v2_natural` の定義変更
- frozen main comparison の上書き

## Problem Statement

現状の `proposed_beam_v1` は次の問題を持つ。

- 連続値改善には効く
  - `Δclr_feasible`
  - `ΔC_vis_start`
  - `ΔOOE_R_rec_entry_surf`
- しかし binary recovery が弱い
  - `Adopt_core`
  - `Adopt_entry`

そのため、main claim としては弱い。

必要なのは、

- 平均的に少し良くする optimizer

ではなく、

- まず fail case を pass に戻す repair-oriented optimizer

である。

## Proposed v2 Design

### Core idea

`proposed_v2` は、連続値最大化ではなく、閾値未達の修復を優先する。

### Threshold-repair

まず各 trial の不足量を定義する。

例:

- `m_core_clr = clr_feasible - tau_clr_feasible`
- `m_core_reach = R_reach - tau_R`
- `m_core_vis = C_vis - tau_V`
- `m_entry = OOE_R_rec_entry_surf - entry_gate.min_value`

このとき `proposed_v2` は、

1. `Adopt_core` を満たしていない場合は core 不足量の修復を最優先
2. `Adopt_core` を満たした後に `Adopt_entry` を満たしていない場合は entry 不足量の修復を優先
3. 上の両方を満たした後に、連続値改善を行う

という優先順で探索する。

### Do-no-harm

いったん回復した条件は壊さない。

初版では次を hard gate にする。

- `validity == 1` を壊さない
- `Adopt_core = 1` になった後は、`Adopt_core = 0` に落ちる候補を禁止
- `Adopt_entry = 1` になった後は、`Adopt_entry = 0` に落ちる候補を禁止

必要なら第 2 段階で margin 付き制約へ拡張するが、初版では hard gate で十分とする。

## Scoring Structure

`proposed_v2` の候補比較は、次の辞書式順序で行う。

1. `validity`
2. `Adopt_core`
3. `Adopt_entry`
4. `repair_score`
5. `improvement_score`
6. `Delta_layout`

### repair_score

repair 段階では、未達指標の margin を大きくする候補を優先する。

イメージ:

- core 未達なら `min(m_core_clr, m_core_reach, m_core_vis)` を最大化
- entry 未達なら `m_entry` を最大化

### improvement_score

repair 後だけ使う。

対象:

- `C_vis_start`
- `OOE_R_rec_entry_surf`
- `clr_feasible`

ただし `repair_score` より優先度は低い。

## Config Strategy

新しい config を追加する。

候補:

- `experiments/configs/refine/proposed_beam_v2_threshold_repair.json`

ここに入れる項目:

- beam width
- depth
- candidate objects per state
- `repair_first = true`
- `do_no_harm = true`
- `primary_entry_metric = OOE_R_rec_entry_surf`
- repair margin settings
- improvement weights

重要:

- `proposed_beam_v1.json` は変更しない
- `proposed_v2` は別 hash として manifest に残す

## Code Changes

主に触るファイル:

- `experiments/src/refine_proposed_beam.py`
- `experiments/src/run_trial.py`

必要なら追加:

- `experiments/src/refine_proposed_threshold_repair.py`

推奨方針:

- 初版は既存 `refine_proposed_beam.py` の分岐で対応してよい
- ただしロジックが膨らむ場合は別ファイル化する

`run_trial.py` 側では:

- `method_config_path` から `v1 / v2` を切り替える
- manifest に
  - `method_config_path`
  - `method_config_sha256`
  - `method_version_label`
  を保存する

## Experiment Plan

### Stage 1: smoke

対象:

- `clutter` 2 case
- `compound` 1 case

目的:

- `Adopt_core` 回復が増えるか
- `do-no-harm` により探索停止しないか
- 可視化上の破綻が増えないか

### Stage 2: full rerun

対象:

- 20 case
- `Original / Heuristic / Proposed_v2`

必要なら比較用に:

- `Proposed_v1`

も並べる。

### Stage 3: figure/report update

更新対象:

- `Adopt_core / Adopt_entry`
- `Δclr_feasible`
- `ΔC_vis_start`
- `ΔOOE_R_rec_entry_surf`
- representative before/after

## Success Criteria

最低条件:

- `Proposed_v2` が `Proposed_v1` より `Adopt_core` 回復数で改善

理想条件:

- `Adopt_entry` 回復も増える
- `plot_with_bg` 上の不自然さが増えない

失敗条件:

- repair は増えず、単に連続値改善だけが増える
- `do-no-harm` が強すぎて探索が止まる
- 可視化の見た目が悪化する

## Estimated Time

実装から初回結果確認まで:

- `半日〜1日`

20ケース rerun と可視化まで:

- `さらに半日〜1日`

図表と本文反映まで:

- `さらに1〜2日`

全体見積もり:

- `2〜4日`

## Position in Paper

現時点では `spec update candidate` として扱う。

つまり:

- frozen main を直接上書きしない
- 成功したら main 採用を検討
- 失敗したら appendix / future work に落とす

この切り分けにより、再現性を壊さずに method 強化を試せる。
