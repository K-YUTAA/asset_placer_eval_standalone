# Heuristic Refine アルゴリズム説明と改善実行結果（2026-02-26）

## 1. 対象

- 実装: `experiments/src/refine_heuristic.py`
- 対象データ: `experiments/runs/stress_cases_v2_from_batch_v2_gpt_medium_e2e_rerun_20260226`
- 対象シナリオ: `bottleneck / occlusion / clutter`（合計15ケース）


## 2. 現在の Heuristic アルゴリズム

## 2.1 スコア関数

`_score()` は以下で構成されています（`experiments/src/refine_heuristic.py:28`）。

- 加点:
  - `C_vis`（可視率）
  - `R_reach`（到達率）
  - `clr_min`（クリアランス）
- 減点:
  - `Delta_layout`
  - `validity==0`、`R_reach<=0` のペナルティ

実装上の特徴:

- 入口観測系（`C_vis_start`, `OOE_*`）はスコアに入っていません。

## 2.2 対象家具の選択

`_select_target_object()`（`experiments/src/refine_heuristic.py:60`）で、毎反復で1つの可動物体を選びます。

- `bottleneck_cell` がある場合:
  - ボトルネック最近傍の可動家具を選択（`experiments/src/refine_heuristic.py:80`）
- ない場合:
  - start-goal 線分への距離が最小の家具を選択（`experiments/src/refine_heuristic.py:111`）

## 2.3 近傍探索

`run_refinement()`（`experiments/src/refine_heuristic.py:123`）で局所探索を行います。

- 1反復で候補を全列挙:
  - `dx ∈ {-step,0,+step}`
  - `dy ∈ {-step,0,+step}`
  - `dθ ∈ {-rot,0,+rot}`
  - ただし `(0,0,0)` は除外
- つまり1反復あたり最大26候補

## 2.4 候補採択条件（強いガード）

候補は以下を満たす必要があります。

- 部屋内（OBB corners が room polygon 内）  
  - `experiments/src/refine_heuristic.py:174`
- `validity == 1`
  - `experiments/src/refine_heuristic.py:177`
- 非退行制約:
  - `R_reach` を悪化させない  
    `experiments/src/refine_heuristic.py:182`
  - `clr_min` を悪化させない  
    `experiments/src/refine_heuristic.py:184`
- 上記を満たし `score` が改善した場合のみ採択  
  - `experiments/src/refine_heuristic.py:188`

## 2.5 停止条件

以下で終了します。

- これ以上改善候補がない
- `max_iterations` 到達
- 対象物体が選べない


## 3. 今回の改善実行条件

15ケースに対して同一条件で実行。

- `max_iterations=30`
- `step_m=0.10`
- `rot_deg=15.0`
- `max_changed_objects=3`
- 再評価時の baseline は各 degraded layout（= repair量として評価）

出力:

- 各ケース:
  - `layout_refined.json`
  - `metrics_refined_eval.json`
  - `debug_refined/*`
  - `plot_with_bg_refined_eval.png`
- 集計:
  - `experiments/runs/stress_cases_v2_from_batch_v2_gpt_medium_e2e_rerun_20260226/refine_summary_15cases.json`
- サマリー画像:
  - `plot_with_bg_refined_eval_summary_bottleneck.png`
  - `plot_with_bg_refined_eval_summary_occlusion.png`
  - `plot_with_bg_refined_eval_summary_clutter.png`
  - `plot_with_bg_refined_eval_summary_all_15.png`


## 4. 改善結果（定量）

## 4.1 全15ケース平均

- `ΔC_vis`: `+0.0041`
- `ΔR_reach`: `+0.0051`
- `Δclr_min`: `+0.0133`
- `Adopt_core` 改善: `2/15`
- `Adopt_entry` 改善: `2/15`
- 悪化ケース: `0`
- 数値変化があったケース: `2/15`

## 4.2 シナリオ別平均

### bottleneck（5ケース）

- `ΔC_vis`: `+0.0062`
- `ΔR_reach`: `+0.0077`
- `Δclr_min`: `+0.0200`
- `Adopt_core` 改善: `1/5`
- `Adopt_entry` 改善: `1/5`

### occlusion（5ケース）

- `ΔC_vis`: `+0.0062`
- `ΔR_reach`: `+0.0077`
- `Δclr_min`: `+0.0200`
- `Adopt_core` 改善: `1/5`
- `Adopt_entry` 改善: `1/5`

### clutter（5ケース）

- `ΔC_vis`: `0.0000`
- `ΔR_reach`: `0.0000`
- `Δclr_min`: `0.0000`
- `Adopt_core` 改善: `0/5`
- `Adopt_entry` 改善: `0/5`

## 4.3 実際に改善したケース

改善が確認できたのは以下2件（同一ケースの別シナリオ）:

- `kugayama_A_18_5.93x3.04_v2 / bottleneck`
  - `ΔR_reach=+0.0384`, `ΔC_vis=+0.0308`, `Δclr_min=+0.1000`
  - `Adopt_core: 0->1`, `Adopt_entry: 0->1`
- `kugayama_A_18_5.93x3.04_v2 / occlusion`
  - `ΔR_reach=+0.0384`, `ΔC_vis=+0.0308`, `Δclr_min=+0.1000`
  - `Adopt_core: 0->1`, `Adopt_entry: 0->1`


## 5. なぜ変化が小さいか（実装上の理由）

主因は以下です。

1. 局所探索が1物体×26近傍で、探索幅が小さい  
2. 非退行制約（`R_reach`, `clr_min`）が強く、探索が保守的  
3. 目的関数が入口観測（`C_vis_start`, `OOE`）を直接最適化していない  
4. `clutter` は追加障害物が固定のため、1手局所移動では回復余地が小さい


## 6. 結論

- 現行 heuristic は「安全側で小さく直す」挙動としては一貫している
- ただし stress 15ケースでの改善率は限定的（`2/15`）
- 提案手法としては、次段で `proposed`（探索幅拡張/目的関数拡張）を独立実装する必要がある

