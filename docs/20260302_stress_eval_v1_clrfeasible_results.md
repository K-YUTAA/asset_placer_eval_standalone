# 2026-03-02 stress 評価結果（eval_v1 / clr_feasible採否）

## 目的

- `eval_v1` の clearance 採否を `clr_feasible` に固定した状態で、stress 20件（5ケース x base/bottleneck/occlusion/clutter）を評価する。
- `Adopt_core` と `Adopt_entry` の変化を、`clr_feasible` と `clr_min_astar` の両指標で比較する。

## 実行対象

- 実行ルート: `experiments/runs/stress_eval_v1_clrdual_20260302`
- マニフェスト: `experiments/runs/stress_eval_v1_clrdual_20260302/stress_cases_manifest.json`
- 件数: 20 layouts（base 5 + stress 15）
- 評価設定: `experiments/configs/eval/eval_v1.json`

## 生成された主要成果物

- 総合サマリー画像: `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_all_variants.png`
- variant別サマリー画像:
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_base.png`
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_bottleneck.png`
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_occlusion.png`
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_clutter.png`
- 変化量集計（CSV）: `experiments/runs/stress_eval_v1_clrdual_20260302/adopt_core_clr_feasible_summary.csv`
- 変化量集計（MD）: `experiments/runs/stress_eval_v1_clrdual_20260302/adopt_core_clr_feasible_summary.md`

## 集計結果（base比 / stress 15件）

| variant | n | core up | core down | core same | entry up | entry down | avg Δclr_feasible | avg Δclr_min_astar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottleneck | 5 | 0 | 1 | 4 | 0 | 1 | -0.0521 | -0.0376 |
| occlusion | 5 | 0 | 0 | 5 | 0 | 0 | 0.0000 | 0.0000 |
| clutter | 5 | 0 | 0 | 5 | 0 | 1 | -0.0766 | -0.1097 |

## 読み取り

- `Adopt_clearance_metric` は全stressケースで `clr_feasible` が使用されている。
- bottleneck は clearance 悪化を作れており、`Adopt_core` が 1件で `1->0` になった。
- occlusion は clearance を崩さない設計どおり、`Adopt_core` 変化は 0件。
- clutter は `clr_min_astar` の低下が比較的大きく、`Adopt_entry` が 1件で `1->0` になった。
- 互換キー `clr_min` は `clr_min_astar` と同値で維持されている。

## 補足

- 個別ケースの差分は `experiments/runs/stress_eval_v1_clrdual_20260302/adopt_core_clr_feasible_summary.md` を参照。
- 本結果は `Step1/Step2` の再生成ではなく、stressレイアウトに対する `eval + plot` の評価結果である。
