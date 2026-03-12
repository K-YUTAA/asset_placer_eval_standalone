# Phase1 実装報告（2026-02-28）

## 目的
今回の Phase1 は、以下を先に固めるための実装です。

- stress ケースの説明可能性を上げる
- plot と eval の整合確認をしやすくする
- 既存パイプライン互換を維持したまま拡張する

---

## 今回実装した内容

## 1. stress ケースごとの標準 manifest を追加

- 対象ファイル: `experiments/src/generate_stress_cases.py`
- 追加出力: 各 variant ディレクトリ直下に `stress_manifest.json`

### 追加した主な項目
- `case_id`
- `variant`
- `stress_type`
- `target_metric`
- `source_layout_path`
- `base_metrics`
- `variant_metrics`
- `delta_metrics`
- `Delta_disturb`
- `status`
- `score`
- `actions`

### 補足
- 既存の `disturb_manifest.json` はそのまま維持
- run-root の `stress_cases_manifest.json` もそのまま維持

---

## 2. plot 側に eval 整合確認オプションを追加

- 対象ファイル: `experiments/src/plot_layout_json.py`

### 新規 CLI 引数
- `--eval_debug_dir`
  - `eval_metrics.py --debug_dir` の出力先（`occupancy.pgm`, `path_cells.json` 等）を指定
- `--show_eval_bounds`
  - `path_cells.json` の `bounds` をプロット範囲として優先使用
- `--show_eval_occupancy`
  - `occupancy.pgm` の occupied セルを半透明で重畳表示
- `--eval_occupancy_alpha`
  - occupied 重畳表示の透明度（既定: `0.22`）

### 追加挙動
- `--eval_debug_dir` 未指定時も、`--path_json` / `--task_points_json` から debug ディレクトリを推定
- `--show_eval_*` 未指定時は従来挙動を維持

---

## 3. pipeline から新オプションを渡せるよう接続

- 対象ファイル: `experiments/src/run_pipeline_from_json.py`

### 追加挙動
- `plot_args.eval_debug_dir` が未指定なら `out_dir/debug` を自動採用
- `plot_layout_json.py` 実行時に `--eval_debug_dir` を自動付与（存在時）
- `plot_args` のパススルーで新オプションを利用可能

---

## 4. 一気通貫ラッパーにも接続

- 対象ファイル: `experiments/src/run_generate_and_eval.py`

### 追加挙動
- `plot_layout_json.py` 呼び出し時に `--eval_debug_dir <debug_dir>` を付与

---

## 変更ファイル一覧

- `docs/20260228_implementation_plan_phase1.md`
- `experiments/src/generate_stress_cases.py`
- `experiments/src/plot_layout_json.py`
- `experiments/src/run_pipeline_from_json.py`
- `experiments/src/run_generate_and_eval.py`

---

## 利用方法（例）

## A. 単体 plot で eval 整合を可視化
```bash
cd /Users/yuuta/Research/asset_placer_eval_standalone && \
uv run python experiments/src/plot_layout_json.py \
  --layout experiments/runs/<run>/<case>/layout_generated.json \
  --out experiments/runs/<run>/<case>/plot_with_bg_evalcheck.png \
  --bg_image inputs_isaac/<image>.png \
  --metrics_json experiments/runs/<run>/<case>/metrics.json \
  --task_points_json experiments/runs/<run>/<case>/debug/task_points.json \
  --path_json experiments/runs/<run>/<case>/debug/path_cells.json \
  --eval_debug_dir experiments/runs/<run>/<case>/debug \
  --show_eval_bounds \
  --show_eval_occupancy
```

## B. JSON パイプラインから有効化
`plot_args` に以下を設定:
```json
{
  "show_eval_bounds": true,
  "show_eval_occupancy": true,
  "eval_occupancy_alpha": 0.22
}
```

---

## 期待される効果

- stress ケースごとに「何を狙ってどう悪化したか」を追跡できる
- plot と eval のズレを直接可視化できる
- 次段（geometry 単一化、proposed 改善）に進むための検証基盤が整う

---

## 現時点の制限

- occupancy 重畳は可視化補助であり、まだ「描画=評価」完全一致を保証する実装ではない
- 壁境界 epsilon の全工程共通化は Phase2 で対応

