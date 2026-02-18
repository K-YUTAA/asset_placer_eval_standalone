# Experiments Workspace (Standalone)

この `experiments/` は Isaac Sim 非依存で eval-loop を回すための実験レイヤーです。

## 主要スクリプト

- `src/generate_layout_json.py`: 画像 + 寸法テキスト + OpenAI でレイアウトJSONを生成
- `src/eval_metrics.py`: 評価指標とデバッグ出力を生成
- `src/plot_layout_json.py`: 背景付き可視化 (`plot_with_bg.png`) を生成
- `src/run_trial.py`: `v0 -> eval -> (optional refine)` を1試行で実行

## 典型フロー

1. `generate_layout_json.py` で `layout_generated.json` を生成
2. `eval_metrics.py` で `metrics.json` + `debug/` を生成
3. `plot_layout_json.py` で `plot_with_bg.png` を出力

出力は通常 `experiments/runs/<run_id>/` に配置します。
