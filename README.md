# Asset Placer Eval Standalone

Isaac Sim / Omniverse 拡張に依存せず、以下だけを実行するための独立リポジトリです。

- 画像 + 寸法テキスト + LLM によるレイアウト JSON 生成
- eval-loop (`C_vis`, `R_reach`, `clr_min`, `C_vis_start`, `OOE_*`) の計算
- デバッグ可視化 (`plot_with_bg.png`, `c_vis_*`) の出力

## 1. セットアップ

```bash
cd /path/to/asset_placer_eval_standalone
uv sync
```

OpenAI API キーを環境変数に設定してください。

```bash
export OPENAI_API_KEY="sk-..."
```

## 2. JSON 生成 (Step1/Step2)

```bash
uv run python experiments/src/generate_layout_json.py \
  --image_path experiments/fixtures/sketches/example.png \
  --dimensions_path experiments/fixtures/hints/example.txt \
  --out_json experiments/runs/demo_generate/layout_generated.json
```

主なオプション:

- `--prompt1_path`, `--prompt2_path`: デフォルトは `prompts/prompt_step*_universal_v7.txt`
- `--step2_text_only`: Step2 で画像 + dimensions を渡さない
- `--analysis_input`: Step1 をスキップして既存の Step1 JSON/テキストを利用

## 3. 評価実行

```bash
uv run python experiments/src/eval_metrics.py \
  --layout experiments/runs/demo_generate/layout_generated.json \
  --config experiments/configs/eval/default_eval.json \
  --out experiments/runs/demo_generate/metrics.json \
  --debug_dir experiments/runs/demo_generate/debug
```

## 4. 可視化 (`plot_with_bg.png`)

```bash
uv run python experiments/src/plot_layout_json.py \
  --layout experiments/runs/demo_generate/layout_generated.json \
  --out experiments/runs/demo_generate/plot_with_bg.png \
  --bg_image experiments/fixtures/sketches/example.png \
  --metrics_json experiments/runs/demo_generate/metrics.json \
  --task_points_json experiments/runs/demo_generate/debug/task_points.json \
  --path_json experiments/runs/demo_generate/debug/path_cells.json
```

## 5. trial 実行

```bash
uv run python experiments/src/run_trial.py \
  --trial_config experiments/configs/trials/sample_local_original.json \
  --eval_config experiments/configs/eval/default_eval.json \
  --out_root experiments/results
```

`sample_local_original.json` は `layout_input` に生成済みJSONを指定する想定です。

## 6. 一括実行 (生成 + 評価 + 可視化)

```bash
uv run python experiments/src/run_generate_and_eval.py \
  --image_path experiments/fixtures/sketches/example.png \
  --dimensions_path experiments/fixtures/hints/example.txt \
  --out_dir experiments/runs/demo_pipeline \
  --bg_image experiments/fixtures/sketches/example.png
```
