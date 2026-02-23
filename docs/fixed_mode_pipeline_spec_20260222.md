# Fixed-mode pipeline spec (baseline/fixed-mode-gpt-high-20260222)

## 目的
- モード切り替えを固定し、ベースライン精度の再現性を優先する。
- 各ステップの入力データと入力プロンプトをファイルとして明示化する。

## 固定モード（呼称統一）
- Step1（OpenAI）: `OpenAI`（`gpt-5.2`, reasoning=`high`）
- Step2（Spatial Understanding）: `Gemini Spatial` `ON`（`gemini-3-flash-preview`）
  - furniture bbox: ON
  - room inner frame: ON
  - openings: ON
- Step3（rule-based）: `rule-based`（LLM Step2は使用しない）

## 実行設定ファイル
- `experiments/configs/pipeline/fixed_mode_v2_gpt_high_batch_20260222.json`

## このブランチで設定として触る Gemini キー（最小）
- `enable_gemini_spatial`
- `gemini_model`
- `gemini_temperature`
- `gemini_prompt_text_path`
- `gemini_room_inner_frame_prompt_text_path`
- `gemini_openings_prompt_text_path`
- `gemini_include_non_furniture`
- `gemini_resize_max`
- `gemini_max_items`
- `gemini_thinking_budget`

上記以外（`gemini_target_prompt` や retry閾値など）はコード既定値を使用する。

## 入力データ（各ステップ）
1. Step1（OpenAI）
- 入力
  - floorplan画像: `cases[].image_path`
  - dimensions: `cases[].dimensions_path`
  - prompt: `prompts/fixed_mode_20260222/step1_openai_prompt.txt`
- 出力
  - `step1_output_raw.json`
  - `step1_output_parsed.json`

2. Step2（Spatial Understanding: furniture detection）
- 入力
  - floorplan画像
  - Step1 category counts（inventory）
  - prompt base: `prompts/fixed_mode_20260222/gemini_furniture_prompt.txt`
- 出力
  - `gemini_spatial_output.json`
  - `gemini_spatial_output_plot.png`

3. Step2（Spatial Understanding: room inner frame detection）
- 入力
  - floorplan画像
  - prompt base: `prompts/fixed_mode_20260222/gemini_room_inner_frame_prompt.txt`
  - dimensions由来制約（`ROOM_CONSTRAINTS`）を実行時に自動追記
- 出力
  - `gemini_room_inner_frame_output.json`
  - `gemini_room_inner_frame_output_plot.png`

4. Step2（Spatial Understanding: openings detection）
- 入力
  - floorplan画像
  - prompt: `prompts/fixed_mode_20260222/gemini_openings_prompt.txt`
- 出力
  - `gemini_openings_output.json`
  - `gemini_openings_output_plot.png`

5. Step3（rule-based）
- 入力
  - Step1（OpenAI）parsed JSON
  - Step2（Spatial Understanding）furniture/inner-frame/openings JSON
  - dimensions
- 出力
  - `layout_generated.json`

6. Eval + Plot
- 入力
  - `layout_generated.json`
  - `experiments/configs/eval/default_eval.json`
- 出力
  - `metrics.json`
  - `debug/*`
  - `plot_with_bg.png`

## 実行コマンド
```bash
cd /Users/yuuta/Research/asset_placer_eval_standalone
uv run python experiments/src/run_pipeline_from_json.py \
  --config experiments/configs/pipeline/fixed_mode_v2_gpt_high_batch_20260222.json
```

## 補足
- OpenAI/GeminiのAPIキーは環境変数 (`OPENAI_API_KEY`, `GOOGLE_API_KEY`) で与える。
- `enable_gemini_spatial` が唯一の Spatial トグル。ON のときは furniture / inner frame / openings の3つを必ず実行する。
