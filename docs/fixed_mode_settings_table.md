# Fixed mode settings table

対象: `experiments/configs/pipeline/fixed_mode_v2_gpt_high_batch_20260222.json`

## 1. ルート設定

| キー | 値 |
|---|---|
| `description` | `Fixed-mode baseline: Step1(OpenAI) + Step2(Spatial Understanding) + Step3(rule-based)` |
| `continue_on_error` | `true` |

## 2. `defaults`

| キー | 値 |
|---|---|
| `python_exec` | `.venv/bin/python` |
| `run_root` | `experiments/runs/batch_v2_gpt_high_fixed_mode_20260222` |

## 3. `defaults.generation_args`

| キー | 値 |
|---|---|
| `model` | `gpt-5.2` |
| `reasoning_effort` | `high` |
| `text_verbosity` | `high` |
| `image_detail` | `high` |
| `max_output_tokens` | `32000` |
| `prompt1_path` | `prompts/fixed_mode_20260222/step1_openai_prompt.txt` |
| `enable_gemini_spatial` | `true` |
| `gemini_model` | `gemini-3-flash-preview` |
| `gemini_temperature` | `0.6` |
| `gemini_include_non_furniture` | `false` |
| `gemini_thinking_budget` | `0` |
| `gemini_max_items` | `24` |
| `gemini_resize_max` | `640` |
| `gemini_prompt_text_path` | `prompts/fixed_mode_20260222/gemini_furniture_prompt.txt` |
| `gemini_room_inner_frame_prompt_text_path` | `prompts/fixed_mode_20260222/gemini_room_inner_frame_prompt.txt` |
| `gemini_openings_prompt_text_path` | `prompts/fixed_mode_20260222/gemini_openings_prompt.txt` |

## 4. `defaults.eval_args`

| キー | 値 |
|---|---|
| `config` | `experiments/configs/eval/default_eval.json` |

## 5. `defaults.plot_args`

| キー | 値 |
|---|---|
| `enabled` | `true` |
| `bg_crop_mode` | `none` |

## 6. `cases`（全5件）

| `name` | `image_path` | `dimensions_path` |
|---|---|---|
| `kugayama_A_18_5.93x3.04_v2` | `inputs_isaac/kugayama_A_18_5.93*3.04_v2.png` | `inputs_isaac/dimensions_kugaA18.txt` |
| `suginamikamiigusa_C_30_5.84x5.14_v2` | `inputs_isaac/suginamikamiigusa_C_30_5.84*5.14_v2.png` | `inputs_isaac/dimensions_sugiC30.txt` |
| `oizumi-gakuen_C_24_5.19x4.63_v2` | `inputs_isaac/oizumi-gakuen_C_24_5.19*4.63_v2.png` | `inputs_isaac/dimensions_oizuC24.txt` |
| `komazawakoen_B1_30_5.1x5.88_v2` | `inputs_isaac/komazawakoen_B1_30_5.1*5.88_v2.png` | `inputs_isaac/dimensions_komaB130.txt` |
| `suginamikamiigusa_B_23_5.75x4.00_v2` | `inputs_isaac/suginamikamiigusa_B_23_5.75*4.00_v2.png` | `inputs_isaac/dimensions_sugiB23.txt` |
