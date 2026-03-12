# Pipeline flow with config meaning and input output

Updated: 2026-02-23  
Historical target config: `experiments/configs/pipeline/latest_design_v2_gpt_high.json`

Status note:
- This document originally explained the `latest_design_v2_gpt_high.json` template/example config.
- The latest frozen latest-design file is `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`.
- The canonical upstream config for fixed-mode experiment reproduction is `experiments/configs/pipeline/fixed_mode_v2_gpt_high_batch_20260222.json`.

---

## 1. Purpose

This document explains the current pipeline as:
1. Which config key controls which stage
2. What each stage reads
3. What each stage writes

---

## 2. Config summary and meaning

## Global keys

| Key | Current value | Meaning |
|---|---|---|
| `continue_on_error` | `true` | Continue next case even if one case fails |
| `defaults.python_exec` | `.venv/bin/python` | Python interpreter used by stage commands |
| `defaults.run_root` | `experiments/runs/batch_v2_gpt_high_latest_design` | Output root for all case runs |

## Generation keys (`defaults.generation_args`)

| Key | Current value | Meaning |
|---|---|---|
| `model` | `gpt-5.2` | LLM used for Step1 |
| `reasoning_effort` | `high` | Step1 reasoning depth |
| `text_verbosity` | `high` | Step1 response detail level |
| `image_detail` | `high` | Image detail mode for Step1 |
| `max_output_tokens` | `32000` | Max tokens for generation call |
| `step2_mode` | `rule` | Use rule based Step2 instead of LLM Step2 |
| `step1_provider` | `openai` | Step1 provider |
| `step2_provider` | `openai` | Provider key kept for compatibility |
| `prompt1_path` | `prompts/prompt_1_universal_v4_posfix2_gemini_bridge_v1.txt` | Step1 prompt |
| `prompt2_path` | `prompts/prompt_2_universal_v4_posfix2_gemini_bridge_v1.txt` | Used only when `step2_mode=llm` |
| `enable_gemini_spatial` | `true` | Enable Gemini furniture detection pass |
| `enable_gemini_openings` | `true` | Enable Gemini openings and inner frame passes |
| `gemini_model` | `gemini-3-flash-preview` | Gemini model name |
| `gemini_task` | `boxes` | 2D bbox output mode |
| `gemini_label_language` | `English` | Label language in Gemini output |
| `gemini_temperature` | `0.6` | Gemini temperature |
| `gemini_thinking_budget` | `0` | Gemini thinking budget |
| `gemini_max_items` | `24` | Max returned detections |
| `gemini_resize_max` | `640` | Inference resize max |
| `gemini_prompt_text` | custom floorplan prompt | Furniture bbox detection instruction |

## Eval and plot keys

| Key | Current value | Meaning |
|---|---|---|
| `defaults.eval_args.config` | `experiments/configs/eval/default_eval.json` | Metric config path |
| `defaults.plot_args.enabled` | `true` | Enable `plot_with_bg.png` |
| `defaults.plot_args.bg_crop_mode` | `none` | Background crop behavior |

---

## 3. Stage by stage input output

## Stage A run coordinator

Program:
- `experiments/src/run_pipeline_from_json.py`

Input:
- pipeline config JSON

Output:
- per case execution
- `batch_manifest.json`

---

## Stage B generation

Program:
- `experiments/src/generate_layout_json.py`

Main inputs:
- `image_path` from case
- `dimensions_path` from case
- `prompt1_path`
- Gemini options from `generation_args`

Main outputs:
- `layout_generated.json`
- `generation_manifest.json`
- `stage1_output_raw.json`
- `step1_output_parsed.json`
- `stage3_output_raw.json`
- `gemini_spatial_output.json`
- `gemini_room_inner_frame_output.json`
- `gemini_openings_output.json`
- each Gemini plot and manifest file

---

## Stage C evaluation

Program:
- `experiments/src/eval_metrics.py`

Input:
- `layout_generated.json`
- eval config

Output:
- `metrics.json`
- `debug/` artifacts

---

## Stage D plotting

Program:
- `experiments/src/plot_layout_json.py`

Input:
- `layout_generated.json`
- original background image
- `metrics.json`
- `debug/task_points.json`
- `debug/path_cells.json`
- `gemini_room_inner_frame_output.json`

Output:
- `plot_with_bg.png`

---

## 4. End to end flow with artifacts

```mermaid
flowchart TB
    C0[pipeline config json] --> C1[run pipeline from json]
    C1 --> C2[for each case]
    C2 --> G0[generate layout json]
    G0 --> E0[eval metrics]
    E0 --> P0[plot layout]
    P0 --> O0[case outputs]
    C1 --> O1[batch manifest json]

    subgraph GEN[generate layout json artifacts]
        G1[stage1 output raw json]
        G2[step1 output parsed json]
        G3[gemini spatial output json]
        G4[gemini room inner frame output json]
        G5[gemini openings output json]
        G6[stage3 output raw json]
        G7[layout generated json]
        G8[generation manifest json]
    end

    G0 --> G1
    G0 --> G2
    G0 --> G3
    G0 --> G4
    G0 --> G5
    G0 --> G6
    G0 --> G7
    G0 --> G8

    E0 --> E1[metrics json]
    E0 --> E2[debug files]

    P0 --> P1[plot with bg png]
```

---

## 5. Generation internal flow with key level handoff

```mermaid
flowchart TB
    A0[input image path]
    A1[input dimensions text]
    A2[prompt1 text]
    A3[generation args]

    A0 --> B0
    A1 --> B0
    A2 --> B0
    A3 --> B0[step1 openai]
    B0 --> B1[step1_json]
    B1 --> B2[stage1_output_raw json]
    B1 --> B3[step1_output_parsed json]

    B1 --> C0[collect step1 category counts]
    C0 --> C1[step1_category_counts]
    C1 --> C2[gemini target prompt]
    A1 --> C3[build furniture guard lines]
    C1 --> C3
    C3 --> C4[gemini furniture prompt text]

    A0 --> D0[gemini furniture pass]
    C2 --> D0
    C4 --> D0
    D0 --> D1[gemini_spatial_json]

    A1 --> E0[build inner frame constraints]
    E0 --> E1[inner_frame_target_prompt]
    E0 --> E2[inner_frame_prompt_text]
    A0 --> E3[gemini inner frame pass]
    E1 --> E3
    E2 --> E3
    E3 --> E4[room_inner_frame_json]
    E4 --> E5[room_inner_frame_objects]
    B1 --> E6[build main_room_inner_boundary_hint]
    E5 --> E6

    A0 --> F0[gemini openings pass]
    F0 --> F1[opening_json]
    F1 --> F2[opening_objects]
    B1 --> F3[openings quality eval]
    F2 --> F3
    F3 --> F4[retry best opening_objects]

    B1 --> R0[step3 rule build_layout_rule_based]
    D1 --> R0
    E5 --> R0
    E6 --> R0
    F4 --> R0
    R0 --> R1[stage3_output_raw json]
    R0 --> R2[layout_generated json]
    R2 --> R3[generation_manifest json]
```

Responsibility split:
1. OpenAI Step1
- semantic understanding
- room inventory consistency
- functional orientation hint
- search prompt text

2. Gemini bridge
- furniture bbox geometry
- room inner frame geometry
- openings geometry

3. Step2 rule
- coordinate transform `norm -> world -> local`
- room and opening alignment to wall geometry
- schema assembly for final layout JSON

---

## 6. Step1 to Gemini furniture handoff detail

```mermaid
flowchart LR
    S1[step1_json objects]
    D1[dimensions text]
    P1[user gemini_prompt_text]

    S1 --> H1[_collect_step1_category_counts]
    H1 --> H2[step1_category_counts map]
    H2 --> H3[_build_default_gemini_target_prompt]
    H3 --> H4[target_prompt string]

    P1 --> H5[_build_gemini_furniture_prompt]
    D1 --> H5
    H2 --> H5
    H5 --> H6[prompt_text with guard lines]

    H4 --> G0[_run_gemini_spatial furniture]
    H6 --> G0
    G0 --> G1[gemini_spatial_output json]
```

What is passed:
1. `target_prompt`
- inventory level instruction
- Example: `objects matching this inventory: bed x1, sink x1, ...`

2. `prompt_text`
- base furniture prompt
- dimensions derived room hints
- guard lines such as:
  - expected furniture inventory
  - room labels are not furniture
  - sink and storage separation rule

What is not passed:
1. Step1 per object bbox
2. Step1 per object front_hint
3. Step1 per object coordinates

---

## 7. Key level transfer map with file write points

```mermaid
flowchart TB
    subgraph S1[step1 openai]
        A1[step1_text]
        A2[extract_json_payload]
        A3[step1_json parsed object]
        A4[write stage1_output_raw json]
        A5[write step1_output_parsed json]
        A1 --> A2 --> A3
        A3 --> A4
        A3 --> A5
    end

    subgraph GFM[gemini furniture pass]
        B1[step1_json objects]
        B2[collect step1_category_counts]
        B3[build target_prompt]
        B4[build furniture_prompt_text]
        B5[run gemini spatial furniture]
        B6[gemini_spatial_json]
        B1 --> B2 --> B3 --> B5
        B2 --> B4 --> B5
        B5 --> B6
    end

    subgraph GIF[gemini inner frame pass]
        C1[dimensions_text]
        C2[build inner_frame_target_prompt]
        C3[build inner_frame_prompt_text]
        C4[run gemini room_inner_frame]
        C5[room_inner_frame_objects]
        C6[build main_room_inner_boundary_hint]
        C1 --> C2 --> C4
        C1 --> C3 --> C4
        C4 --> C5 --> C6
    end

    subgraph GOP[gemini openings pass]
        D1[run gemini openings]
        D2[opening_objects]
        D3[evaluate with step1 openings]
        D4[retry select best opening_objects]
        D1 --> D2 --> D3 --> D4
    end

    subgraph S3[step3 rule]
        E1[build_layout_rule_based]
        E2[write stage3_output_raw json]
        E3[write layout_generated json]
        E4[write generation_manifest json]
        E1 --> E2
        E1 --> E3 --> E4
    end

    A3 --> B1
    A3 --> D3
    A3 --> E1
    B6 --> E1
    C5 --> E1
    C6 --> E1
    D4 --> E1
```

---

## 8. Actual data example from one run

Example source:
- `experiments/runs/archive/archive_20260222_220433/batch_v2_gpt_high_latest_design/komazawakoen_B1_30_5.1x5.88_v2/generation_manifest.json`

Observed handoff values:
1. `step1_category_counts`
- `{"bed":1,"sink":1,"sofa":1,"storage":1,"table":1,"toilet":1,"tv_cabinet":1}`

2. furniture `target_prompt`
- `objects matching this inventory: bed x1, sink x1, sofa x1, storage x1, table x1, toilet x1, tv_cabinet x1`

3. inner frame target
- `room_inner_frame x1, subroom_inner_frame x1`

4. openings target
- `doors, sliding doors, windows`

5. detected object counts
- `room_inner_frame.object_count = 2`
- `openings.object_count = 3`

Interpretation:
1. Step1 parsed output is used as a control signal for Gemini furniture inventory.
2. Gemini furniture pass does not receive per object bbox or front_hint.
3. Step3 rule receives both semantic guide from Step1 and geometry from Gemini.

---

## 9. Step2 rule input output contract

Input contract:
- `step1_json`
- `gemini_spatial_json`
- `room_inner_frame_objects`
- `opening_objects`
- `main_room_inner_boundary_hint`

Output contract:
- `area_name`
- `area_size_X`
- `area_size_Y`
- `size_mode`
- `outer_polygon`
- `rooms`
- `windows`
- `area_objects_list`

---

## 10. One case output tree example

```
experiments/runs/batch_v2_gpt_high_latest_design/<case_name>/
  layout_generated.json
  generation_manifest.json
  stage1_output_raw.json
  step1_output_parsed.json
  stage3_output_raw.json
  gemini_spatial_output.json
  gemini_spatial_output_plot.png
  gemini_room_inner_frame_output.json
  gemini_room_inner_frame_output_plot.png
  gemini_openings_output.json
  gemini_openings_output_plot.png
  metrics.json
  debug/
  plot_with_bg.png
```

---

## 11. Step by step input process output diagram

```mermaid
flowchart TB
    subgraph STEP1[step1 openai]
        S1I[input image path, dimensions text, prompt1 text, model args]
        S1P[process image understanding and semantic layout inference]
        S1O1[output step1_text]
        S1O2[output step1_json parsed]
        S1O3[write stage1_output_raw json, write step1_output_parsed json]
        S1I --> S1P --> S1O1 --> S1O2 --> S1O3
    end

    subgraph STEP2A[step2a gemini furniture]
        S2AI1[input image path]
        S2AI2[input step1_json objects]
        S2AI3[input dimensions text, gemini prompt config]
        S2AP1[process category counts from step1 objects]
        S2AP2[process build target_prompt inventory string]
        S2AP3[process build furniture_prompt_text with guard lines]
        S2AO1[output gemini_spatial_output json]
        S2AO2[output furniture objects bbox_2d_norm]
        S2AI2 --> S2AP1 --> S2AP2
        S2AI3 --> S2AP3
        S2AP1 --> S2AP3
        S2AI1 --> S2AO1
        S2AP2 --> S2AO1
        S2AP3 --> S2AO1 --> S2AO2
    end

    subgraph STEP2B[step2b gemini room inner frame]
        S2BI1[input image path]
        S2BI2[input dimensions text]
        S2BI3[input step1_json area and objects]
        S2BP1[process build inner_frame_target_prompt from expected room counts]
        S2BP2[process run gemini for room_inner_frame and subroom_inner_frame]
        S2BP3[process build main_room_inner_boundary_hint]
        S2BO1[output gemini_room_inner_frame_output json]
        S2BO2[output room_inner_frame_objects]
        S2BI2 --> S2BP1 --> S2BP2 --> S2BO1 --> S2BO2 --> S2BP3
        S2BI1 --> S2BP2
        S2BI3 --> S2BP3
    end

    subgraph STEP2C[step2c gemini openings]
        S2CI1[input image path]
        S2CI2[input openings prompt config]
        S2CI3[input step1_json openings for retry evaluation]
        S2CP1[process run gemini openings detection]
        S2CP2[process extract opening_objects]
        S2CP3[process quality eval against step1 openings]
        S2CP4[process retry and keep best opening_objects]
        S2CO1[output gemini_openings_output json]
        S2CO2[output opening_objects]
        S2CI1 --> S2CP1
        S2CI2 --> S2CP1 --> S2CO1
        S2CO1 --> S2CP2 --> S2CO2 --> S2CP3 --> S2CP4
        S2CI3 --> S2CP3
    end

    subgraph STEP3[step3 rule based builder]
        S3I1[input step1_json parsed]
        S3I2[input gemini_spatial_json furniture bbox]
        S3I3[input room_inner_frame_objects and boundary hint]
        S3I4[input opening_objects]
        S3P1[process coordinate normalize to world to local]
        S3P2[process room and opening alignment to walls]
        S3P3[process schema assembly rooms windows area_objects_list]
        S3O1[output stage3_output_raw json]
        S3O2[output layout_generated json]
        S3O3[output generation_manifest json]
        S3I1 --> S3P1
        S3I2 --> S3P1
        S3I3 --> S3P1
        S3I4 --> S3P1
        S3P1 --> S3P2 --> S3P3 --> S3O1
        S3P3 --> S3O2 --> S3O3
    end

    S1O2 --> S2AI2
    S1O2 --> S2BI3
    S1O2 --> S2CI3

    S2AO2 --> S3I2
    S2BP3 --> S3I3
    S2CP4 --> S3I4
    S1O2 --> S3I1
```

Key point:
1. `step1_json parsed` is a real runtime input to three places:
- Gemini furniture control signal
- Gemini openings retry evaluation
- Step3 final builder
2. Gemini furniture receives inventory level guidance from Step1, not per object coordinates.
