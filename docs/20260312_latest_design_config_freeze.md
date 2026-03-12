# Latest Design Config Freeze (2026-03-12)

## Purpose

This note freezes a parser-consistent "latest design" pipeline config for the current codebase.

It does not replace the fixed-mode experiment authority.

- Fixed-mode experiment authority remains `experiments/configs/pipeline/fixed_mode_v2_gpt_high_batch_20260222.json`
- The frozen latest-design snapshot is `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`

## Why a new file was needed

`experiments/configs/pipeline/latest_design_v2_gpt_high.json` contains stale keys from an older execution path.

The current batch runner `experiments/src/run_pipeline_from_json.py` passes `defaults.generation_args` directly to `experiments/src/generate_layout_json.py`.
However, the old latest-design config still contains keys that belong to `experiments/src/run_generate_and_eval.py` and are not accepted by the current `generate_layout_json.py` parser.

Verified unsupported keys in the old file:

- `enable_gemini_openings`
- `prompt2_path`
- `step1_provider`
- `step2_mode`
- `step2_provider`

## Frozen config

- Latest frozen latest-design file in this repository as of 2026-03-12: `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`
- Config path: `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`
- Prompt snapshot dir: `prompts/latest_design_frozen_20260312/`

## Freeze rules

- Do not edit `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json` unless the user explicitly requests a latest-design spec update.
- Do not edit prompt snapshots under `prompts/latest_design_frozen_20260312/` unless the user explicitly requests a prompt/spec update.
- Keep `experiments/configs/eval/eval_v1.json` as the evaluation config for this frozen latest-design snapshot.

## What was made explicit

The new config keeps only arguments accepted by the current `generate_layout_json.py` parser and makes current behavior explicit where practical:

- Step1 prompt path
- Gemini furniture prompt path
- Gemini room inner frame prompt path
- Gemini openings prompt path
- Gemini model / task / label language / temperature / item cap / resize cap
- Gemini openings retry thresholds
- Gemini inner-frame retry thresholds
- `summary_args.enabled = true`

## Scope note

This file is a parser-consistent latest-design snapshot for current execution.
It is not the canonical upstream config for fixed-mode experiment reproduction.
