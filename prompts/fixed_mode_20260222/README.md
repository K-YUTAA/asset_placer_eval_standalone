# Fixed mode prompts (2026-02-22 baseline)

- 呼称統一:
  - `Step1` = OpenAI
  - `Step2` = Spatial Understanding（Gemini）
  - `Step3` = rule-based

- `step1_openai_prompt.txt`
  - Step1（OpenAI）で使うプロンプト（ベースライン固定）
- `gemini_furniture_prompt.txt`
  - Step2（Spatial Understanding）家具 bbox 検出用ベースプロンプト
- `gemini_room_inner_frame_prompt.txt`
  - Step2（Spatial Understanding）部屋内枠検出用ベースプロンプト
  - 実行時に `dimensions.txt` 由来の `ROOM_CONSTRAINTS` が自動追記される
- `gemini_openings_prompt.txt`
  - Step2（Spatial Understanding）開口部（door/sliding_door/window）検出用プロンプト
