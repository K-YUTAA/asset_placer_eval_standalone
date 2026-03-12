# 2026-03-02 実装報告: 開口部補正と Step3 以降の全ケース再生成

## 目的

- Gemini 由来の opening（door/window）を基準にしつつ、最終 `layout_generated.json` 側で位置・向きが崩れる問題を改善する。
- `Step1/Step2` を再実行せず、既存出力を使って `Step3 -> eval -> plot` を全ケースで再適用し、可視化と評価の整合を回復する。

## 今回の実装内容

### 1. 開口部向き判定の強化（Step3 rule-based）

- 対象: `experiments/src/step2_rule_based.py`
- 変更:
  - `_opening_orientation` で `internal_wall_x_/internal_wall_y_` 系などの文字列パターンを明示的に解釈するように修正。
  - `partition_x/y`, `shared_wall_x_/y_` など内部壁ラベルにも対応。
- 目的:
  - 文字列由来の壁方向ヒントがある場合に、door/window の向き判定が不安定になる問題を抑制。

### 2. Gemini 中心座標を優先する採択ロジックの追加

- 対象: `experiments/src/step2_rule_based.py`
- 変更:
  - `_accept_opening_candidate_center` を導入。
  - 幾何条件（幅や比率）が厳密一致しない場合でも、中心候補が妥当なら Gemini の中心を採用。
  - 中心のみ採択した場合でも `_orientation_from_bbox` で bbox 形状から向きを更新。
- 目的:
  - 「Gemini 側で opening を正しく検出しているのに、Step3 で位置がずれる」ケースを減らす。

### 3. 壁線へのスナップ処理を追加

- 対象: `experiments/src/step2_rule_based.py`
- 変更:
  - `_collect_wall_lines_from_rooms` を追加し、部屋ポリゴンから壁線情報を収集。
  - `_snap_opening_center_to_wall_lines` を追加し、opening 中心を最寄り壁線へ投影。
- 目的:
  - 開口部中心が壁から浮く・室内側へ寄り過ぎる症状を低減。

### 4. Komazawakoen_B1 向けに採択許容域を調整

- 対象: `experiments/src/step2_rule_based.py`
- 変更:
  - `_accept_opening_candidate_center` の許容距離を、壁ロール（外壁/内壁）に応じて拡張。
  - 中心採択時の向き更新を強制。
- 目的:
  - `komazawakoen_B1` の door_1 で発生していた中心ズレを抑える。

## 実行運用

- `Step1/Step2` は再実行せず、既存の以下を利用:
  - `stage1_output_parsed.json`
  - `gemini_spatial_output.json`
  - `gemini_room_inner_frame_output.json`
  - `gemini_openings_output.json`（存在時は `gemini_openings_output_refined.json` 優先）
- そのうえで各ケースごとに以下を再生成:
  - `layout_generated.json`
  - `metrics.json`
  - `plot_with_bg.png`
- 最後に batch summary を再生成:
  - `plot_with_bg_batch_summary.png`
  - `c_vis_area_batch_summary.png`
  - `c_vis_start_area_batch_summary.png`
  - `c_vis_objects_area_batch_summary.png`
  - `c_vis_start_objects_area_batch_summary.png`

## 反映先

- Run root:
  - `experiments/runs/batch_v2_gpt_high_e2e_rerun_20260302`
- 対象ケース:
  - `kugayama_A_18_5.93x3.04_v2`
  - `suginamikamiigusa_C_30_5.84x5.14_v2`
  - `oizumi-gakuen_C_24_5.19x4.63_v2`
  - `komazawakoen_B1_30_5.1x5.88_v2`
  - `suginamikamiigusa_B_23_5.75x4.00_v2`

## 補足

- 本更新は「LLM再実行なし」の後段再構築であり、APIコストを増やさずに Step3 側ロジック差分の反映を行った。
- 評価設定は `experiments/configs/eval/eval_v1.json` 固定運用（凍結）を前提とする。
