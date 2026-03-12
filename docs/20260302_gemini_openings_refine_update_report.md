# Gemini開口部改善の実装報告（2026-03-02）

## 背景
- `suginamikamiigusa_B` で、Sliding Door の「開口部」ではなく「ドア収納部（ポケット側）」まで検出される問題があった。
- 目的は、後段で強制補正しすぎず、Gemini 側の認識品質を上げること。

## 実装方針
- やりすぎな分割推論（開口ごとのROI推論）は採用せず、
- 既存フローに対して **開口部だけ追加1パス** で自己修正する方式を採用した。

## 実装内容

## 1. 開口部プロンプトの明確化
- ファイル: `prompts/fixed_mode_20260222/gemini_openings_prompt.txt`
- 追加:
  - 開口中心は wall-break gap 上に置く
  - 過大幅（壁/ポケット/収納領域の巻き込み）を禁止

## 2. 開口部2パス（Refine pass）を追加
- ファイル: `experiments/src/generate_layout_json.py`
- 追加:
  - `--gemini_openings_refine_pass` フラグ
  - Pass1 の候補を列挙して Gemini に再提示し、開口部のみ再評価するプロンプト生成
  - 出力: `gemini_openings_output_refined.json`（使用時）
- 仕様:
  - Step1 への依存は最小化し、主判断は Gemini + 画像
  - Step1 は最終強制ではなく補助的情報

## 3. 開口部品質評価の強化
- ファイル: `experiments/src/generate_layout_json.py`
- 追加:
  - `max_outer_door_width_ratio` を評価条件に追加
  - 既存の `min_outer_door_width_ratio`, `max_outer_center_dist_m` と合わせて、過小/過大/中心ズレを判定

## 4. Step2 側の安全ガード追加
- ファイル: `experiments/src/step2_rule_based.py`
- 追加:
  - Gemini 開口候補を採用する前に幾何条件を検証
  - 条件不一致なら Step1 開口へフォールバック
  - Sliding Door の過大幅候補を弾くガードを追加

## 5. 可視化整合の補助（同日実装）
- ファイル: `experiments/src/plot_layout_json.py`
- 追加:
  - `--eval_debug_dir`, `--show_eval_bounds`, `--show_eval_occupancy`, `--eval_occupancy_alpha`
  - eval occupancy を重畳して、plot/eval のズレ確認を容易化

## 実施した反映（対象ケース）
- ケース: `suginamikamiigusa_B_23_5.75x4.00_v2`
- Run root:
  - `experiments/runs/batch_v2_gpt_high_e2e_rerun_20260302/suginamikamiigusa_B_23_5.75x4.00_v2`
- 実行:
  - OpenAI は再実行せず（`analysis_input=stage1_output_parsed.json`）
  - Gemini 3-pass + openings refine pass を再実行
  - `layout_generated.json` → `metrics.json` → `plot_with_bg.png` を更新

## 主な更新ファイル（ケース出力）
- `gemini_openings_output.json`
- `gemini_openings_output_refined.json`
- `layout_generated.json`
- `metrics.json`
- `plot_with_bg.png`
- `generation_manifest.json`（使用プロンプト/設定履歴）

## 既知事項
- 改善は確認できるが、外側 sliding door 幅がケースによりわずかに過大になる余地がある。
- 必要に応じて `max_outer_door_width_ratio` をさらに厳格化して再調整可能。

## まとめ
- 今回の変更で、Gemini 開口部検出の責務を強化しつつ、後段の過剰な強制補正を減らす方向へ移行できた。
- 既存フローとの互換を保ったまま、開口部品質改善のための実行可能な拡張（追加1パス）を導入した。

