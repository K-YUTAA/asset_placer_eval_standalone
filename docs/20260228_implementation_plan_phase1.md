# 実装計画書（Phase1）

## 背景
現在の課題は以下の4点。
1. 壁・占有・描画の不整合（見た目と評価がずれる）
2. `proposed` の改善量が小さい
3. ドア位置・`room_inner_frame` の安定性
4. stress ケースの説明可能性不足

Phase1 では、後続改善の前提になる「評価信頼性」と「実験説明性」を先に固める。

## 目的（Phase1）
1. stress ケースごとの意図と結果をファイルで明示化する。
2. plot 時に eval 側のグリッド情報を使えるようにして、整合確認を容易にする。
3. `eval_v1` 凍結運用を開始するための実装上の接続を整える。

## 実装範囲

### A. stress manifest の標準化（生成器）
- 対象: `experiments/src/generate_stress_cases.py`
- 変更:
  - 各 variant 出力ディレクトリに `stress_manifest.json` を保存
  - 必須キー:
    - `case_id`
    - `variant`
    - `stress_type`
    - `target_metric`
    - `source_layout_path`
    - `base_metrics`
    - `variant_metrics`
    - `delta_metrics`
    - `actions`
    - `status`
    - `score`
  - 既存 `disturb_manifest.json` / run-root `stress_cases_manifest.json` は維持

### B. plot/eval 整合確認オプション追加
- 対象: `experiments/src/plot_layout_json.py`
- 変更:
  - 新規CLI:
    - `--eval_debug_dir`（`eval_metrics.py --debug_dir` で生成したディレクトリ）
    - `--show_eval_occupancy`（`occupancy.pgm` を薄く重畳）
    - `--show_eval_bounds`（`path_cells.json` の `bounds` を優先）
  - 目的:
    - eval の occupancy と plot のズレを目視検証可能にする
    - 既存挙動を壊さない（オプション未指定時は現状維持）

### C. pipeline 側の引数パススルー
- 対象: `experiments/src/run_pipeline_from_json.py`
- 変更:
  - `plot_args.eval_debug_dir` が未指定なら `out_dir/debug` を自動採用
  - `plot_args.show_eval_bounds / show_eval_occupancy` を `plot_layout_json.py` に渡せるようにする

## 非対象（Phase1ではやらない）
- `proposed beam` の再スコアリング調整
- ドア割当アルゴリズム再設計
- `room_inner_frame` 推論プロンプト再設計

## 完了条件（DoD）
1. stress 出力15ケースすべてに `stress_manifest.json` が存在する。
2. `plot_layout_json.py` が `--eval_debug_dir` 経由で eval bounds/occupancy を可視化できる。
3. `run_pipeline_from_json.py` から上記オプションを設定JSONで有効化できる。

## 実装順序
1. `generate_stress_cases.py`（manifest）
2. `plot_layout_json.py`（eval整合可視化）
3. `run_pipeline_from_json.py`（引数接続）

## リスクと対応
- リスク: 可視化が重くなる
  - 対応: `--show_eval_occupancy` は明示指定時のみ有効
- リスク: 既存run互換
  - 対応: 新規引数はすべて optional、未指定時は既存挙動

