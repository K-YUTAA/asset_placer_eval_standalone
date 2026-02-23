# Implementation Report: Latest Design Refactor + JSON Driven Execution (2026-02-20)

## 1. 今回やりたかったこと

- 最新設計（GPT Step1 + Gemini Spatial + ルールベース Step2）を、運用しやすい形に整理する。
- 実行時パラメータを CLI 直指定ではなく JSON に集約する。
- 生成・評価・可視化の実行完了判定を、ログ依存ではなく出力ファイル存在で確実化する。

## 2. 今回できたこと

- `experiments/src/run_pipeline_from_json.py` を運用向けに整理。
- `defaults.env` / `cases[].env` を追加し、JSON から環境変数注入可能にした。
- 各ステージの実行ログを `stage:start/done` で明示するようにした。
- 各ステージ終了時に期待ファイルの存在チェックを追加した。
- `eval_args.enabled` を追加し、評価実行の ON/OFF を JSON で制御可能にした。
- `plot` 実行時、`metrics/task_points/path_cells/bg_inner_frame_json` は存在時のみ引き渡す安全仕様にした。
- 最新設計のバッチ設定テンプレートを `experiments/configs/pipeline/latest_design_v2_gpt_high.json` として追加した。

## 3. 改善したこと（Before / After）

- Before: 実行完了を subprocess 戻り値中心で判定していた。
- After: subprocess 完了 + 必須生成物の存在確認で判定するようにした。

- Before: 環境変数はシェル依存だった。
- After: JSON の `defaults.env` / `cases[].env` で明示注入可能にした。

- Before: plot 引数に存在しないファイルパスが混在し得た。
- After: 存在確認済みのファイルのみ plot へ渡すようにした。

## 4. 追加・更新ファイル

- `experiments/src/run_pipeline_from_json.py`
- `experiments/configs/pipeline/latest_design_v2_gpt_high.json`
- `README.md`
- `docs/implementation_report_latest_design_json_refactor_20260220.md`

## 5. 実行方法（JSON駆動）

```bash
uv run python experiments/src/run_pipeline_from_json.py \
  --config experiments/configs/pipeline/latest_design_v2_gpt_high.json
```

## 6. 既知の注意点

- API キーはセキュリティ上、設定JSONへ直書きせずシェル環境変数で渡す運用を推奨。
- `cases[].image_path` と `cases[].dimensions_path` の対応関係は手動管理のため、設定更新時に要確認。
