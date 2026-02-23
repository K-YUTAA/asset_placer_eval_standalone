# 実装報告: Gemini Step1化 + bbox/rotation 分離（2026-02-19）

- 作成日: 2026-02-19
- 対象ブランチ: `feature/step2-rule-based`
- 目的:
  - OpenAI が使えない状況でも実験を継続できるようにする
  - 家具 bbox（上面サイズ）と `rotationZ`（機能的正面）を分離して扱う
  - Gemini 検出結果を Step2 で強制反映し、再解釈によるズレを抑える

## 1. 今回実装した内容

## 1.1 Step2 ルールベース実装

- 追加: `experiments/src/step2_rule_based.py`
- 反映内容:
  - 家具の位置・サイズは Gemini bbox を優先採用
  - 内枠ヒント（`main_room_inner_boundary_hint`）を outer/座標変換に利用
  - 開口部（door/window）は Gemini 検出を使って中心・幅を補正
  - `rotationZ` は Step1 `front_hint` 優先（欠損時のみ補完）
  - `search_prompt` は Step1 を優先、欠損時はカテゴリ別デフォルト

## 1.2 Step2 モード切替

- 更新: `experiments/src/generate_layout_json.py`
- 更新: `experiments/src/run_generate_and_eval.py`
- 追加オプション:
  - `--step2_mode {llm,rule}`
- 動作:
  - `rule` では Step2 LLM を呼ばず、`step2_rule_based.py` で最終JSONを生成
  - `generation_manifest.json` に `step2_mode` を記録

## 1.3 Step1/Step2 の LLM プロバイダ切替（Gemini 対応）

- 更新: `experiments/src/generate_layout_json.py`
- 更新: `experiments/src/run_generate_and_eval.py`
- 追加オプション:
  - `--step1_provider {openai,gemini}`
  - `--step2_provider {openai,gemini}`
- Gemini 呼び出し:
  - `generateContent` を直接呼ぶ経路を追加
  - 画像付き Step1 入力を Gemini で実行可能化
- 今回の主運用:
  - `step1_provider=gemini`
  - `step2_mode=rule`

## 1.4 bbox と rotation の分離（再発修正）

課題:
- セグメンテーション側の bbox は正しいのに、最終 plot で回転して見えるケースがあった

対応:
- `experiments/src/step2_rule_based.py`
  - `Length/Width` は Gemini bbox の値をそのまま採用（`rotationZ` で入れ替えない）
  - ドアは向きごとに軸平行寸法を明示
  - 出力 `size_mode` を `world` に統一
- `experiments/src/layout_tools.py`
  - `size_mode=world` のとき、評価用占有ジオメトリは回転を適用しない（軸平行扱い）
  - 機能的向きは `functional_yaw_rad` として保持

## 2. 実行・検証結果（最新）

代表実行ディレクトリ:
- `experiments/runs/e2e_gemini3_openings_case2_rule_step1gemini_bboxfix2`

生成物:
- `layout_generated.json`
- `generation_manifest.json`
- `metrics.json`
- `plot_with_bg.png`

主要指標（latest）:
- `validity`: `1`
- `C_vis`: `0.9475`
- `R_reach`: `0.9863`
- `Delta_layout`: `0.0`

補足:
- OpenAI 疎通は最小プロンプトでも `429 insufficient_quota` を再現
- そのため、今回の連続検証は Gemini 中心で実行

## 3. 変更ファイル一覧

- `experiments/src/step2_rule_based.py`（新規）
- `experiments/src/generate_layout_json.py`
- `experiments/src/run_generate_and_eval.py`
- `experiments/src/layout_tools.py`

## 4. 既知の注意点

- 現在の評価/可視化系は、`size_mode=world` で bbox を軸平行として扱う
- これは「bbox と機能向きの分離」には有効だが、USD 実配置で同じ解釈を採用するかは別途設計が必要
- USD 連携時は以下を明示的に決める必要がある:
  - 実配置衝突判定に `rotationZ` を使うか
  - 使う場合の bbox/footprint の定義（軸平行か回転OBBか）
  - 評価系との整合ポリシー

## 5. 次アクション案

1. USD 側の入力仕様（`size_mode`, `rotationZ`, footprint 定義）を先に固定する  
2. その仕様に合わせて `layout_tools` の変換ルールを分岐化する  
3. 同一ケースで `plot_with_bg` と USD 実配置結果を突き合わせる
