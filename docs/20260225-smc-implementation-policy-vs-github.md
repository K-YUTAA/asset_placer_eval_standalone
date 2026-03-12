# 今後の実装方針（SMC向け）: GitHub現状比較版

## 0. 比較条件

- 方針元: `research/snippets/20260224-smc-direction-next-actions.md`
- 進捗元: `research/legacy/slides_md/20260223_AI_北野雄大_.md`
- 一次参照GitHub: `https://github.com/K-YUTAA/asset_placer_isaac`
- 比較対象ブランチ:
  - `origin/master`: `d3af84c9f2af08216b1dd27ab91490a3c4295e5b`
  - `origin/exp/eval-loop-v1`: `3944d56f058f5b583579e80038c14df081b4faab`
- 差分レポート: `research/snippets/20260225-github-progress-diff.md`

## 1. SMC方針とのギャップ（結論）

1. `master` には導入可否のコア評価（`R_reach`, `clr_min`, `C_vis`, `Delta_layout`, `Adopt`, `validity`）と task start/goal 自動解決はある。
2. 入口観測系（`C_vis_start`, `OOE_*`）は `master` には未導入で、`exp/eval-loop-v1` にのみ存在。
3. `Adopt_core` / `Adopt_entry` の二段判定は未実装（両ブランチとも）。
4. 実験2本目の本命に必要な `controlled perturbation`（Bottleneck/Occlusion/Clutter）生成器は未実装。
5. `method=proposed` は実質 `heuristic` と同じ経路で、Beam Search提案法は未実装。
6. `metrics.csv` は旧列中心で、入口観測系・二段判定・disturb/repair分離を記録できない。

## 2. 現状比較（ファイル基準）

### 2.1 すでに使える土台（master）

- 実験ワークスペースと再実行基盤:
  - `experiments/README.md`
  - `experiments/src/run_v0_freeze.py`
  - `experiments/src/run_trial.py`
- task point 自動解決:
  - `experiments/src/task_points.py`
  - `experiments/src/eval_metrics.py`（`eval.task` の解決とスナップ）
- コア評価指標:
  - `experiments/src/eval_metrics.py`（`R_reach`, `clr_min`, `C_vis`, `Delta_layout`, `Adopt`, `validity`）
  - `experiments/configs/eval/default_eval.json`

### 2.2 先行実装があるが未統合（exp/eval-loop-v1）

- 入口観測系:
  - `experiments/src/eval_metrics.py` に `C_vis_start`, `OOE_*`, `entry_observability` ロジック
  - `experiments/configs/eval/default_eval.json` に `entry_observability` 設定
- 代表コミット:
  - `a48b72c` (`experiments: add entry observability metrics at room entry`)

### 2.3 未実装（SMC方針で必要）

- `Adopt_core` と `Adopt_entry` の明示分離
- stress-test ケース自動生成（通常5件→悪化15件）
- `proposed` を独立アルゴリズム化（Beam Search）
- `LLM指示リファイン` の比較条件固定
- `disturb量` と `repair量` の分離記録
- 実験1/2を自動で集計・図表化するスクリプト

## 3. 実装方針（実行順）

### Phase 1: 評価器の固定（`eval_v1` 凍結）

- 追加/更新:
  - `experiments/configs/eval/eval_v1.json` を新規作成
  - `experiments/src/eval_metrics.py` に以下を追加
    - `Adopt_core`
    - `Adopt_entry`（`entry_observability.enabled=true` 時のみゲート追加）
    - 互換維持のため `Adopt` は残す（`Adopt_core` と同義または明示規約化）
- 完了条件:
  - 同一入力で再実行して、`metrics.json` に `Adopt_core/Adopt_entry/validity` が必ず出力される。

### Phase 2: stress-test 生成器（実験2の入力基盤）

- 新規:
  - `experiments/src/generate_stress_cases.py`
- 仕様:
  - 通常5件から `Bottleneck/Occlusion/Clutter` を各1種以上生成し合計20ケース化
  - 可動家具のみ摂動、`validity=1` を維持
  - seed・摂動対象・摂動量を manifest に保存
- 完了条件:
  - 20ケースを再生成して同一seedで同一結果になる。

### Phase 3: 手法比較ランナーの分離（Original/LLM/Heuristic/Proposed）

- 更新:
  - `experiments/src/run_trial.py`
  - `experiments/configs/trials/*.json`
- 仕様:
  - `method=proposed` を `heuristic` と別実装に切り分け
  - `LLM指示リファイン` の入出力を固定（キャッシュ前提で再現可能に）
  - `metrics.csv` 列を拡張（`C_vis_start`, `OOE_*`, `Adopt_core`, `Adopt_entry`, `disturb_delta`, `repair_delta`）
- 完了条件:
  - 4手法が同一ケース群で自動実行でき、1つのCSVに統合される。

### Phase 4: 提案法（Beam Search）

- 新規:
  - `experiments/src/refine_beam.py`
- 更新:
  - `experiments/src/refine_heuristic.py`（heuristic専用化）
  - `experiments/src/run_trial.py`（method dispatch）
- 目的関数:
  - 改善: `R_reach`, `clr_min`, `C_vis`, `C_vis_start`, `OOE`
  - 制約: `validity=1`, 非退行ガード
  - コスト: `Delta_layout` 最小化
- 完了条件:
  - `proposed` の探索ログ（候補数・採択履歴）が保存され、`heuristic` と差分比較可能。

### Phase 5: 集計・図表化（論文直結）

- 新規:
  - `experiments/src/aggregate_results.py`
- 出力:
  - 実験1: Fidelity表
  - 実験2: Adopt率比較、入口観測改善量、修正コスト（disturb/repair分離）
- 完了条件:
  - SMC本文に貼れる表/図の元データをワンコマンドで再生成できる。

## 4. ブランチ運用方針（衝突回避）

1. ベースは `origin/master`。
2. `entry_observability` は `exp/eval-loop-v1` から `experiments` 関連のみ選択取り込み（全面マージしない）。
3. 取り込み候補の最小単位は `a48b72c` を起点に、`experiments/src/eval_metrics.py` と `experiments/configs/eval/default_eval.json` を優先。
4. extension本体（`my/research/...`）の変更は本タスクから切り離し、評価ループの再現性を先に確保する。

## 5. 直近の実装TODO（そのまま着手順）

1. `eval_v1.json` を新設してしきい値と entry gate を固定。
2. `eval_metrics.py` に `Adopt_core/Adopt_entry` を追加。
3. `generate_stress_cases.py` で20ケース生成を自動化。
4. `run_trial.py` の method を4条件に分離し、`metrics.csv` 列を拡張。
5. `refine_beam.py` を実装し、`proposed` を heuristic から独立。
6. `aggregate_results.py` で実験1/2の表を自動出力。

## 6. 今回の判断

- SMC方針とGitHub現状を比較すると、最優先の不足は「入口観測系のmaster統合」と「proposed独立実装（Beam）」。
- したがって、次の1コミット群は `eval_v1凍結 + Adopt_core/entry + stress-test生成器` を最小セットにする。
