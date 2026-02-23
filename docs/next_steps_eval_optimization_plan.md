# 次ステップ実装計画（評価基盤→比較実験→最適化）

## 目的
- 既存の評価基盤（`eval_metrics.py`）を研究用途で再現可能な状態に固定する。
- 4条件比較（Original / LLM-Refine / Heuristic / Proposed）を同一条件で回せる実験導線を作る。
- 最適化（Proposed）の有効性を示せるよう、意図的な失敗ケースを先に整備する。

## 現状認識
- スカラー指標は安定して返せる（`C_vis`, `C_vis_start`, `R_reach`, `clr_min`, `Delta_layout`, `Adopt`）。
- start/goal の自動生成は動作している。
- OOE（entry observability）は per-object 出力まで取得済み。
- 評価1試行が軽いため、探索型最適化を実装可能。
- 一部ケースでベースラインが既に良く、改善効果が見えにくい可能性がある。

## 優先順位（必須）

## 1. 評価仕様の凍結（最優先）
### 固定する項目
1. Task定義（start / goal）
- start: 入口ドア中心から室内側へオフセット（既定 0.4m）+ freeセルへのsnap
- goal: ベッドサイド（既定 0.6m）+ freeセルへのsnap

2. 指標セット
- コア: `R_reach`, `clr_min`, `C_vis`, `Delta_layout`
- 入口観測: `C_vis_start`, `OOE_*`

3. Adopt判定
- 閾値（`tau_R`, `tau_clr`, `tau_vis`, `tau_Delta`）の固定
- 入口観測を Adopt 判定に入れるかを明確化（入れる/入れないを固定）

### DoD
- `configs/eval_v1.json`（名称は実装時に最終決定）に閾値と定義が固定される。
- runごとに同一設定が manifest / config で追跡可能。

## 2. 失敗ケース（Adopt=0）を意図的に作る
### ケース設計
- 4部屋 × 3悪化シナリオを自動生成（合計12）
- 悪化シナリオ:
  - Bottleneck: 経路上を狭くして `clr_min` を低下
  - Occlusion at Entry: 入口近傍を遮蔽して `C_vis_start` / `OOE` を低下
  - Clutter: 小物密度を上げて `R_reach` を低下

### DoD
- 合計16ケース（通常4 + 悪化12）を生成できる。
- 少なくとも半数で Adopt=0 もしくは入口観測の有意な悪化が確認できる。

## 3. 4条件比較ランナーを先に完成
### 比較条件
1. Original（v0）
2. LLM-Refine（指標フィードバック型）
3. Heuristic（ルール補正）
4. Proposed（最適化）

### DoD
- 同一 `layout_id` で4条件を実行し、`metrics.csv` に1行ずつ追記できる。
- 16ケース×4条件のバッチ実行が可能。

## Proposed（最適化）実装方針

## 4. 制約付き最適化として定式化
- 目的（例）:
  - `C_vis`, `C_vis_start`, `OOE` を上げる
  - `Delta_layout` を下げる
- 制約:
  - `R_reach >= tau_R`
  - `clr_min >= tau_clr`
  - `validity = 1`

## 5. 探索手法（初版）
- Beam Search を初版採用
- 行動: 家具の `(±dx, ±dy, ±dtheta)`
- 変更予算: 動かす家具数 K（例: 2〜4）
- Beam幅 B（例: 10〜30）

## 6. 候補家具の優先付け
- `clr_min` 悪化時: ボトルネック近傍家具を優先
- `C_vis_start` / OOE 悪化時: 入口遮蔽家具を優先
- 衝突時: overlap大の家具を優先

## 7. OOE主指標の確定（要決定）
- カメラ認識寄り: `OOE_*_surf` を主指標
- LiDAR寄り: `OOE_*_hit` を主指標

### 決定が必要な理由
- 同一レイアウトでも `hit` と `surf` で評価傾向が変わるため、研究上の解釈が変わる。

## 実装順（短期）
1. `eval_v1` 設定ファイル作成・適用
2. 失敗ケース生成スクリプト実装
3. 4条件比較ランナーで `metrics.csv` 生成
4. Proposed（Beam Search）実装
5. 重み・閾値の感度分析

## 成果物（この計画で最終的に揃えるもの）
- 固定評価設定ファイル
- ケースセット（通常 + 劣化）
- 4条件比較の集計CSV
- 各条件の可視化（`plot_with_bg`、`c_vis*`、`c_vis_objects*` サマリー）
- 実験報告MD（条件・設定・結果・考察）
