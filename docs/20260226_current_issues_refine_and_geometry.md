# 現在の問題点整理（Refine / Geometry / Stress, 2026-02-26）

## 1. 対象範囲

このメモは、以下の実行結果と実装を対象に、現時点の問題点を整理したものです。

- 実行結果:
  - `experiments/runs/stress_cases_v2_from_batch_v2_gpt_medium_e2e_rerun_20260226`
- 関連比較:
  - `refine_compare_heuristic_vs_proposed.json`
- 関連実装:
  - `experiments/src/refine_heuristic.py`
  - `experiments/src/refine_proposed_beam.py`
  - `experiments/src/eval_metrics.py`
  - `experiments/src/generate_stress_cases.py`


## 2. 現在の主要問題

## 2.1 壁貫通・幾何整合の問題

- 症状:
  - Refine後に家具が壁際で不自然な位置になるケースがある
  - 一部で「壁貫通」に見える配置が発生する
- 影響:
  - 評価結果の信頼性低下
  - 提案手法比較以前に、制約充足の妥当性が揺らぐ

## 2.2 候補可否判定が評価器ロジックと完全一致していない

- 症状:
  - refine側の quick constraints と、`eval_metrics.py` 側の実際の占有/経路判定が二重管理
  - 可視化上OKでも、評価上の制約解釈とズレる可能性がある
- 影響:
  - 受理された候補の幾何妥当性が一貫しない

## 2.3 衝突判定が近似的

- 症状:
  - 一部処理でAABB近似を利用しており、回転OBBの取りこぼしが起きうる
- 影響:
  - 見かけ上の非貫通でも、厳密には干渉している候補が残る可能性がある

## 2.4 Proposedの改善幅が小さい

- 現状結果（`refine_compare_heuristic_vs_proposed.json`）:
  - `improved_core`: heuristic `2` → proposed `3`
  - `improved_entry`: heuristic `2` → proposed `3`
  - 改善はあるが限定的
- 症状:
  - 提案手法としての差が小さい
- 影響:
  - 研究としての主張が弱くなる

## 2.5 Clutterケースでの回復が弱い

- 症状:
  - `clutter` で改善しないケースが多い
  - 追加障害物を固定物として扱うため、局所探索で回復できる自由度が不足
- 影響:
  - stressケースの難度に対して、refineが有効に機能しない


## 3. すでに解消済みの関連問題

- stress可視化で向きが崩れる問題は修正済み
  - contract正規化で `functional_yaw_rad` 保持
  - `plot_layout_json.py` で `functional_yaw_rad` 優先描画


## 4. 根本原因（現時点の見立て）

1. 幾何制約レイヤが分散している
2. refine側の候補受理条件が保守的かつ近似的で、探索可能領域が狭い
3. proposal探索の前提となる「幾何妥当な遷移空間」が十分に厳密でない
4. clutterの難度設計に対して、現在の行動集合と制約が適合していない


## 5. 優先度つき対応方針

## 5.1 最優先（幾何制約の統一）

1. refine候補の受理判定を `eval_metrics` の占有/有効性判定に合わせる
2. 壁・内壁・開口keepout違反をハード制約として明示
3. 衝突判定をAABB依存から脱却し、OBBまたは占有グリッド基準へ統一

## 5.2 次点（提案手法の有効化）

1. Proposedの目的関数を scenario-aware にする
2. 探索深さ/行動幅を clutterに合わせて調整する
3. ただし、幾何制約統一後に再評価する

## 5.3 検証運用

1. 幾何制約統一版で stress 15ケース再実行
2. `plot_with_bg_refined_*` で目視確認
3. `refine_compare_heuristic_vs_proposed.json` を再生成


## 6. 完了判定（DoD）

1. 壁貫通・明確な干渉が可視化で再現しない
2. refine候補の受理ロジックと評価ロジックが矛盾しない
3. Proposedがheuristicより統計的に改善を示す
4. 結果が再実行で再現可能

