# Bottleneck / Occlusion / Clutter 実装報告（2026-02-26）

## 1. 目的

本実装の目的は、通常生成レイアウト（5ケース）に対して、評価器の挙動を確認できる再現性の高い stress ケースを自動生成することです。  
シナリオは以下の3種です。

- `bottleneck`: 通路クリアランス悪化
- `occlusion`: 入口観測悪化
- `clutter`: 外部障害物追加による悪化


## 2. 対象コードと主な変更点

### 2.1 stress 生成本体

- `experiments/src/generate_stress_cases.py`

主な変更:

- `clutter` を「既存家具移動」から「外部障害物追加」に変更
- 追加障害物は `category=clutter`, `movable=false` で layout に追加
- `layout` に加えて `clutter_objects` 配列も出力
- `Delta_disturb` を `clutter` 用に拡張（追加障害物分のペナルティを加算）

### 2.2 degrade 設定

- `experiments/configs/degrade/degrade_v1.json`

`clutter` シナリオを以下へ変更:

- `max_added_objects: 2`
- `object_sizes_xy_m: [[0.35,0.35],[0.60,0.40]]`
- `object_height_m: 1.0`
- `delta_disturb_per_added_object: 0.02`

### 2.3 可視化向きの整合修正（stress 表示品質）

- `experiments/src/layout_tools.py`
  - contract 正規化時に `functional_yaw_rad` を保持
- `experiments/src/plot_layout_json.py`
  - contract 描画時の矢印向きを `functional_yaw_rad` 優先に変更


## 3. stress 生成アルゴリズム

## 3.1 共通入力

- `source_root`: 通常生成結果（5ケース）
- `eval_config`: `experiments/configs/eval/eval_v1.json`
- `degrade_config`: `experiments/configs/degrade/degrade_v1.json`

## 3.2 共通制約

`degrade_v1.json` の `constraints`:

- `same_room_only: true`
- `translation_max_m: 0.6`
- `rotation_max_deg: 30.0`
- `door_keepout_radius_m: 0.5`
- `overlap_ratio_max: 0.05`

上記を満たさない候補は棄却。さらに `evaluate_layout` で `validity==1` を必須条件にしています。

## 3.3 Bottleneck

狙い:

- `clr_min` を閾値近傍まで低下させる
- 経路は残す（`require_path: true`）

実装概要:

1. 可動家具候補（`preferred_categories` 優先）を列挙
2. `start->goal` 上の `anchor_t=0.55` 付近をアンカーとして候補配置を生成
3. 候補ごとに `evaluate_layout` 実行
4. `score_bottleneck` でランキングして最良採用

## 3.4 Occlusion

狙い:

- 入口観測系（既定: `OOE_R_rec_entry_surf`）を悪化
- 到達系を維持（`keep_reach_and_clr: true`）

実装概要:

1. 可動家具候補を列挙
2. `anchor_t=0.20` 近傍に候補配置
3. `R_reach >= tau_R` と `clr_min >= tau_clr` を満たす候補のみ有効
4. `score_occlusion` で最良採用

## 3.5 Clutter（今回の中心変更）

狙い:

- 外部障害物（一時的障害）追加で悪化を作る
- 既存家具を動かさず stress を作る

実装概要:

1. `anchor_t_1` 近傍で1個目の clutter 追加候補を探索
2. `max_added_objects>=2` の場合は `anchor_t_2` 近傍で2個目を探索
3. 追加は beam 的に組み合わせ（`clutter_beam_width`）
4. `target_mode=entry_only_or_reach_drop` のため、まず
   - `Adopt_core=1 and Adopt_entry=0` を優先
   - なければ `score_clutter` 最大を採用

追加オブジェクト仕様:

- `id`: `clutter_XX`
- `category`: `clutter`
- `size_lwh_m`: `[L, W, H]`
- `pose_xyz_yaw`: `[x, y, 0.0, yaw]`
- `movable: false`

加えて `clutter_objects` に同等情報を保存:

- `shape`, `size_xy`, `height_m`, `pose_xytheta`, `movable`


## 4. スコアリングの考え方

### 4.1 Bottleneck

- 閾値下回りを強く評価
- `clr_min` の目標近傍を優先
- `Delta_layout` は軽いペナルティ

### 4.2 Occlusion

- primary metric の低下量を最大化
- `Delta_layout` ペナルティで過剰変更を抑制

### 4.3 Clutter

- `entry_only`（`Adopt_core=1, Adopt_entry=0`）を強く優先
- `R_reach` 低下、入口観測低下を加点
- `Delta_layout` ペナルティ


## 5. Delta の扱い

`disturb_manifest.json` では以下を分離:

- `variant_metrics.Delta_layout`: 評価器ベースの変更量
- `Delta_disturb`: stress 用指標
  - 非 clutter: `Delta_layout` と同じ
  - clutter: `Delta_layout + delta_disturb_per_added_object * 追加数`

これにより、障害物追加を含む外乱量を独立に扱えます。


## 6. 出力構造

出力ルート例:

- `experiments/runs/stress_cases_v2_from_batch_v2_gpt_medium_e2e_rerun_20260226`

ケースごとの構成:

- `base/layout_generated.json`
- `bottleneck/layout_generated.json`
- `occlusion/layout_generated.json`
- `clutter/layout_generated.json`
- 各 variant に `disturb_manifest.json`

全体 manifest:

- `stress_cases_manifest.json`
  - `total_base_cases: 5`
  - `total_variants: 20`
  - `scenario_counts: {base:5,bottleneck:5,occlusion:5,clutter:5}`


## 7. 評価可視化（今回作成した成果物）

stress 15ケース（`bottleneck/occlusion/clutter`）について、`eval_metrics.py` を再実行し、評価指標オーバーレイ付き画像を作成済み。

各ケース:

- `*/bottleneck/metrics.json`, `*/bottleneck/debug/*`, `*/bottleneck/plot_with_bg_eval.png`
- `*/occlusion/metrics.json`, `*/occlusion/debug/*`, `*/occlusion/plot_with_bg_eval.png`
- `*/clutter/metrics.json`, `*/clutter/debug/*`, `*/clutter/plot_with_bg_eval.png`

サマリー:

- `plot_with_bg_eval_summary_bottleneck.png`
- `plot_with_bg_eval_summary_occlusion.png`
- `plot_with_bg_eval_summary_clutter.png`
- `plot_with_bg_eval_summary_all_15.png`


## 8. 現状の仕様確定ポイント

- `clutter` は「追加障害物」であり、既存家具移動ではない
- 追加障害物は固定物（`movable=false`）
- 可視化の向きは `functional_yaw_rad` 優先で表示
- stress の背景重ね合わせは `bg_inner_frame_json` を利用して通常バッチと整合


## 9. 今後の拡張候補

- `clutter` のサイズセット・追加数を case 難易度で可変化
- `Delta_disturb` を面積や障害物位置重みで高度化
- `entry_only` ケース比率を目標値で制御する stop 条件の導入

