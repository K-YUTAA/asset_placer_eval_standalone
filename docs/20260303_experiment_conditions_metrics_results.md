# 2026-03-03 実験条件・評価項目・実験結果まとめ

## 1. 実験目的

- `eval_v1` を凍結設定として、stress レイアウト（`base/bottleneck/occlusion/clutter`）に対する評価挙動を確認する。
- clearance 指標を二本立てで扱う:
- `clr_min_astar`（経路依存）
- `clr_feasible`（幾何・連結性ベース）
- 採否（`Adopt_core`）を `clr_feasible` 基準に切替した影響を確認する。

## 2. 実験データと実行対象

- 元レイアウト（5件）:
- `kugayama_A_18_5.93x3.04_v2`
- `suginamikamiigusa_C_30_5.84x5.14_v2`
- `oizumi-gakuen_C_24_5.19x4.63_v2`
- `komazawakoen_B1_30_5.1x5.88_v2`
- `suginamikamiigusa_B_23_5.75x4.00_v2`

- stress 生成:
- `base + bottleneck + occlusion + clutter`（各ケース4 variant）
- 合計 `20 layouts`

- 実行出力:
- run root: `experiments/runs/stress_eval_v1_clrdual_20260302`
- manifest: `experiments/runs/stress_eval_v1_clrdual_20260302/stress_cases_manifest.json`

## 3. 実験条件（固定）

### 3.1 評価設定（`eval_v1`）

参照: `experiments/configs/eval/eval_v1.json`

- `grid_resolution_m`: `0.05`
- `robot_radius_m`: `0.28`
- `clr_feasible_max_m`: `2.0`
- `clr_feasible_tol_m`: `0.01`
- `clr_feasible_max_iters`: `14`
- task:
- start: `entrance_slidingdoor_center`, `in_offset_m=0.50`
- goal: `bedside`, `offset_m=0.60`, `choose=closest_to_room_centroid`
- `tau_R=0.90`, `tau_V=0.60`, `tau_Delta=0.10`
- `tau_clr_feasible=0.10`, `tau_clr_astar=0.10`
- `adopt.clearance_metric="clr_feasible"`（採否に使用）
- entry gate:
- `enabled=true`
- `metric=OOE_R_rec_entry_surf`
- `min_value=0.70`

### 3.2 stress 設定（`degrade_v1`）

参照: `experiments/configs/degrade/degrade_v1.json`

- movable categories:
- `chair, table, coffee_table, small_storage, storage, tv_cabinet, cabinet, sofa`
- fixed categories:
- `bed, toilet, sink, door, window, opening, floor`
- 共通制約:
- `same_room_only=true`
- `translation_max_m=0.6`
- `rotation_max_deg=30`
- `door_keepout_radius_m=0.5`
- `overlap_ratio_max=0.05`
- variant 別:
- `bottleneck`: 可動家具1つ移動、clearance悪化狙い
- `occlusion`: 可動家具1つ移動、入口観測悪化狙い（reach/clearance維持）
- `clutter`: 障害物追加（最大2個）

## 4. 評価項目の定義

### 4.1 コア指標

- `R_reach`: 自由セルのうち、start から到達可能な割合
- `C_vis`: 経路上センサ点から可視な自由セル割合
- `C_vis_start`: start 点のみから可視な自由セル割合
- `Delta_layout`: baseline からの変更量

### 4.2 clearance 指標（二本立て）

- `clr_min_astar`:
- A* 経路（raw path）上での最小余裕
- 各経路セルの `distance_to_occupied - robot_radius` の最小値
- 経路選択に依存する

- `clr_feasible`:
- 追加クリアランス `c` を仮定して障害物を `robot_radius + c` まで膨張
- それでも start-goal が連結な最大 `c`
- 幾何・連結性ベース（プランナ非依存）

### 4.3 入口観測（OOE）

- `OOE_C_obj_entry_hit`, `OOE_R_rec_entry_hit`
- `OOE_C_obj_entry_surf`, `OOE_R_rec_entry_surf`
- 本実験の entry gate は `OOE_R_rec_entry_surf` を使用

### 4.4 採否判定

- `Adopt_core = 1`  iff  
  `R_reach >= tau_R` かつ `clearance_metric_value >= clearance_threshold` かつ `C_vis >= tau_V` かつ `Delta_layout <= tau_Delta`
- 本実験では `clearance_metric = clr_feasible`

- `Adopt_entry = 1` iff  
  `Adopt_core = 1` かつ `OOE_R_rec_entry_surf >= 0.70`

- `Adopt` は互換のため `Adopt_entry` と同義で出力

## 5. 実験結果

集計元:
- `experiments/runs/stress_eval_v1_clrdual_20260302/adopt_core_clr_feasible_summary.csv`

### 5.1 variant別集計（base比, n=5ずつ）

| variant | n | core up | core down | core same | entry up | entry down | avg Δclr_feasible | avg Δclr_min_astar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottleneck | 5 | 0 | 1 | 4 | 0 | 1 | -0.0521 | -0.0376 |
| occlusion | 5 | 0 | 0 | 5 | 0 | 0 | 0.0000 | 0.0000 |
| clutter | 5 | 0 | 0 | 5 | 0 | 1 | -0.0766 | -0.1097 |

### 5.2 ケース別の主な変化

- `suginamikamiigusa_B_23_5.75x4.00_v2` の `bottleneck` で `Adopt_core: 1 -> 0`
- 同ケースの `clutter` で `Adopt_entry: 1 -> 0`（coreは維持）
- `occlusion` は全ケースで `Adopt_core` 変化なし（設定どおり clearance 非悪化）

## 6. 生成物

- 可視化サマリー:
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_base.png`
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_bottleneck.png`
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_occlusion.png`
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_clutter.png`
- `experiments/runs/stress_eval_v1_clrdual_20260302/plot_with_bg_summary_all_variants.png`

- 集計表:
- `experiments/runs/stress_eval_v1_clrdual_20260302/adopt_core_clr_feasible_summary.md`
- `experiments/runs/stress_eval_v1_clrdual_20260302/adopt_core_clr_feasible_summary.csv`

## 7. 結論（今回時点）

- 採否判定は `clr_feasible` へ移行済みで、`clr_min_astar` は補助指標として機能している。
- stress の種類ごとに意図した劣化傾向が概ね分離できている:
- bottleneck: clearance 主体で悪化
- occlusion: 入口観測主体で悪化
- clutter: 入口採否/経路余裕に影響
