# 現行 Refine 手法 実装サマリ（2026-03-02）

## 目的

本ドキュメントは、現行コードで実装済みのレイアウト最適化（refine）手法を実装ベースで整理する。

対象コード:

- `experiments/src/run_trial.py`
- `experiments/src/refine_heuristic.py`
- `experiments/src/refine_proposed_beam.py`
- `experiments/configs/refine/proposed_beam_v1.json`

## 全体像（method dispatch）

`run_trial.py` の `method` 分岐は以下。

- `original`: refine なし（v0出力をそのまま評価）
- `heuristic`: `refine_heuristic.run_refinement(...)`
- `proposed`: `refine_proposed_beam.run_refinement(...)`

参照: `experiments/src/run_trial.py:140`, `experiments/src/run_trial.py:155`

## 共通フロー

1. `run_v0_freeze` で初期レイアウト（baseline）を生成/読込
2. `evaluate_layout` で baseline の task points を解決し、`cfg.start_xy/goal_xy` を補正
3. refine 実行（`heuristic` or `proposed`）
4. refine 結果を再度 `evaluate_layout` し `metrics.json` を確定
5. `trial_manifest.json` と `metrics.csv` へ記録

参照: `experiments/src/run_trial.py:126`, `experiments/src/run_trial.py:133`, `experiments/src/run_trial.py:180`

## 手法1: Heuristic Local Search

実装: `experiments/src/refine_heuristic.py`

### 探索単位

- 毎反復で対象物体を1つ選ぶ
- 近傍行動は `(dx, dy, dtheta)` の3値グリッド:
  - `dx ∈ {-step, 0, +step}`
  - `dy ∈ {-step, 0, +step}`
  - `dtheta ∈ {-rot, 0, +rot}`
  - ゼロ移動を除くので 26 候補/反復

参照: `experiments/src/refine_heuristic.py:165`

### 対象物体選択

`_select_target_object(...)` の優先ロジック:

1. `bottleneck_cell` がある場合:
   - ボトルネック位置に最も近い可動物体を選択
2. ない場合:
   - `start-goal` 線分への距離が最小の可動物体を選択
3. すでに `max_changed_objects` に達している場合:
   - 既変更物体の再調整のみ許可

参照: `experiments/src/refine_heuristic.py:68`

### 候補受理条件（ハード）

- 物体OBBの4隅が room polygon 内
- `evaluate_layout` 結果 `validity == 1`
- Non-regression:
  - `R_reach` を悪化させない
  - クリアランス値（`clr_feasible`/`clr_min_astar`/`clr_min`）を悪化させない

参照: `experiments/src/refine_heuristic.py:182`, `experiments/src/refine_heuristic.py:186`, `experiments/src/refine_heuristic.py:190`

### スコア関数

```
score =
  + 1.0 * C_vis
  + 1.0 * R_reach
  + 1.0 * clamp(clearance, 0..1)
  - 0.5 * Delta_layout
  - penalty
```

`penalty`:

- `validity==0` で `+5.0`
- `R_reach<=0` で `+2.0`

参照: `experiments/src/refine_heuristic.py:36`

### 停止条件

- 反復上限 `max_iterations` 到達
- 対象物体が選べない
- 26近傍で改善候補が見つからない

参照: `experiments/src/refine_heuristic.py:150`, `experiments/src/refine_heuristic.py:203`

### 出力

- `layout_refined.json`
- `metrics_refined.json`
- `refine_log.json`（各反復の before/after と move）

参照: `experiments/src/run_trial.py:151`

## 手法2: Proposed Beam Search

実装: `experiments/src/refine_proposed_beam.py`

### 探索空間

- 行動集合は heuristic と同じ 26 近傍
- 深さ `depth`、幅 `beam_width` の Beam Search
- 各状態で上位 `candidate_objects_per_state` 物体のみ展開
- 既訪問状態（量子化 pose）を `visited` で除外

参照: `experiments/src/refine_proposed_beam.py:301`, `experiments/src/refine_proposed_beam.py:329`, `experiments/src/refine_proposed_beam.py:257`, `experiments/src/refine_proposed_beam.py:204`

### 物体候補選択

優先スコア（大きいほど優先）:

```
priority =
  1/(distance to start-goal segment + eps)
  + 0.7/(distance to start + eps)
  + 0.8/(distance to bottleneck + eps)
```

加えて:

- 非可動、`floor/door/window/opening/clutter` は除外
- `max_changed_objects` 超過時は既変更IDのみ

参照: `experiments/src/refine_proposed_beam.py:242`, `experiments/src/refine_proposed_beam.py:257`

### 候補受理条件（ハード）

- 物体OBBが room 内
- `door_keepout_radius_m` 内に入らない（設定時）
- AABB overlap比が `overlap_ratio_max` を超えない
- `validity == 1`
- （任意）`allow_intermediate_regression=false` のとき:
  - `R_reach` と clearance を非悪化制約

参照: `experiments/src/refine_proposed_beam.py:395`, `experiments/src/refine_proposed_beam.py:397`, `experiments/src/refine_proposed_beam.py:401`, `experiments/src/refine_proposed_beam.py:406`, `experiments/src/refine_proposed_beam.py:408`

### スコア設計

`score_tuple = (valid, adopt_entry, adopt_core, continuous)`

- `use_lexicographic=true` 時は tuple の辞書式比較で優劣決定
- `use_lexicographic=false` 時は continuous 主体

参照: `experiments/src/refine_proposed_beam.py:186`

#### continuous score

しきい値マージンを `[-1,1]` に正規化して合成:

```
cont =
  1.0*m_clr
  + 1.0*m_reach
  + 0.2*m_vis
  + 1.0*m_start_vis
  + 1.0*m_ooe_primary
  - delta_weight*Delta_layout
```

ここで `m_x = clip((value - tau)/tau, -1, 1)`。

参照: `experiments/src/refine_proposed_beam.py:152`

### 探索予算

- `max_eval_calls<=0` のとき自動設定:
  - `beam_width * candidate_objects_per_state * 26 * depth`
- 予算到達で探索打ち切り

参照: `experiments/src/refine_proposed_beam.py:338`, `experiments/src/refine_proposed_beam.py:381`

### 出力

- `layout_refined.json`
- `metrics_refined.json`
- `refine_log.json`（`search`層ログ + `best_steps` + `eval_calls`）

参照: `experiments/src/refine_proposed_beam.py:463`, `experiments/src/run_trial.py:175`

## パラメータ注入元

`run_trial.py` は trial config から以下を読み取る。

### heuristic 用

- `refine_max_iterations`（default 30）
- `refine_step_m`（default 0.1）
- `refine_rot_deg`（default 15）
- `refine_max_changed_objects`（default 3）

参照: `experiments/src/run_trial.py:145`

### proposed 用

- `refine_step_m`, `refine_rot_deg`, `refine_max_changed_objects`
- `refine_beam_width`（default 5）
- `refine_depth`（default 3）
- `refine_candidate_objects_per_state`（default 2）
- `refine_eval_budget`（default 0 = 自動）
- `refine_ooe_primary`（default `OOE_R_rec_entry_surf`）
- `refine_use_lexicographic`（default true）
- `refine_allow_intermediate_regression`（default true）
- `refine_door_keepout_radius_m`（default 0.0）
- `refine_overlap_ratio_max`（default 0.05）
- `refine_delta_weight`（default 0.3）

参照: `experiments/src/run_trial.py:160`

固定値サンプル: `experiments/configs/refine/proposed_beam_v1.json`

## 既知の実装上の注意点

1. quick constraints と `eval_metrics` の厳密判定は完全一致ではない  
`proposed` は探索前段で OBB/AABB 近似を使うため、評価器グリッド判定と差が出る可能性がある。

2. `clutter` カテゴリは `proposed` の候補選択対象から除外  
`cat in {"floor","door","window","opening","clutter"}` で除外している。

3. `proposed` の `door_centers` は depth単位で1回計算  
`_collect_door_centers(best.layout)` を層外で計算しており、各ノードの door 配置ではなく「その時点の best」依存。

4. `Adopt` 判定の重心は `evaluate_layout` 側  
refine本体は `metrics` を利用しているだけで、採択ルールの一次定義は evaluator にある。

## まとめ

- 実装済み refine は **2手法**（`heuristic` / `proposed beam`）。
- `heuristic` は「1物体ずつの貪欲局所探索」。
- `proposed` は「候補物体選別 + Beam Search + 辞書式採択」。
- いずれも最終品質評価は `evaluate_layout` に依存し、`Adopt_core/entry` を共通指標として比較可能。
