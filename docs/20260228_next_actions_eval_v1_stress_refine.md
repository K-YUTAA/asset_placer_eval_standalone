# 次アクション実装仕様（eval_v1 / stress manifest / proposed beam）

## 目的
評価の信頼性を先に固定し、その後に `heuristic` と `proposed` の差が明確に出る実験系へ移行する。

## Status note

- This document is a pre-freeze implementation proposal.
- The current implemented freeze spec is `experiments/configs/eval/eval_v1.json`.
- In the current implementation, `Adopt_core` uses `adopt.clearance_metric = "clr_feasible"` and `tau_clr_feasible = 0.10`.
- The old proposal value `adopt.thresholds.clr_min_m = 0.25` below is historical and is not the active rule anymore.

## 優先順
1. 幾何整合（壁・占有・描画の単一化）
2. `eval_v1` 凍結
3. stress manifest 追加
4. `proposed`（Beam Search）再定義
5. 20ケース x 4条件の再実行

---

## 1) eval_v1 凍結仕様

### ファイル
`experiments/configs/eval/eval_v1.json`

### 当時の提案スケッチ（historical, not current freeze spec）
```json
{
  "grid_resolution_m": 0.05,
  "robot_radius_m": 0.30,
  "task": {
    "start_mode": "entrance_slidingdoor_center",
    "start_in_offset_m": 0.40,
    "goal_mode": "bedside",
    "goal_offset_from_bed_edge_m": 0.60,
    "goal_side_rule": "closest_to_room_centroid",
    "snap_to_free": true
  },
  "ooe": {
    "enabled": true,
    "primary_mode": "surf",
    "secondary_log_mode": "hit",
    "tau_p": 0.02,
    "tau_v": 0.30
  },
  "adopt": {
    "thresholds": {
      "R_reach": 0.90,
      "clr_min_m": 0.25,
      "C_vis": 0.70,
      "validity": 1
    },
    "entry_gate": {
      "enabled": true,
      "metric": "OOE_R_rec_entry_surf",
      "min_value": 0.80
    },
    "report_both": true
  }
}
```

### 現行実装との差分メモ

- Current frozen `robot_radius_m` is `0.28`, not `0.30`.
- Current task schema is `task.start.*` / `task.goal.*`, not `start_mode` / `goal_mode` flat keys.
- Current clearance adoption rule is `clr_feasible >= tau_clr_feasible (0.10)`.
- `clr_min_astar` is still recorded, but it is not the active `Adopt_core` clearance metric in `eval_v1`.

### 出力要件
- `metrics.json` に `Adopt_core` と `Adopt_entry` を両方出力する。
- OOEは `surf` を主指標、`hit` は補助ログとして常時出力する。

### DoD
- 全 run で `eval_v1.json` のみ参照し、実行時分岐で閾値が変化しない。
- `Adopt_core` と `Adopt_entry` が全ケースで保存される。

---

## 2) stress manifest 仕様

### ファイル
各ケース配下に `stress_manifest.json` を保存。

### 必須キー
```json
{
  "case_id": "string",
  "stress_type": "bottleneck|occlusion|clutter",
  "target_metric": "clr_min|C_vis_start|OOE_primary|R_reach",
  "seed": 0,
  "moved_objects": [],
  "added_clutter_objects": [],
  "before_metrics": {},
  "after_metrics": {},
  "delta_metrics": {}
}
```

### 生成ルール
- Bottleneck: 既存可動家具を移動し、`clr_min` 悪化を狙う。
- Occlusion: 入口視線の遮蔽を狙い、`C_vis_start` または `OOE_primary` を悪化させる。
- Clutter: `movable=false` の追加障害物を入れて悪化させる。

### DoD
- stress 15ケースすべてで `stress_manifest.json` が存在。
- `target_metric` と `delta_metrics` の符号が整合する（狙った指標が低下）。

---

## 3) proposed（Beam Search）仕様

### 実装ファイル
`experiments/src/refine_proposed_beam.py`

### 比較公平性
- action set は heuristic と同一（移動/回転近傍）。
- `max_changed_objects` は同一。
- 評価呼び出し上限（eval budget）を同一に揃える。

### 推奨パラメータ
`experiments/configs/refine/proposed_beam_v2.json`
```json
{
  "beam_width": 5,
  "depth": 3,
  "candidate_objects_per_state": 2,
  "eval_budget": 780,
  "step_m": 0.10,
  "rot_deg": 15.0,
  "max_changed_objects": 3,
  "lexicographic": true,
  "objective": {
    "use_entry_metrics": true,
    "ooe_primary_metric": "OOE_R_rec_entry_surf",
    "w_clr": 1.0,
    "w_reach": 1.0,
    "w_start": 1.0,
    "w_ooe": 1.0,
    "w_vis": 0.2,
    "w_delta": 0.3
  }
}
```

### スコア比較順（辞書式）
1. `validity == 1`
2. `Adopt_core == 1`
3. `Adopt_entry == 1`
4. 連続スコア `J`

### 連続スコア
- 閾値マージンで正規化した `clr_min`, `R_reach`, `C_vis_start`, `OOE_primary`, `C_vis` を加点。
- `Delta_layout(layout_out, degraded_input)` を減点。

### DoD
- `heuristic` と `proposed` で、同条件（同budget）比較が可能。
- stress 15ケースで、`improved_entry` が heuristic 以上。

---

## 4) 幾何整合の最小修正（先行実施）

### 実施内容
- 評価器 occupancy grid を plot の描画基準に合わせる。
- 壁境界 epsilon を全工程で共通値化。
- start/goal/path を eval と同じ座標系で描画。

### DoD
- 「見た目は壁貫通、評価は通行可」の矛盾が 0 件。

---

## 5) 実行計画（固定）
1. 20ケース再作成（5通常 + 15stress）
2. 4条件実行（Original / LLM-Refine / Heuristic / Proposed）
3. サマリー生成（all / by-stress）
4. 比較CSVと比較JSONを保存

### 出力
- `metrics_all_methods.csv`
- `compare_by_case.json`
- `plot_with_bg_summary_all_methods.png`
