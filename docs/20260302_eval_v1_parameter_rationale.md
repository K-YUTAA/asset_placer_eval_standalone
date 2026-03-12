# eval_v1 Parameter Rationale (2026-03-02)

## Scope

本メモは `experiments/configs/eval/eval_v1.json` の値を固定した理由を整理する。

前提:

- 対象は中型の屋内配膳AMR（Servi/BellaBot級）。
- 評価タスクは現行どおり `entrance -> bedside`。
- 実装互換性のため、現行コードが読むキー形式（`task.start.mode`, `entry_observability`, `tau_*` など）で定義する。

## Final Values

```json
{
  "grid_resolution_m": 0.05,
  "robot_radius_m": 0.28,
  "occupancy_exclude_categories": ["floor"],
  "task": {
    "start": {
      "mode": "entrance_slidingdoor_center",
      "in_offset_m": 0.50,
      "door_selector": { "strategy": "largest_opening" }
    },
    "goal": {
      "mode": "bedside",
      "offset_m": 0.60,
      "bed_selector": { "strategy": "first" },
      "choose": "closest_to_room_centroid"
    },
    "snap": { "max_radius_cells": 30 }
  },
  "start_xy": [0.8, 0.8],
  "goal_xy": [5.0, 5.0],
  "sample_step_m": 0.50,
  "max_sensor_samples": 10,
  "tau_R": 0.90,
  "tau_clr": 0.10,
  "tau_V": 0.60,
  "tau_Delta": 0.10,
  "lambda_rot": 0.50,
  "entry_observability": {
    "enabled": true,
    "mode": "both",
    "exclude_categories": ["floor", "door", "window"],
    "target_categories": [],
    "height_by_category_m": {},
    "sensor_height_m": 0.60,
    "num_rays": 720,
    "max_range_m": 10.0,
    "tau_p": 0.05,
    "tau_v": 0.30
  },
  "adopt": {
    "report_both": true,
    "entry_gate": {
      "enabled": true,
      "metric": "OOE_R_rec_entry_surf",
      "min_value": 0.70
    }
  }
}
```

## Rationale by Parameter

`grid_resolution_m = 0.05`

- Nav2 costmapで一般的な解像度（5 cm）に合わせ、屋内評価で過粗視化を避ける。
- 5 cmは経路・クリアランス評価の安定性と計算量のバランスが良い。

`robot_radius_m = 0.28`

- Servi/BellaBot級の半幅レンジ（概ね 0.24-0.285 m）に合わせたやや保守的代表値。
- evaluator が円形近似前提なので、半幅ベースで固定するのが自然。

`task.start.mode = entrance_slidingdoor_center`

- 入口起点を幾何的に一意に取りやすく、ケース間再現性が高い。

`task.start.in_offset_m = 0.50`

- 入口直後を表現しつつ、ドアしきい/境界スナップ誤差の影響を減らす。
- 0.40よりやや内側に置いて開始点の不安定さを抑える。

`task.goal.mode = bedside`

- 現行研究タスクを維持し、比較軸を変えないため。

`task.goal.offset_m = 0.60`

- ベッド縁に過接近しない実用距離として設定。
- 近すぎると局所的な衝突判定と視認性の不安定を誘発しやすい。

`task.goal.choose = closest_to_room_centroid`

- 人姿勢推定を持たない前提で、より開けた側を選ぶ簡易proxyとして妥当。

`task.snap.max_radius_cells = 30`

- `snap_to_free` 相当を現行実装で担うキー。
- 5 cm解像度では最大1.5 m探索になり、開始点/目標点が占有に落ちた際の救済として十分。

`tau_R = 0.90`

- ほぼ全域の到達性を要求しつつ、自由セル定義の細部ノイズで過剰不合格になりにくいライン。
- 外部標準値ではなく、ベンチマーク運用ポリシー。

`tau_clr = 0.10`

- `robot_radius=0.28` では必要通路幅は約 `2 * (0.28 + 0.10) = 0.76 m`。
- 配膳AMRの公称最小通路幅（0.55-0.70 m）より少し保守的で、`0.25` より現実に近い。

`tau_V = 0.60`

- `C_vis` は全自由セルに対する可視率で、室内家具配置に敏感。
- 0.60に下げ、OOE側（入口可観測性）と合わせて判定する設計にする。

`tau_Delta = 0.10`

- refine を局所調整中心に制約し、大規模再配置を抑える運用値。
- 外部規格由来ではなく、比較実験の一貫性確保のためのポリシー値。

`entry_observability.mode = both`

- `surf` を主、`hit` を補助ログとして同時保存するため。
- 現行実装では `primary_mode`/`secondary_mode` ではなく `mode=both` を使う。

`entry_observability.tau_v = 0.30`

- 可視面積比30%を認識閾値とする既存設定を踏襲。
- 入口視認性の主判定は `surf` 系で運用する。

`entry_observability.tau_p = 0.05`

- `first-hit` は補助メトリクス用途のため、低め閾値でログ感度を確保。
- 採否の主判定には使わず、過剰な不合格要因にしない。

`adopt.entry_gate.enabled = true`

- Core採否に加えて入口可観測性を採否に反映するため有効化。

`adopt.entry_gate.metric = OOE_R_rec_entry_surf`

- 入口可観測の主指標を `surf` 側に統一。

`adopt.entry_gate.min_value = 0.70`

- 「入口から主要物体の大半が認識可能」を要求する実務的ライン。
- 病院寄りで厳格化する場合は 0.80 を検討余地あり。

`adopt.report_both = true`

- `Adopt_core` と `Adopt_entry` を同時に記録し、分析時に判定分解できるようにする。

`occupancy_exclude_categories = ["floor"]`

- デフォルト値と同じだが、設定ファイル側に明示して環境差異を減らす。

## Notes

- `validity` は設定値ではなく、start/goal が自由セルかどうかの計算結果。
- 今回の `eval_v1` は「外部仕様で強く決まる値」と「ベンチマークポリシー値」を混在させている。
- 比較実験開始後は、ポリシー値（`tau_R`, `tau_V`, `tau_Delta`, `entry_gate.min_value`, `tau_p`）を途中変更しない。

## References

- Servi spec (SoftBank Robotics): https://www.softbankrobotics.com/jp/product/delivery/servi/spec/
- BellaBot Pro / corridor width (Pudu): https://www.pudurobotics.com/en/products/bellabotpro
- Nav2 costmap configuration: https://docs.nav2.org/configuration/packages/configuring-costmaps.html
- ADA entrances/doors/gates: https://www.access-board.gov/ada/guides/chapter-4-entrances-doors-and-gates/
- ADA accessible routes: https://www.access-board.gov/ada/guides/chapter-4-accessible-routes/
- PUDU HolaBot: https://www.pudurobotics.com/product/detail/holabot
- ROMAN 2016 paper link provided by user: https://www.aisl.cs.tut.ac.jp/robocup/pdf/ROMAN2016.pdf
- CVPR 2010 paper link provided by user: https://vision.ist.i.kyoto-u.ac.jp/pubs/PBariya_CVPR10.pdf
- Optica JOSAA link provided by user: https://opg.optica.org/abstract.cfm?uri=josaa-29-9-1794
- Bear Robotics Servi: https://www.bearrobotics.ai/servi
