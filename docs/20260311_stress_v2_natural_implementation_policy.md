# 2026-03-11 stress_v2_natural 実装方針

## 目的

本方針の目的は、stress 生成を次の 2 系統に分離し、主評価と診断評価の意味を明確化することです。

- 主評価:
  - `stress_v2_natural`
  - 自然に起こりそうな乱れに対する強さを測る
- 診断評価:
  - `stress_v1_targeted`
  - 特定の弱点を狙い撃ちした厳しい試験として残す

これにより、以下を同時に満たします。

- benchmark の意味を明確にする
- `Delta_layout` の解釈を壊さない
- refine 比較の公平性を保つ
- 後から stress 生成内容を再現・説明できる

## 結論

### 採用方針

- 主評価は `stress_v2_natural`
- 診断評価は `stress_v1_targeted`
- 評価器は `eval_v1.json` を凍結のまま使う
- 比較手法は当面現行凍結版を維持する
- manifest を benchmark 仕様の一部として扱う

### 主評価の variant

- `base`
- `usage_shift`
- `clutter`
- `compound`

### 診断評価の variant

- `targeted_bottleneck`
- `targeted_occlusion`

## 採用理由

### 1. 主評価では自然な乱れを使う

主評価で見たいのは、日常使用や現場運用のあとに自然に起こる乱れに対して配置改善手法がどれだけ強いかです。

そのため、評価指標を直接悪化させるように stress を作るのではなく、

- 家具の生活ずれ
- 後から入る追加障害物
- その同時発生

を主評価の対象にします。

### 2. 狙い撃ち stress は診断評価へ分離する

現在の `bottleneck` / `occlusion` は有用ですが、自然な乱れというより、弱点をあぶり出す試験です。

したがって、

- 主表の平均には入れない
- 付録または補助実験として残す

という位置づけにします。

### 3. 指標は生成のためではなく説明のために使う

stress 生成で評価指標を直接目的関数に使うと、評価に都合のよい人工ケースを作りやすくなります。

そのため `stress_v2_natural` では、

1. まず先験分布ベースで乱れを生成する
2. その後に `eval_v1` で悪化量を測定する
3. 難度ラベルを後付けする

という順序にします。

## 凍結して維持するもの

### 評価仕様

- `experiments/configs/eval/eval_v1.json`

これは frozen evaluator として維持し、今回の stress 再設計では変更しません。

特に以下は固定前提です。

- `task.start.in_offset_m = 0.50`
- `task.goal.offset_m = 0.60`
- `robot_radius_m = 0.28`
- `tau_Delta = 0.10`
- `adopt.entry_gate.metric = OOE_R_rec_entry_surf`
- `adopt.entry_gate.min_value = 0.70`

### 比較手法仕様

- `experiments/configs/refine/proposed_beam_v1.json`

比較手法も当面変更しません。まず stress 側だけを `v2` に分離し、手法差と問題差が混ざらないようにします。

## stress_v2_natural の定義

## `base`

- 無変更の基準レイアウト
- すべての比較の基準点

## `usage_shift`

### 目的

日常使用で起こりそうな家具のずれを再現する。

### 操作

- 既存可動家具のみ移動
- 追加障害物は使わない

### 初期対象家具

- 第1優先:
  - `chair`
  - `coffee_table`
  - `small_storage`
- 条件付き:
  - `table`
- 当初は対象外または低頻度:
  - `sofa`
  - `storage`
  - `tv_cabinet`
  - `cabinet`

### 動かす数

- 75%: 1個
- 25%: 2個
- 3個以上は使わない

### 乱数の考え方

家具ごとのローカル座標で前後・左右・回転をずらす。
世界座標基準ではなく、家具の向き基準でサンプリングする。

### pilot 用初期仮定分布

以下の数値は正式仕様ではなく、5ケースの目視確認を通したあとに固定するための初期仮定です。
論文本文では、この段階では canonical な分布値として断定しない。

- `chair`
  - 前後: 標準偏差 `0.12m`, 上限 `±0.30m`
  - 左右: 標準偏差 `0.05m`, 上限 `±0.12m`
  - 回転: 標準偏差 `7deg`, 上限 `±15deg`
- `coffee_table` / `small_storage`
  - 前後: 標準偏差 `0.08m`, 上限 `±0.20m`
  - 左右: 標準偏差 `0.08m`, 上限 `±0.20m`
  - 回転: 標準偏差 `5deg`, 上限 `±10deg`
- `table`
  - 前後: 標準偏差 `0.05m`, 上限 `±0.12m`
  - 左右: 標準偏差 `0.05m`, 上限 `±0.12m`
  - 回転: 標準偏差 `3deg`, 上限 `±6deg`

## `clutter`

### 目的

後から入る一時障害物を再現する。

### 操作

- 既存家具は動かさない
- 外部障害物を追加する

### 基本仕様

- `category = clutter`
- `movable = false`
- 追加数:
  - 80%: 1個
  - 20%: 2個

### サイズ

初期値は現行仕様を維持する。

- `0.35 x 0.35`
- `0.60 x 0.40`
- `height = 1.0`

### 配置方針

指標悪化を直接狙うのではなく、置かれやすい帯からサンプリングする。
帯の定義は幾何的に固定し、再現可能にする。

#### clutter placement bands

- `entry_staging_band`
  - 定義元: entrance 由来の `start` 点
  - 形状: `start` を中心とする局所矩形帯
  - 初期幅:
    - 接線方向 `1.0m`
    - 法線方向 `0.6m`
  - 初期抽選確率: `0.45`
- `path_staging_band`
  - 定義元: `start-goal` 線分
  - 形状: 線分まわりの帯領域
  - 初期幅:
    - 線分沿い区間 `t in [0.25, 0.75]`
    - 法線方向 `±0.45m`
  - 初期抽選確率: `0.35`
- `bedside_staging_band`
  - 定義元: bed の外周と goal 側領域
  - 形状: ベッド外周の部分矩形帯
  - 初期幅:
    - ベッド外周から `0.2m` 〜 `0.8m`
  - 初期抽選確率: `0.20`

#### clutter pose jitter

- 帯内部の位置:
  - 一様サンプリングを基本とする
- 向き:
  - `0deg / 90deg` を中心に、`±10deg` の jitter を与える
- 2個目の clutter:
  - 1個目と別 band を優先
  - ただし制約を満たさない場合は同じ band 内でも可

## `compound`

### 目的

生活ずれと追加障害の同時発生を扱う。

### 標準形

- 家具 1 個の `usage_shift`
- 追加障害 1 個の `clutter`

### 順序

1. `usage_shift`
2. `clutter`

### 注意

`compound` は変更量の解釈が混ざりやすいため、当面は変更量を分離して記録する。
集約量 `Delta_total` は現時点では未固定とし、必要になった段階で別途定義する。

- `Delta_movable`
- `num_added_clutter`
- `added_clutter_area`
- `compound = true/false`

## stress_v1_targeted の定義

## `targeted_bottleneck`

- 目的:
  - 通路余裕の弱点を診断する
- 操作:
  - 既存可動家具を意図的に導線近傍へ寄せる
- 用途:
  - 主評価ではなく補助実験

## `targeted_occlusion`

- 目的:
  - 入口観測の弱点を診断する
- 操作:
  - 既存可動家具を意図的に入口近傍へ寄せる
- 用途:
  - 主評価ではなく補助実験

## 共通制約

現行の部屋内制約を維持する。

- `same_room_only = true`
- 最大移動 `0.6m`
- 最大回転 `30deg`
- ドア keep-out `0.5m`
- 重なり率 `0.05` 以下
- `validity == 1` 必須

固定家具は動かさない。

- `bed`
- `toilet`
- `sink`
- `door`
- `window`
- `opening`
- `floor`

## 難度ラベルの付け方

難度は生成時に最適化せず、生成後に付ける。

### 方針

1. 乱数サンプルから複数候補を作る
2. 制約を満たす候補だけを残す
3. `eval_v1` で `metrics_before` / `metrics_after` を計算する
4. 指標差分から難度を決める
5. 最終セットは層別抽出する

### 想定ラベル

- `mild`
- `borderline`
- `hard_recoverable`

## manifest 方針

manifest は benchmark 仕様の一部として扱う。

### 各サンプル manifest の必須項目

- `stress_version`
- `stress_family`
- `base_case_id`
- `scene_id`
- `seed`
- `config_hash`
- `generator_commit`
- `selected_from_pool_index`
- `pool_size`
- `validity_checks`
- `rejection_summary`
- `moved_objects`
- `added_clutter`
- `metrics_before`
- `metrics_after`
- `metric_deltas`
- `difficulty_label`
- `selection_notes`

### moved_objects

各要素は最低限以下を持つ。

- `object_id`
- `category`
- `dx_local`
- `dy_local`
- `dtheta_deg`
- `same_room_ok`

### added_clutter

各要素は最低限以下を持つ。

- `clutter_id`
- `size_xy`
- `height`
- `pose`

### dataset QA report

R23 を解くため、各サンプル manifest と、データセット全体の QA 結果を分ける。

#### sample manifest

- 実際の生成内容を保存する
- 各 case / variant ごとに 1 ファイル

#### dataset QA report

- manifest 品質検査を保存する
- 単一サンプルの field ではなく、データセット単位の検査結果として持つ

最低限の検査状態:

- `exists`
  - 期待される manifest が存在するか
- `complete`
  - 必須項目が埋まっているか
- `explainable`
  - 人が読んで生成内容を追えるか

## コード構成案

最小変更で進めるため、入口は現行 generator を活かす。

- `experiments/src/generate_stress_cases.py`
  - `mode = natural_main | targeted_diag`
- `experiments/src/stress_priors.py`
  - 家具ずれ分布
  - clutter 配置分布
- `experiments/src/stress_pool.py`
  - 候補生成
  - 制約チェック
  - 難度ラベル付け
- `experiments/src/stress_manifest.py`
  - sample manifest 保存
  - 必須項目検証
- `experiments/src/stress_dataset_qa.py`
  - dataset QA report 生成
  - `exists / complete / explainable` の検査
- `experiments/configs/stress/stress_v2_natural.json`
- `experiments/configs/stress/stress_v1_targeted.json`

## 実装順

1. `stress_version` と `stress_family` を定義する
2. manifest schema を固定する
3. `usage_shift` を実装する
4. `clutter` を配置分布ベースに変更する
5. `compound` を追加する
6. `bottleneck / occlusion` を `targeted_diag` へ明示的に分離する
7. 5ケースで目視確認する
8. 全比較を再実行する

## 今回の運用ルール

- `eval_v1.json` は変更しない
- `proposed_beam_v1.json` は変更しない
- 今回更新するのは stress 定義と manifest 仕様
- stress 側を `v2` 化したら、比較は全条件で再実行する

## 期待する効果

- 主評価が「自然な乱れへの強さ」を測る benchmark になる
- targeted stress を捨てずに、診断評価として活かせる
- `Delta_layout` と外乱量の解釈が整理される
- 後から stress の生成過程を再現・説明できる

## 論文用の要約文

> 本研究では、主評価において評価指標を直接悪化させるような stress 生成は採らず、日常使用や現場運用で自然に生じうる乱れを確率的に生成する。評価指標はその乱れを作るためではなく、生成後の悪化量の測定と難度の層別化に用いる。家具移動と追加障害は原因の異なる乱れとして分離し、同時発生は別群として扱う。狙い撃ちの厳しい stress は補助的な診断評価として保持する。
