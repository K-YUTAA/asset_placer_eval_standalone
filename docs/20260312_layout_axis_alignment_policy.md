# Furniture Axis-Alignment Prior Policy

## 目的

`heuristic` と `proposed` の furniture refine に対して、**部屋の軸・壁方向に平行 / 直交する配置を基本とする prior** を導入する。

狙いは次の 3 つ。

1. `C_vis_start` や `OOE` を稼ぐための不自然な斜め回転を抑える
2. 人が見て違和感の少ないレイアウトを維持する
3. 回転による見かけ上の指標改善と、配置としての自然さを分離して扱う

## 背景

現状の `proposed` は、入口観測や可視性を改善するために家具を回転させることがあり、その結果として次が起きている。

- 指標は改善する
- ただし家具が中途半端な角度で配置される
- レイアウトとしての自然さが崩れる

同じ問題は程度の差はあっても `heuristic` 側にも起こりうる。

したがって、**`heuristic` と `proposed` の両方に同じ配置 prior を入れる**必要がある。

## この方針の位置づけ

これは `main` の frozen comparison を直接上書きするものではなく、**次期 spec update candidate** として扱う。

凍結されている正本:

- `experiments/configs/eval/eval_v1.json`
- `experiments/configs/refine/proposed_beam_v1.json`

この方針で新たに切る対象:

- `heuristic_v2_layout_aligned`
- `proposed_beam_v2_layout_aligned`

つまり、今回は **実装方針の固定** が目的であり、既存の `v1` 比較結果は残す。

## 基本原則

### 原則 1: 家具は部屋軸に整列する

主要家具は、部屋の主軸に対して次の角度だけを基本候補にする。

- `0`
- `90`
- `180`
- `270`

ここでの `0/90/180/270` は world 固定ではなく、**room axis 基準**で定義する。

### 原則 2: 斜め回転は原則禁止する

主要家具については、`±15°` や `±30°` のような自由回転をやめる。

最初の実装では、**hard rule として候補角を制限**する。

### 原則 3: 例外は category 単位で扱う

すべての家具を同じ回転ルールで扱うと逆に不自然になるので、category ごとに扱いを分ける。

## Room axis の定義

room axis は次の順で決める。

1. `main_inner_frame` または room polygon の長辺方向
2. その直交方向

今回の floorplan はほぼ軸平行なので、初版では **room polygon の辺方向から最頻の 2 軸** を取れば十分。

初版では簡単化して、実装上は次でよい。

- room polygon の辺ベクトルを集める
- 角度を `0/90` 系に量子化する
- 最も支配的な軸を `room_axis_0`
- その直交を `room_axis_90`

## 家具カテゴリごとの扱い

### A. orthogonal_only

対象:

- `bed`
- `sofa`
- `storage`
- `cabinet`
- `tv_cabinet`
- `sink`
- `toilet`
- 長方形 `table`

ルール:

- 候補角は `room_axis_0 + {0, 90, 180, 270}` のみ
- 中途半端な角度は許可しない

### B. orthogonal_with_anchor_choice

対象:

- `chair`
- 必要なら一部の `table`

ルール:

- 候補角自体は `room_axis_0 + {0, 90, 180, 270}` のみ
- その中から
  - `front_hint`
  - テーブル / ソファ / ベッドなどの anchor
  - room centroid
  を使って向きを選ぶ

つまり、**自由回転ではなく、4方向の中から機能的に最も自然な向きを選ぶ**。

### C. free_rotation

対象:

- `clutter`
- 円形 `table`
- 一部の特殊家具

今回の furniture refine では主対象外。

`clutter` は別 protocol なので、この prior は適用しない。

## 実装方針

### `heuristic`

現状:

- `step_m`
- `rot_deg`
- `(-rot, 0, +rot)` の局所回転候補

変更:

- `orthogonal_only` / `orthogonal_with_anchor_choice` に対しては
  - `rot_deg` の近傍回転候補を廃止
  - 候補角を room-axis 基準の 4 候補に置き換える

実装上は、

- 各 candidate layout 生成時に
  - `current_yaw + dtheta`
  ではなく
  - `canonical_yaw_candidates(category, room_axis)`
  を使う

とする。

### `proposed`

現状:

- `step_m`
- `rot_deg`
- beam search で局所回転候補も探索

変更:

- `heuristic` と同じ canonical yaw 候補に揃える
- つまり `proposed` だけ特別に自由回転を許さない

これにより、

- action set の公平性
- 形状 prior の一貫性

を保つ。

## scoring への入れ方

初版では **hard constraint を主** にする。

つまり、

- そもそも斜め角度を候補生成しない

を基本にする。

### 補助的な soft penalty

必要なら補助として、最近傍の canonical yaw からのズレに軽い penalty を入れる。

ただし初版では次の理由で不要。

- hard rule だけでかなり不自然さを抑えられる
- penalty を入れると調整パラメータが増える
- 今は spec を複雑にしない方がよい

## `clutter` との関係

この prior は **furniture refine 専用** とする。

適用対象:

- `M1`
- `X2` の前段 furniture refine

適用しない対象:

- `X1` の clutter refine
- `X2` の後段 clutter refine

理由:

- `clutter` は外部障害物であり、元位置保持や整列性より回復を優先する
- `clutter` にまで同じ整列 rule をかけると、extension の意味が崩れる

## 期待される変化

この prior を入れると、次の変化を期待する。

### 良くなるもの

- レイアウトの自然さ
- 人間が見たときの納得感
- 家具の配置の一貫性

### 落ちる可能性があるもの

- `C_vis_start`
- `OOE`
- 一部ケースでの連続値改善量

ただし、それは **不自然な回転で稼いでいた分が消える** という意味なので、むしろ解釈しやすくなる。

## 比較の置き方

この prior を入れた版は、最初は次の位置づけにする。

### main

- 現行 frozen comparison

### appendix / spec update candidate

- axis-aligned prior 付き `heuristic`
- axis-aligned prior 付き `proposed`

理由:

- 現行正本を壊さない
- prior の効果を単独で見られる
- 後で `threshold-repair + do-no-harm` を足しても切り分けられる

## 実装順

1. room axis 推定 helper を追加
2. category ごとの rotation policy table を追加
3. `heuristic` の rotation candidate を canonical yaw に置換
4. `proposed` の rotation candidate を canonical yaw に置換
5. `X1` には適用しないことを明示
6. 数ケースで見た目を確認
7. 必要なら `soft penalty` を追加検討

## 今回の採用判断

### 断定

- `heuristic` と `proposed` の両方に同じ整列 prior を入れるべき
- 主要家具は room axis 基準の平行 / 直交配置を基本にすべき
- 初版は hard rule のみでよい

### 保留

- `soft penalty` の追加
- `chair` の細かい例外処理
- category ごとの anchor 選択の複雑化

### 非採用

- 主要家具に自由回転を残したまま penalty だけで抑える案
- `clutter` に同じ整列 prior をかける案

## 一文で言うと

> furniture refine では、評価指標改善のための自由回転を許さず、部屋軸・壁方向に平行 / 直交する canonical orientation のみを候補とすることで、可視性改善と配置自然性のバランスを取る。
