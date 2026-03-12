# 20260312 Layout Axis Alignment with Major-Gain Exception Policy

## 目的

家具 refine において、部屋の軸・壁方向に平行/直交する自然な配置を基本としつつ、**評価指標が十分に大きく改善する場合に限って** 少量の off-axis 回転を許す。

この方針の狙いは次の 2 つである。

- visibility / entry observability を稼ぐためだけの不自然な斜め配置を抑える
- 一方で、採択条件の回復や大幅な評価改善に本当に必要な場合だけは、少しの回転自由度を残す

## 位置づけ

- これは `layout_axis_alignment_prior` の拡張版である
- frozen main protocol の silent update ではない
- `eval_v1.json` は変更しない
- `proposed_beam_v1.json` も現時点では変更しない
- `heuristic` / `proposed` の furniture refine に対する **spec update candidate** として扱う

## 前提

本プロジェクトでは、元の家具配置はほぼ全て

- 部屋の主軸
- 壁方向

に対して平行/直交しているとみなす。

したがって、refine 中に現れる大きな斜め回転は、基本的に

- score exploit
- 生活上の一時的なずれ

のどちらかであり、原則として抑制すべきである。

## 基本原則

### 原則

家具は room axis に対して

- 平行
- 直交

する向きを基本とする。

### 例外

ただし、**採択条件の回復**または**十分に大きい評価改善**がある場合のみ、少量の off-axis 回転を許す。

## 適用対象

### 対象 furniture

この方針を適用するのは furniture phase のみとする。

- `bed`
- `sofa`
- `storage`
- `tv_cabinet`
- `sink`
- `cabinet`
- 長方形 `table`
- `coffee_table`
- `small_storage`
- `chair`

### 対象外

- `clutter`
- `door`
- `window`
- `opening`
- `toilet`
- 円形 `table`

## protocol ごとの適用範囲

### M1

適用する。

- `heuristic`
- `proposed`

の furniture refine に適用する。

### X1

適用しない。

- `clutter_assisted` は clutter のみを動かす extension であり、家具の回転 prior の対象外

### X2

- furniture phase には適用する
- clutter phase には適用しない

## room axis の定義

各 room の polygon edge から dominant wall direction を取る。

初版は次で十分である。

- room polygon の edge 角度を 90 度対称で畳み込む
- edge 長さ重み付きで dominant direction を決める
- その直交方向を second axis とする

## 候補 yaw の定義

### orthogonal candidate set

各家具について、基本候補は次の 4 方向とする。

- `theta_room + 0°`
- `theta_room + 90°`
- `theta_room + 180°`
- `theta_room + 270°`

### off-axis exception candidate set

例外候補は、最近傍 orthogonal yaw からの微小回転だけを許す。

初版の許容値:

- `max_off_axis_deg = 15°`
- 候補集合:
  - `nearest_orthogonal - 15°`
  - `nearest_orthogonal - 10°`
  - `nearest_orthogonal + 10°`
  - `nearest_orthogonal + 15°`

必要なら 10 度だけでもよいが、初版では 10/15 を両方持ってよい。

### 禁止する回転

- 30° 以上の off-axis 回転
- 45° 近傍の大きな斜め回転
- visibility 向上だけを目的とした大回転 exploit

## 評価関数への入れ方

### rotation penalty

off-axis 回転には penalty を課す。

概念上は次でよい。

- `rotation_penalty = lambda_rot_align * (deviation_deg / max_off_axis_deg)^2`

ここで

- `deviation_deg`
  - 最近傍 orthogonal yaw からの角度差
- `max_off_axis_deg = 15°`

とする。

### 使い方

- orthogonal candidate には penalty 0
- off-axis candidate には penalty > 0
- score 上はこの penalty を減点する

## off-axis を許す条件

off-axis candidate は、次のどれかを満たす場合のみ採択候補として残す。

### binary recovery 条件

- `Adopt_core: 0 -> 1`
- `Adopt_entry: 0 -> 1`

### major gain 条件

少なくとも 1 つを満たす。

- `Δclr_feasible >= 0.03`
- `ΔC_vis_start >= 0.05`
- `ΔOOE_R_rec_entry_surf >= 0.10`

### do-no-harm 条件

さらに必須条件として、次を守る。

- `validity` を落とさない
- 既に通っている `Adopt_core` を壊さない
- 既に通っている `Adopt_entry` を壊さない
- orthogonal candidate に対して改善が明確に大きい

## heuristic への実装方針

現状の `heuristic` では、家具 phase で `±rot_deg` 近傍を候補としている。

これを次に置き換える。

- 並進候補は現行のまま
- 回転候補は
  - orthogonal candidate set
  - 条件付き off-axis exception set
  のみ
- 候補スコアから `rotation_penalty` を減点する
- off-axis candidate は上記の recovery / major gain 条件を満たさない限り不採用とする

## proposed への実装方針

現状の `proposed` でも furniture yaw 候補を同様に置き換える。

- beam 内の furniture yaw 候補を orthogonal candidate set に制限
- off-axis candidate は条件付きで追加
- 連続スコアから `rotation_penalty` を減点
- binary recovery / major gain 条件を満たす場合のみ off-axis を残す

## 初期固定値

初版の推奨値は次とする。

- `max_off_axis_deg = 15°`
- `lambda_rot_align = 0.5`
- major gain thresholds
  - `Δclr_feasible >= 0.03`
  - `ΔC_vis_start >= 0.05`
  - `ΔOOE_R_rec_entry_surf >= 0.10`

## 期待する効果

- 家具が斜めに置かれる不自然さを大幅に減らせる
- visibility 指標 exploit を抑えられる
- 採択回復に本当に必要な small rotation だけを残せる
- `heuristic` と `proposed` の両方で見た目の plausibility を改善できる

## 既知のトレードオフ

- 連続値改善だけを見ると、改善量が少し落ちる可能性がある
- furniture rotate による visibility 稼ぎは抑えられる
- ただし layout plausibility は改善する

## 非対象

今回の方針では次はまだ入れない。

- furniture と clutter の同時最適化
- human preference score の学習
- aesthetic model の導入
- threshold-repair + do-no-harm の全面再設計

## 結論

家具 refine の自然さを改善するには、**room-axis orthogonal placement を基本**とし、**大幅な評価改善がある場合に限って小さな off-axis 回転を許す**方針が妥当である。

これは `heuristic` と `proposed` の両方に共通して適用でき、かつ現行の frozen main protocol とは切り分けて評価できる。
