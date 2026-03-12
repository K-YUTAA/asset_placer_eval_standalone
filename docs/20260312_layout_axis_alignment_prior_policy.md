# 20260312 Layout Axis Alignment Prior Policy

## 目的

現在の furniture refine では、特に `proposed` が `C_vis_start` や `OOE` を改善するために家具を斜め回転させることがあり、見た目の自然さと配置 plausibility を損ねている。

この方針では、**部屋の軸・壁方向に平行/直交する配置を基本とする prior** を furniture refine に導入する。対象は `heuristic` と `proposed` の両方とし、`clutter_assisted` には適用しない。

## 位置づけ

- 本方針は **main frozen protocol の直接上書きではない**。
- `eval_v1.json` は変更しない。
- `proposed_beam_v1.json` も現時点では変更しない。
- これは **spec update candidate** として扱う。
- もし採用する場合は、同条件で比較を再実行する。

## 何を防ぎたいか

現状の問題は次の通り。

- visibility / entry observability を稼ぐために家具が斜めを向く
- score は改善しても、見た目が明らかに不自然になる
- 「人が見て納得するレイアウト」でなくなる

したがって、refine には geometrically valid であるだけでなく、**layout plausibility** の prior が必要である。

## 基本方針

### 1. 家具配置の基準軸

各 room について、壁方向から得られる **room axis** を求める。

- 主軸: 最も支配的な壁方向
- 従軸: 主軸に直交する方向

家具の向きは、この room axis に対して

- 平行
- 直交

のいずれかを基本とする。

### 2. hard rule を優先する

初版では soft penalty よりも **hard constraint** を優先する。

対象 furniture は、候補 yaw を次に制限する。

- `0°`
- `90°`
- `180°`
- `270°`

ただし、これは world axis ではなく **room axis 基準** の 4 方向である。

### 2.5. 元レイアウトのズレは原則として補正対象とみなす

本プロジェクトの前提として、元の家具配置はほぼ全て

- 部屋の主軸
- 壁方向

に対して平行/直交している。

したがって、refine 中に room axis から外れた向きが現れた場合は、

- visibility を稼ぐための不自然な exploit
- 生活上の一時的なずれ

のいずれかであるとみなしてよい。

この前提に基づき、**元 yaw を守る soft prior は採らない**。
代わりに、家具の自然な向きは **room-axis orthogonal set** によって定義する。

つまり、元レイアウトが room axis から外れている場合でも、それは保持対象ではなく、**補正してよいズレ**として扱う。

### 3. 適用対象

#### orthogonal-only 対象

以下は room axis に対する平行/直交のみ許可する。

- `bed`
- `sofa`
- `storage`
- `tv_cabinet`
- `sink`
- `cabinet`
- 長方形 `table`
- `coffee_table`
- `small_storage`

#### 例外対象

以下は hard orthogonal-only から外すか、別ルールとする。

- `chair`
  - anchor furniture を向くことが自然なので、別ルール候補
- 円形 `table`
  - 回転の意味が薄い
- `door`, `window`, `opening`
  - structure 側で決まるため対象外
- `toilet`
  - 機能方向と room axis が必ずしも一致しないため対象外

## protocol ごとの適用範囲

### M1: furniture refine

適用する。

- `heuristic`
- `proposed`

の両方で furniture candidate の yaw 候補を room-axis orthogonal set に制限する。

### X1: clutter-assisted recovery

適用しない。

- `clutter` は furniture ではない
- もともと平行移動のみ
- 元位置保持より指標改善を優先する extension なので、今回の prior の対象外

### X2: furniture -> clutter

- furniture phase (`M1`) には適用する
- clutter phase (`X1`) には適用しない

## 実装方針

### Step 1. room axis を取得する

候補:

- room polygon の edge 方向を集計する
- 最長辺群から支配方向を取る
- その直交方向を second axis とする

初版は最も単純な方法でよい。

- room polygon edge の角度を 90 度対称で畳み込む
- 長さ重み付きヒストグラムで dominant axis を選ぶ

### Step 2. category ごとに yaw 候補集合を切る

`layout_only` の furniture refine では、現在の `±rot_deg` 近傍ではなく、次を使う。

- room axis + 0°
- room axis + 90°
- room axis + 180°
- room axis + 270°

必要なら current yaw に最も近い 2 候補だけを使ってもよいが、初版は 4 候補全列挙でよい。

### Step 2.5. rotation cost は room-axis deviation のみで定義する

候補 yaw の評価では、元 yaw からのずれではなく、**room axis からのずれ**だけを見る。

- `alignment_cost`
  - room axis に対する最近傍 orthogonal yaw からのずれ

concept 上は次で十分である。

- `rotation_cost = w_align * alignment_cost`

ただし、初版ではさらに単純化し、**rotation 候補自体を room-axis orthogonal set に限定**する。

そのため、実際の score では `rotation_cost` は tie-breaker としてだけ使えばよい。

狙いは次の通り。

- visibility 指標を稼ぐためだけの斜め exploit を止める
- 元レイアウトに入った人為的なずれを補正する
- 家具配置を壁方向に平行/直交した自然な見え方へ戻す

### Step 3. heuristic への反映

現状の `heuristic` は local neighborhood の中で `±rot_deg` を候補にしている。

これを furniture phase では置き換える。

- translation 候補はそのまま
- rotation 候補だけ room-axis orthogonal set に変更
- 候補評価時に `rotation_cost` を減点として入れる
- ただし `rotation_cost` は room-axis deviation のみを見る

### Step 4. proposed への反映

現状の `proposed` も rotation を連続近傍として扱っている。

家具 phase では次に変更する。

- furniture candidate の yaw を room-axis orthogonal set に量子化
- beam 内ではその離散候補だけを試す
- beam score の連続部分に `rotation_cost` を penalty として入れる

これにより、`proposed` が visibility を稼ぐためだけに家具を回す挙動を抑制する。

### Step 5. clutter protocol は変えない

`clutter_assisted` は今回の prior の対象外。

- clutter は furniture ではない
- 現状どおり translation のみ
- score は clutter recovery 用のものを維持

## logging / 比較

この方針を採用する場合、比較では少なくとも次を分ける。

- 旧 furniture refine
- room-axis prior 付き furniture refine

見るべきもの:

- `Adopt_core`
- `Adopt_entry`
- `Δclr_feasible`
- `ΔC_vis_start`
- `ΔOOE_R_rec_entry_surf`
- `Δlayout_furniture`
- 可視化上の見た目

## 期待する効果

- 不自然な斜め配置が減る
- 元レイアウトに入った人為的なずれを補正しやすくなる
- 人間が見て納得しやすい layout になる
- visibility 指標だけを稼ぐ回転 exploit を抑制できる

## 既知のトレードオフ

- score の上限は下がる可能性がある
- 連続値改善だけを見ると悪化する場合がある
- ただし、layout plausibility は改善する

## 非対象

今回の方針では次はまだ入れない。

- threshold-repair + do-no-harm の再設計
- furniture と clutter の同時最適化
- aesthetic score の追加設計
- human preference model の導入

## 結論

次の段階として、`heuristic` と `proposed` の furniture refine に対し、**room-axis orthogonal prior** を導入するのは妥当である。

ただしこれは frozen main protocol の silent update ではなく、**spec update candidate** として扱い、採用時には再比較を前提にする。
