# Clutter Recovery Complete Specification

## 目的

`clutter` を使う評価と改善の仕様を、`main` の frozen comparison を壊さずに一枚で読める形で固定する。

この文書の目的は次の 3 つです。

1. `clutter` の意味を benchmark 上で明確にする
2. `main` と `extension` の protocol を分離する
3. `M0 / M1 / X1 / X2` の比較条件、実装範囲、ログ項目を明示する

## この仕様での `clutter` の定義

この仕様での `clutter` は、`stress` 生成で後から追加される外部障害物を指す。

- 既存家具ではない
- `stress_added` の object である
- 主に `stress_v2_natural` の `clutter` / `compound` variant で現れる
- benchmark 上は「後から置かれた一時障害物」として扱う

したがって、`furniture` と `clutter` は同じ「物体」でも意味が違う。

- `furniture`
  - 元レイアウトの一部
  - `Delta_layout` を小さく保つ意味がある
- `clutter`
  - 後付けの障害物
  - 元位置保持の意味は薄い
  - 通行・入口観測を改善する位置への再配置を優先してよい

## 凍結されているもの

この仕様では、以下は frozen artifact として維持する。

### 評価仕様

- `experiments/configs/eval/eval_v1.json`

### main の提案手法仕様

- `experiments/configs/refine/proposed_beam_v1.json`

### latest-design の生成設定

- `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`

この文書で変更対象にするのは `extension` 側だけであり、`main` の評価仕様と furniture refine の正本は変えない。

## Protocol 全体像

### `M0`: no refine

- stress 入力をそのまま評価する
- baseline として使う
- 物体は一切動かさない

### `M1`: furniture refine

- 現行 `main`
- movable furniture のみ可動
- stress で追加された `clutter` は固定
- 比較条件:
  - `Original`
  - `Heuristic`
  - `Proposed`

### `X1`: clutter refine

- extension
- stress で追加された `clutter` のみ可動
- furniture は固定
- `clutter` の元位置維持より、評価指標改善を優先する

### `X2`: furniture -> clutter refine

- まず `M1`
- その出力に対して `X1`
- 家具改善と `clutter` 再配置の効果を分離して見られるようにする

## main と extension の境界

### main

- `layout_only`
- 既存家具だけを動かす
- frozen comparison として扱う
- 論文本文の主比較はこちら

### extension

- `clutter_assisted`
- stress で追加された `clutter` だけを動かす
- 論文では appendix / extension として扱う

この分離を保つ理由は次の通り。

- 過去の main 比較結果を壊さない
- `clutter` 可動化の効果を単独で測れる
- 後で `threshold-repair + do-no-harm` を追加しても、何の効果で改善したかを分離できる

## Object metadata

各 object には、少なくとも次の metadata を持たせる。

- `origin`
  - `original`
  - `stress_added`
- `stress_kind`
  - `base`
  - `usage_shift`
  - `clutter`
  - `compound`
  - `targeted_bottleneck`
  - `targeted_occlusion`
- `object_role`
  - `furniture`
  - `clutter`
  - `diagnostic_obstacle`
- `movable_in_main`
- `movable_in_clutter_recovery`
- `source_case_id`
- `stress_case_id`

この metadata の役割は、各 protocol で「何を動かしてよいか」を case-wise に追跡可能にすることにある。

## stress taxonomy との関係

### 主評価

`stress_v2_natural`

- `base`
- `usage_shift`
- `clutter`
- `compound`

### 診断評価

`stress_v1_targeted`

- `targeted_bottleneck`
- `targeted_occlusion`

`clutter recovery` の主対象は `stress_v2_natural` のうち、

- `clutter`
- `compound`

である。

`base` と `usage_shift` は `clutter` を持たないので、`X1` では基本的に no-op になる。

## v1 の実装範囲

今回の `clutter_assisted` 実装では、次だけを許す。

- `clutter` の平行移動
- 回転なし
- 削除なし
- 室外退避なし
- furniture と `clutter` の同時可動なし

今回まだ入れないもの:

- `threshold-repair + do-no-harm`
- furniture と `clutter` の同時最適化
- `X2` 専用の一括 runner 最適化
- `clutter` の削除・回転・室外移動

## `X1` の目的関数

`X1` では、`clutter` の元位置保持は主目的にしない。

優先順位は次の通り。

1. `validity == 1`
2. `Adopt_entry == 1`
3. `Adopt_core == 1`
4. `clr_feasible` 最大化
5. `C_vis_start` 最大化
6. `OOE_R_rec_entry_surf` 最大化
7. `R_reach` 最大化
8. `C_vis` 最大化
9. `delta_layout_clutter` 最小化

ここでのポイントは、`delta_layout_clutter` は tie-breaker に落としていること。

理由:

- `clutter` は元レイアウトの一部ではない
- 「少しだけ動かす」ことより「通れる・見える」を優先する方が自然

## `X1` の candidate generation

`clutter` は近傍 26 候補だけでは弱いので、room 全体の free space を候補に含める。

### 候補制約

- same room
- wall crossing なし
- door keep-out を守る
- object overlap 上限を守る
- evaluator と同じ occupancy 真値で `validity == 1`

### 候補位置

- room bounding box 上の格子点を候補にする
- `yaw` は v1 では固定
- 現在位置近傍だけでなく room 全体を見る
- 候補数には上限を設ける

### quick ranking

候補評価前に、次のような quick score で上位候補だけに絞る。

- `start-goal` 線分から遠い
- `start` から遠い
- bottleneck から遠い
- room の端に寄っている

## Method ごとの挙動

### `heuristic-C`

- `clutter` のみ対象
- 各反復で `clutter x candidate_position` を全探索
- 最良の 1 手だけ採用
- 反復回数は小さく保つ

### `proposed-C`

- `clutter` のみ対象
- beam search で複数候補を保持
- ただし scoring は `main` の `proposed` とは分離
- `clutter` 再配置専用の objective を使う

## `X2` の扱い

`X2` は 1 本の新しい optimizer ではなく、明示的な chained execution として扱う。

1. `M1` で furniture refine
2. その `layout_refined.json` を入力に `X1`

この扱いにする理由は、家具改善と `clutter` 改善の効果を混ぜないためである。

## Logging と変更量の定義

`clutter_assisted` では、変更量を 1 本に潰さず、少なくとも次を分けて保存する。

- `delta_layout_furniture`
- `delta_layout_clutter`
- `moved_furniture_count`
- `moved_clutter_count`
- `sum_furniture_displacement_m`
- `sum_clutter_displacement_m`
- `max_clutter_displacement_m`

これにより、

- 家具をどれだけ直したのか
- `clutter` をどれだけどかしたのか

を分けて解釈できる。

## 実装ファイル

### main / frozen side

- `experiments/configs/eval/eval_v1.json`
- `experiments/configs/refine/proposed_beam_v1.json`

### extension side

- `experiments/configs/refine/clutter_assisted_v1.json`
- `experiments/src/refine_clutter_recovery.py`

### dispatch / execution

- `experiments/src/run_trial.py`

### stress metadata source

- `experiments/src/generate_stress_cases.py`

## 実装状態

現時点で入っているもの:

- `clutter_assisted` protocol
- `clutter` 専用 scoring
- room-wide candidate generation
- `run_trial.py` の protocol dispatch
- furniture / `clutter` 分離 logging

現時点でまだ main に入れていないもの:

- `clutter` 可動化を frozen comparison に混ぜること
- `threshold-repair + do-no-harm`
- hybrid simultaneous optimization

## 評価の見方

最小限、次の比較を出せば意味が通る。

### `M0 vs M1`

- 家具 refine の効果

### `M0 vs X1`

- `clutter` cleanup 単独の効果

### `M1 vs X2`

- 家具 refine 後に `clutter` cleanup を足す価値

主に見る指標:

- `Adopt_core`
- `Adopt_entry`
- `clr_feasible`
- `C_vis_start`
- `OOE_R_rec_entry_surf`
- `delta_layout_furniture`
- `delta_layout_clutter`

`X1 / X2` では、とくに `clutter / compound` を主解析対象とする。

## 論文上の位置づけ

### Main text

- `layout_only`
- `Original / Heuristic / Proposed`
- frozen comparison

### Appendix / Extension

- `clutter_assisted`
- movable-obstacle assumption の下での追加回復可能性

この整理により、main の claim を壊さずに extension を追加できる。

## 非対象

この仕様では、次は対象外とする。

- `clutter` の削除
- `clutter` の回転
- 室外退避
- furniture と `clutter` の同時最適化
- `threshold-repair + do-no-harm` の本格導入

## まとめ

この仕様のポイントは次の 4 点である。

1. `main` は frozen protocol のまま維持する
2. `clutter` は外部障害物として別扱いする
3. `X1 / X2` では `clutter` の元位置保持より指標改善を優先する
4. 変更量と物体種別を分離して記録する

これにより、`clutter` 可動化の効果を main 比較と混同せずに評価できる。
