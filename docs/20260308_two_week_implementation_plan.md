# 20260308 Two-Week Implementation Plan

## 目的

提出までの2週間では、新規要素を増やすのではなく、既存の生成・評価・stress・refine の流れを論文主張として成立する形に仕上げる。

論文の芯は次で固定する。

> 実在介護居室レイアウトをもとに、入口直後の状況把握を含む導入可否指標を設計し、乱れた環境に対して最小変更で導入可能側へ修正する評価・改善ループを提案する。

## 現状整理

現状の強みは以下。

- 実在レイアウト 5 件 + stress 15 件 = 20 ケースがある
- `R_reach`, `clr_feasible`, `C_vis`, `C_vis_start`, `OOE`, `Adopt_core`, `Adopt_entry` が回っている
- `heuristic` と `proposed` の両方が実装済み
- summary 図と case-wise 可視化が生成できる

一方で、現状の主要課題は以下。

- `proposed` が平均連続値改善には効くが、`Adopt_core` / `Adopt_entry` の回復率で `heuristic` に勝てていない
- geometry / occupancy / plot の不整合が残ると評価の信用性が下がる
- `eval_v1` の凍結実装は完了したため、以後は参照徹底と変更禁止を運用で守る
- stress の失敗強度が弱いケースがあり、手法差が見えづらい

## 今回の方針

### やること

1. `proposed` を threshold-repair 型に改修する
2. geometry / occupancy / plot の整合性を完全に揃える
3. `eval_v1` を frozen spec として維持し、今後の比較条件をこれに統一する
4. stress generator を少しだけ強くして、失敗ケース数を増やす
5. `Original / Heuristic / Proposed` の 3 条件で主結果を固める
6. 余力があれば `LLM-Refine` を最後に追加する

### やらないこと

- 人の動的シミュレーション追加
- 新しい評価指標の大量追加
- generator の大改修
- 新しい task の追加
- 施設全体や浴室・厨房へのスコープ拡張

## 実装優先度

### Priority A: `proposed` を修復器に変える

現状の `proposed` は連続値改善寄りで、閾値超えの回復器として弱い。これを次の 2 段階方式に変更する。

#### Stage 1: Heuristic Repair

- まず `heuristic` で `validity` と `Adopt_core` を回復する
- 可能なら `Adopt_entry` まで回復する
- この段階では「通る状態に戻す」ことを優先する

#### Stage 2: Beam Enhancement

- Stage 1 出力を初期解として beam search を実行する
- 改善対象は以下
  - `C_vis_start`
  - `OOE_primary`
  - `Delta_repair`
- ただし一度回復した条件は落とさない

#### 実装方針

- `proposed` の採択条件を、単純な平均改善ではなく threshold margin 優先にする
- `Adopt_core == 1` / `Adopt_entry == 1` を最上位条件に置く
- 既に満たしている条件を悪化させる候補は破棄する
- `heuristic` の結果を warm start として beam の探索空間を狭める

#### DoD

- `proposed` が `heuristic` と同等以上の `Adopt_core` 回復率を出す
- 少なくとも一部ケースで `Adopt_entry` 回復率または `C_vis_start` の明確な優位が出る

### Priority B: geometry / occupancy / plot の単一化

評価と可視化の不整合は論文の根幹を崩すため、次を統一する。

- occupancy grid を唯一の真値とする
- path 描画、壁描画、境界 clipping、inflate の規則を occupancy 由来に揃える
- `start` / `goal` / `path` の可視化を評価結果と一致させる

#### DoD

- 20 ケースすべてで見た目と評価結果が矛盾しない
- path が壁貫通しない
- summary 図でも個別図でも同じ挙動になる

### Priority C: `eval_v1` frozen spec の維持

今後の比較は `eval_v1` を唯一の評価条件として運用する。

#### 凍結済み対象

- grid resolution
- robot radius
- start / goal 決定ルール
- `Adopt_core` 条件
- `Adopt_entry` 条件
- OOE の primary 指標
- 各 tau 値

#### 方針

- OOE の primary は `surf` を採用する
- `Adopt_core` と `Adopt_entry` は両方残す
- 以後の実験では `eval_v1.json` を変更しない
- 実行 manifest から `eval_v1.json` の path / hash を必ず追跡できる状態を維持する

#### DoD

- 実験導線がすべて `eval_v1.json` を既定参照している
- `strict_eval_spec=true` で hidden default なしに動作する
- レポート側でも同じ定義で統一されている

### Priority D: stress generator の強度調整

20 ケース構成は維持しつつ、失敗ケースを少しだけ強くする。

#### 調整方針

- `bottleneck`
  - `clr_feasible` が閾値 `0.10` を少し下回るケースを増やす
- `occlusion`
  - `OOE_R_rec_entry_surf` が `0.70` を少し下回るケースを増やす
  - 経路性は維持する
- `clutter`
  - `start` 近傍またはベッド・トイレ導線近傍に寄せて置く

#### DoD

- `Adopt_core` failure が 5〜6 件程度ある
- `Adopt_entry` only failure が 2〜3 件程度ある

## 実験条件

主比較条件は次の 3 条件で固定する。

1. `Original`
2. `Heuristic`
3. `Proposed`

`LLM-Refine` は比較対象として魅力はあるが、優先順位は低い。以下を満たせる場合のみ追加する。

- object 追加・削除禁止
- movable 家具のみ変更可
- 最大変更数を他手法と一致
- prompt / response をキャッシュ
- degraded input を他手法と一致

## 2週間スケジュール

### Day 1-2

- 論文の主張を固定
- `eval_v1` frozen spec の確認と周辺 manifest の整理
- 20 ケース manifest を整理
- 実験条件表のたたき台を作る

### Day 3-4

- `proposed` を `heuristic repair + beam enhancement` へ変更
- threshold-aware な採択ロジックを導入
- do-no-harm 制約を実装

### Day 5

- stress 強度を必要最小限だけ調整
- fail case 数を増やす

### Day 6-7

- 20 ケース × 3 条件を再実行
- `metrics.csv` / case-wise table / scenario-wise table を更新
- `Adopt_core` / `Adopt_entry` / `Delta_repair` を確認

### Day 8

- pipeline overview 図
- 20 ケース summary
- `Adopt_core` / `Adopt_entry` 比較図
- `Delta OOE` / `Delta C_vis_start` 箱ひげ
- `Delta_repair` 箱ひげ
- 代表例 before / after 図

### Day 9-10

- Method / Experiment / Results 草稿を書く
- 図表番号と用語を固定する

### Day 11

- 関連研究を絞って書く
- floorplan/layout generation
- robotics evaluation
- patient room / bedside / fall risk
- furniture layout optimization

### Day 12

- `LLM-Refine` を入れるか最終判断
- 1日で入らないなら見送る

### Day 13

- Abstract / Intro / Conclusion を完成させる

### Day 14

- 全体整合チェック
- 用語統一
- 投稿準備

## 成果物一覧

提出までに最低限必要な成果物は以下。

- `eval_v1.json` 最終版
- 20 ケース一覧表
- 20 ケース × 3 条件の metrics 集計 CSV
- scenario-wise summary 表
- case-wise before / after 可視化
- recovery rate 図
- `Delta OOE`, `Delta C_vis_start`, `Delta_repair` 図
- Method / Experiment / Results の草稿

## 論文上の見せ方

最終的な主張は、次の表現に寄せる。

> 図面 / スケッチから生成した介護居室レイアウトに対して、入口直後の状況把握を含む導入可否評価を定義し、乱れた環境でも最小変更で導入可能側に戻す評価・改善ループを実現した。

そのために、結果の見せ方は次を優先する。

- `Adopt_core` 回復率
- `Adopt_entry` 回復率
- `Delta OOE_primary`
- `Delta C_vis_start`
- `Delta_repair`

## 直近の実装タスク

次に着手する具体作業は以下。

1. `proposed` を heuristic warm start 型に改修する
2. occupancy / plot の不整合を点検して潰す
3. stress generator の強度を微調整する
4. 20 ケース × 3 条件を再実行する
5. 結果図と本文草稿を並行して作る

## 判断基準

この計画で最重要なのは、「新しいことを増やす」よりも「主張を成立させる」こと。

したがって、以下を満たせば次へ進む。

- 評価条件が固定されている
- 可視化と評価が一致している
- `proposed` の役割が `heuristic` と明確に分かれている
- 主張に必要な図表が揃っている

逆に、これらが固まる前に新規要素を追加しない。
