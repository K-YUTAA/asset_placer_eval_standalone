以下に、**いまの研究状態を正直に反映した、6ページ論文の骨子・章立て・主張・結果の書き方**をまとめます。
前提は「現時点の実装・実験結果を根拠にし、**まだ言い切れないことは言い切らない**」です。

---

# 論文タイトル案

**Deployability-Oriented Evaluation and Minimal-Change Refinement of Care-Room Layouts Generated from Sketches and Dimension Hints**

日本語なら：

**スケッチと寸法情報から生成した介護居室レイアウトに対する，導入可否評価と最小変更リファイン**

---

# 6ページ論文の基本方針

いまの進捗なら、**「生成＋評価＋改善」全部を均等に書くのではなく，主役は“評価と改善”**にするのが最も強いです。

## この論文で一番言いたいこと

* 介護居室にロボットを導入できるかどうかは、**見た目の再現性**だけでなく、
  **到達性・クリアランス・可視性・入口直後の状況把握**で決まる
* 図面/スケッチから作ったレイアウトに対し、その導入可否を**定量評価**し、
* stress 環境では **最小変更で導入可能側へ戻す**ことができる

## この論文で言いすぎないこと

現状の結果では、**提案手法（proposed）が heuristic より明確に優れている、とまではまだ言い切れません**。
現時点ではむしろ、

* **heuristic** は `Adopt_core` / `Adopt_entry` の回復に強い
* **proposed** は `C_vis_start` や `clr_feasible` のような連続値改善に強い

という結果です。
したがって、論文では **「提案手法は entry-aware な連続値改善が得意で、threshold-based repair に向けて改善中」**という書き方が安全です（後述）。
あなたの比較要約でも、20ケースで heuristic の `adopt_core_gain_sum = 4, adopt_entry_gain_sum = 4` に対し、proposed は `1,1` です。一方、平均 `Δclr_feasible` と平均 `ΔC_vis_start` は proposed の方が大きいです。これは正直に書く方が強いです。
（この部分はあなたの手元の `refine_compare_summary.md` と `refine_visual_report.md` に基づく整理です。）

---

# 6ページ構成案（ページ配分つき）

## p1: 1. Introduction

## p1–p2: 2. Related Work

## p2–p3: 3. System Overview and Problem Formulation

## p3–p4: 4. Deployability Evaluator

## p4–p5: 5. Stress Scenarios and Refinement Methods

## p5–p6: 6. Experiments, Current Results, and Discussion

（Conclusionは6節末尾に短く統合）

---

# 各章の骨子

---

## 1. Introduction

### 書くべき内容

1. **背景**

   * 介護・医療系居室では、ロボット導入可否はロボット性能だけでなく、**部屋レイアウト・開口・家具配置・障害物の乱れ**に依存する
   * 実際、患者/利用者支援では「部屋に入る」「ベッドサイドに位置決めする」「入口で状況把握する」といった空間依存タスクが重要である。Chen & Kemp は患者室における bedside positioning を直接扱い、Kapusta らはベッドサイド支援システムを実証している。病院環境でのロボットナビ評価自体も、標準化が不十分で定量ベンチマークが必要とされている。 ([Sites@GeorgiaTech][1])

2. **問題**

   * スケッチ/図面から 3D レイアウトを自動生成する研究はあるが、
     **「その部屋にロボットが導入可能か」を運用指標で評価し、改善するループ**は十分整理されていない
   * floor plan image analysis や reconstruction の研究は進んでいるが、そこから **deployability evaluation** に繋げるギャップがある。 ([arXiv][2])

3. **本研究の狙い**

   * 手書きスケッチ＋寸法ヒントから生成した介護居室レイアウトに対し、

     * 到達性
     * 最小クリアランス
     * 経路上可視性
     * 入口直後の状況把握
     * 変更量
       を用いて導入可否を評価する
   * stress 環境に対して、minimal-change refinement を行う

4. **本稿の貢献（現時点で安全な言い方）**

   * 介護居室向けの **deployability evaluator** を設計した
   * stress benchmark（base / bottleneck / occlusion / clutter）を構築した
   * ルールベース修正と beam-based refinement を比較した
   * 現時点の結果として、**heuristic は閾値回復に強く，proposed は入口観測や clearance の連続値改善に強い**ことを示した

### 査読ブロックポイント

* 「これは生成論文なのか、ロボット論文なのか？」

  * 回答: **ロボット制御そのものではなく、ロボット運用に適した環境評価・改善論文**である、と明記
* 「本当にその部屋を評価しているのか？」

  * 回答: fidelity/validity を前提条件として確認し、その上で deployability を評価する、と構成で先に宣言

---

## 2. Related Work

この節は**長くしない**ことが大事です。
6ページでは3小節で十分です。

### 2.1 Floor-plan / layout understanding

* CubiCasa5K は floorplan image analysis の代表データセット
* Zeng ら（ICCV 2019）は walls / rooms / doors / windows を multi-task で認識
* Lv ら（CVPR 2021）は floor plan recognition and reconstruction を 3D/ベクトル化まで繋げた
  ([arXiv][2])

**この研究との違い**

* 既存は「認識・再構成」が中心
* 本研究は、その出力を **ロボット導入可否評価**に接続する

### 2.2 Language-guided layout generation / editing

* Tell2Design は言語による floor plan generation の代表
* LLplace は LLM による 3D indoor scene layout generation / editing を扱う
  ([ACL Anthology][3])

**この研究との違い**

* 既存はレイアウト生成/編集が主
* 本研究は **生成結果を評価し、運用指標で修正する**ことが主

### 2.3 Robot navigation and perception-oriented evaluation

* occupancy grid はロボット環境表現の古典
* A* と configuration space は経路評価の基礎
* visibility maps は 2D grid 上で可視性と到達性を扱う枠組みを与える
* patient room / bedside / hospital robot benchmarking の文脈で、空間要件と評価軸が重要とされる
  ([IEEE Computer Society][4])

**この研究との違い**

* 既存は robot-side algorithm が主
* 本研究は **room-side evaluation and refinement** に焦点を当てる

---

## 3. System Overview and Problem Formulation

### 3.1 パイプライン

ここは図1で示すと良いです。

* **G0: Frozen Upstream Generator**

  * 入力: 手書きスケッチ + dimension hints
  * 処理: GPT による semantic hypothesis + Gemini Spatial Understanding + rule integration
  * 出力: `layout_v0.json`, USD
* **S1: Stress Scenario Generation**

  * `base`, `bottleneck`, `occlusion`, `clutter`
* **E1: Evaluator**

  * `R_reach`, `clr_feasible`, `C_vis`, `C_vis_start`, `OOE`, `Delta_layout`, `Adopt_core`, `Adopt_entry`
* **R2: Refiner**

  * `heuristic`
  * `proposed`

> ここで、`v0/v1/v2` という用語は混線しやすいので、本文では G0/E1/R2/S1 にした方が安全です。

### 3.2 問題設定

各 layout (\mathcal{L}) に対して

* `start`: 入口 sliding door 中心から室内方向へオフセット
* `goal`: bedside rule で自動生成
* evaluator が導入可否を返す

Refinement は
[
\mathcal{L}^{out} = \arg\max_{\mathcal{L}' \in \mathcal{N}(\mathcal{L})} J(\mathcal{L}')
]
のような constrained optimization として定義する。

### 3.3 Fidelity/validity の位置づけ

ここは短く入れてください。

* 本研究は「現実の部屋の完全デジタルツイン」を主張しない
* まず **評価に必要十分な整合性**を満たしたレイアウトだけを対象にする
* validity を満たさないものは evaluation の対象外とする

> これで「再現できてないなら評価不能では？」という査読コメントに先回りできます。

---

## 4. Deployability Evaluator

ここは本研究の技術の中心の一つです。

### 4.1 Occupancy and reachability

* JSON から 2D occupancy grid を生成
* robot radius を configuration space として反映
* `R_reach` を計算
  occupancy grid, configuration space, A* の流れで説明。 ([IEEE Computer Society][4])

### 4.2 Clearance

* `clr_min` / `clr_feasible`
* 距離変換ベースの clearance
* なぜ clearance が必要か

  * path existence だけでは不十分
  * 狭すぎる経路は導入上危険
    Willms & Yang や distance transform の文献を使う。 ([atrium.lib.uoguelph.ca][5])

### 4.3 Visibility

* `C_vis`: 経路上で見える床面積率
* `C_vis_start`: 入口固定視点での見える床面積率
* visibility maps の文脈に乗せる。 ([CMU School of Computer Science][6])

### 4.4 Entry Observability (OOE)

* 入口固定視点 S から、家具ごとの見え方を測る
* 現状の主指標は `OOE_R_rec_entry_surf`
* `hit` は補助ログ

**ここで現場との接続を書く**
施設見学では、転倒や障害物確認のために「入室直後に何が見えるか」が重要と示唆されたため、経路全体の可視性とは別に、入口直後の observability を定義した。

### 4.5 Delta_layout と Adopt

* `Delta_layout`: 変更量
* `Adopt_core`: core metrics に基づく採択
* `Adopt_entry`: entry observability を含む採択

**ここで現在の設定値を表で出す**

* `tau_R = 0.90`
* `tau_clr_feasible = 0.10`
* `tau_V = 0.60`
* `OOE_R_rec_entry_surf >= 0.70`
  など。

---

## 5. Stress Scenarios and Refinement Methods

### 5.1 Stress benchmark

* 実在レイアウト5件
* 各 layout から

  * `base`
  * `bottleneck`
  * `occlusion`
  * `clutter`
* 合計20ケース

ここで、stress は「人工的に都合よく壊した」のではなく、
**施設見学で得た“可動家具の乱れ・入口付近の障害・床上 clutter” を模擬した controlled perturbations** と書くと強いです。

### 5.2 Compared methods

#### Original

* 生成されたレイアウトをそのまま評価

#### Heuristic

* greedy local refinement
* 1物体ずつ 26近傍を評価
* 非悪化制約つき

#### Proposed

* beam search
* 同じ 26近傍 action set
* lexicographic / margin-based scoring
* entry-aware objective

### 5.3 ここで正直に書くべきこと

現状の proposed は、**平均改善には強いが threshold repair にはまだ弱い**。
この節では、あまり盛らずに

> We compare a greedy heuristic and a beam-based metric-guided refinement. At the current implementation stage, the beam-based method shows stronger continuous improvements in entry visibility and feasible clearance, whereas the greedy baseline remains competitive in recovering binary adoptability under the current thresholds.

くらいに留めるのが安全です。

---

## 6. Experiments, Current Results, and Discussion

この節は、**現状結果を正直に書く**のがポイントです。

### 6.1 実験1: fidelity / validity

ここは表1でよいです。

書く内容：

* 実在5件
* 上流生成器が評価可能なレイアウトを出せること
* validity / main geometry consistency を確認

> もしまだ表が揃っていないなら、「本稿では詳細を supplementary / ongoing」に落としてもよいですが、本来は最低1表ほしいです。

### 6.2 実験2: stress repair benchmark

ここが本命です。

#### 現状の結果（正直に）

* 20ケース比較で

  * **heuristic** は `Adopt_core`, `Adopt_entry` の回復が強い
  * **proposed** は `Δclr_feasible` と `ΔC_vis_start` の平均改善が大きい
* したがって、現時点では

  * heuristic = threshold repair に強い
  * proposed = continuous enhancement に強い

#### 具体的に書くべき数値

現状の比較要約では、20ケースで

* heuristic: `adopt_core_gain_sum = 4`, `adopt_entry_gain_sum = 4`, `mean_delta_clr_feasible = 0.0426`, `mean_delta_C_vis_start = 0.0071`
* proposed: `adopt_core_gain_sum = 1`, `adopt_entry_gain_sum = 1`, `mean_delta_clr_feasible = 0.0618`, `mean_delta_C_vis_start = 0.0314`

です。
さらに scenario 平均では、

* heuristic は base / bottleneck / occlusion で `adopt_core_rate = 0.8`, `adopt_entry_rate = 0.4`
* proposed は同条件で `0.6 / 0.2`
* ただし proposed の `mean_C_vis_start` と `mean_clr_feasible` は概ね高い
  という傾向です。
  （この部分は、あなたの最新の比較要約と visual report を根拠に書く）

#### この結果から安全に言えること

* entry-aware metric を入れた evaluator は、stress による難化を可視化できている
* greedy と beam は、改善の性格が異なる
* proposed は現時点で “repair” より “improvement” に寄っており、threshold-aware redesign が必要

### 6.3 Discussion

ここで査読ブロックポイントを回収します。

#### Block point 1: 「その部屋を評価したと言えるのか」

* 回答: 完全 twin ではなく、評価に必要十分な fidelity を前提に deployability を評価する

#### Block point 2: 「入口観測は本当に必要か」

* 回答: 施設見学で、入室直後の状況把握が重要であると確認された。ゆえに `C_vis_start` / `OOE` を導入した

#### Block point 3: 「proposed が heuristic に負けているのでは？」

* 回答: 現状は threshold recovery では heuristic が強いが、proposed は entry-aware continuous improvement で優位を示している。
  今後は threshold-margin objective により repair performance を強化する予定

---

# 6ページ論文の “現状に合った主張” の書き方

ここが一番大事です。

## 今なら言ってよい主張

* 図面から生成した介護居室を、**ロボット運用観点で定量評価する evaluator** を設計した
* 入口直後の状況把握を含む指標を導入した
* stress benchmark を構築した
* heuristic / proposed の比較から、**binary recovery と continuous improvement のトレードオフ**が見えた

## 今は言わない方がよい主張

* proposed が heuristic より明確に優れている
* 完全に “その部屋” を再現して評価できている
* 人・ロボット・環境を完全統合した最適化ができている

---

# 引用すべき論文と「どこで使うか」

## Introduction

* Chen & Kemp 2011 — bedside positioning task の妥当性 ([Sites@GeorgiaTech][1])
* Kapusta et al. 2019 — bedside assistance の応用文脈 ([PLOS][7])
* Rondoni et al. 2024 — hospital navigation benchmarking の必要性 ([Nature][8])

## Related Work (layout understanding)

* CubiCasa5K 2019 — floorplan image analysis dataset ([arXiv][2])
* Zeng et al. 2019 — floorplan element recognition ([CVF Open Access][9])
* Lv et al. 2021 — floorplan recognition and reconstruction ([CVF Open Access][10])
* Tell2Design 2023 — language-guided floorplan generation ([ACL Anthology][3])
* LLplace 2024 — LLM-based 3D layout generation/editing ([arXiv][11])

## Method (Evaluator)

* Elfes 1989 — occupancy grid ([IEEE Computer Society][4])
* Hart et al. 1968 — A* ([cs.auckland.ac.nz][12])
* Lozano-Perez 1983 — configuration space / inflation ([Springer][13])
* Felzenszwalb & Huttenlocher 2012 — distance transform ([Theory of Computing][14])
* Pereira et al. 2016 — visibility maps ([CMU School of Computer Science][6])
* Willms & Yang 2008 — obstacle clearance emphasis ([atrium.lib.uoguelph.ca][5])

## Method (Refinement)

* Yu et al. 2011 Make it Home — layout optimization with visibility/accessibility costs ([UCLA Computer Science][15])

## Discussion / Context

* Pati et al. 2009 — patient room multi-dimensional assessment ([Brikbase][16])
* Fay et al. 2021 — patient room design attributes for healthcare workers ([ResearchGate][17])
* Piatkowski et al. 2021 — patient room design as fall protection strategy ([PMC][18])

---

# いまの段階での “実験仮説” を明示するとこうなります

## H1

上流生成器は、評価可能な fidelity / validity を持つレイアウトを生成できる。
→ **部分的に支持**（実在5件ベースの evaluator 入力は成立しているが、定量表はまだ整理が必要）

## H2

入口直後の状況把握（`C_vis_start`, `OOE`）は、従来の経路可視性や到達性だけでは捉えられない deployability の要件を表す。
→ **概念的に支持**（施設見学の知見と evaluator 仕様は一致）

## H3

metric-guided refinement は、stress layout に対して deployability を改善できる。
→ **部分的に支持**
現状では heuristic が binary recovery に強く、proposed は continuous metric improvement に強い。
→ proposed の threshold-aware redesign が必要

---

# 最後に：この論文を“いまの状態で最も通しやすくする”ための編集方針

* **提案法を過大主張しない**
* **Evaluator と entry observability の新規性を前面に出す**
* **stress benchmark と real care-room context を強く押す**
* 結果は「優劣」より **“何が分かったか”** を主にする

  * heuristic は recovery-oriented
  * proposed は enhancement-oriented
* 最後に「threshold-aware beam への拡張」で締める

---

必要なら次に、これをそのまま **論文本文の文章（日本語ドラフト）** に落とします。
たとえば、

* Abstract（250〜300語相当）
* Introduction（2〜3段落）
* Contributions 箇条書き
* Method の本文ドラフト
* Results の本文ドラフト

まで、そのまま書けます。

[1]: https://sites.gatech.edu/hrl/wp-content/uploads/sites/231/2016/05/adv_robotics_2011.pdf?utm_source=chatgpt.com "Full paper A Direct Physical Interface for Navigation and ..."
[2]: https://arxiv.org/abs/1904.01920?utm_source=chatgpt.com "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis"
[3]: https://aclanthology.org/2023.acl-long.820.pdf?utm_source=chatgpt.com "A Dataset for Language-Guided Floor Plan Generation"
[4]: https://www.computer.org/csdl/magazine/co/1989/06/r6046/13rRUzpzeFt?utm_source=chatgpt.com "Using Occupancy Grids for Mobile Robot Perception and ..."
[5]: https://atrium.lib.uoguelph.ca/bitstream/10214/7592/1/WillmsYang_IEEE_SMCB_2008.pdf?utm_source=chatgpt.com "Real-Time Robot Path Planning via a Distance-Propagating ..."
[6]: https://www.cs.cmu.edu/~mmv/papers/16iros-tiago.pdf?utm_source=chatgpt.com "Visibility Maps for Any-Shape Robots"
[7]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0221854&utm_source=chatgpt.com "A system for bedside assistance that integrates a robotic bed ..."
[8]: https://www.nature.com/articles/s41598-024-69040-z?utm_source=chatgpt.com "Navigation benchmarking for autonomous mobile robots in ..."
[9]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Zeng_Deep_Floor_Plan_Recognition_Using_a_Multi-Task_Network_With_Room-Boundary-Guided_ICCV_2019_paper.pdf?utm_source=chatgpt.com "Deep Floor Plan Recognition Using a Multi-Task Network ..."
[10]: https://openaccess.thecvf.com/content/CVPR2021/papers/Lv_Residential_Floor_Plan_Recognition_and_Reconstruction_CVPR_2021_paper.pdf?utm_source=chatgpt.com "Residential Floor Plan Recognition and Reconstruction"
[11]: https://arxiv.org/abs/2406.03866?utm_source=chatgpt.com "LLplace: The 3D Indoor Scene Layout Generation and Editing via Large Language Model"
[12]: https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf?utm_source=chatgpt.com "A Formal Basis for the Heuristic Determination of Minimum ..."
[13]: https://link.springer.com/content/pdf/10.1007/978-1-4613-8997-2_20.pdf?utm_source=chatgpt.com "Spatial Planning: A Configuration Space Approach"
[14]: https://theoryofcomputing.org/articles/v008a019/?utm_source=chatgpt.com "Distance Transforms of Sampled Functions"
[15]: https://web.cs.ucla.edu/~dt/papers/siggraph11/siggraph11.pdf?utm_source=chatgpt.com "Automatic Optimization of Furniture Arrangement"
[16]: https://www.brikbase.org/sites/default/files/CADRE_A_Multidimensional_Framework_for_Assessing_Patient_Room_Configurations.pdf?utm_source=chatgpt.com "A Multidimensional Framework for Assessing Patient Room ..."
[17]: https://www.researchgate.net/publication/352883359_Patient_Room_Design_A_Qualitative_Evaluation_of_Attributes_Impacting_Health_Care_Professionals?utm_source=chatgpt.com "(PDF) Patient Room Design: A Qualitative Evaluation of ..."
[18]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8392568/?utm_source=chatgpt.com "Designing a Patient Room as a Fall Protection Strategy - PMC"
