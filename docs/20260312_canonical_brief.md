# Canonical Brief Unified v1

参照記法: A–E = 5本の brief。Rxx = 比較表の論点ID。User裁定 = あなたが今回確定した裁定。

## 1. 研究目的

* 本研究は、手書きスケッチ＋寸法情報から介護居室レイアウトを構造化し、ロボット導入可否を定量評価し、必要なら最小変更で改善する枠組みを作ることを目的とする。[A/C/D/E; R01,R02]
* 主目的は見た目のよい3D生成ではなく、到達可能性・安全余裕・可視性・入口直後の状況把握・変更量に基づく deployability evaluation and refinement にある。[A/C/D/E; R01,R06,R15]
* スコープは単一個室（トイレ付き含む）を主対象とし、主タスクは entrance → bedside を中心に扱う。[A/C/D/E; R03,R09]
* 論文の主役は固定上流生成器の改良ではなく、Evaluator + Refiner による evaluate-and-refine loop である。[A/B/C/D/E; R02,R04,R07]

## 2. 現在採用している結論

* [採用中] 単一個室（トイレ付き含む）を主対象にする。
  採用理由: 5本の brief の共通芯であり、個室内のベッド周辺・トイレ移動・入口把握に論点を集中できるため。[A/C/D/E; R03]

* [採用中] 上流生成器は固定し、論文の主提案は Evaluator + Refiner とする。
  採用理由: 生成器を変動要因から外すことで、下流の評価・改善比較の再現性を守れるため。[A/B/C/D/E; R02,R04]

* [採用中] 上流生成器は GPT + Gemini Spatial Understanding + rule-based integration 系として扱う。
  採用理由: 5本とも、固定上流としてこの系統を前提に下流議論を組み立てているため。[A/C/D/E; R04]

* [採用中] 上流設定は役割分離で管理する。実験再現の正本は `fixed_mode_v2_gpt_high_batch_20260222.json`、README / user-facing 実行例の最新凍結ファイルは `latest_design_v3_gpt_high_frozen_20260312.json` とする。
  採用理由: 実験再現用の正本と説明用の実行例を分けた方が、再開性と誤読防止の両方に有利だから。[User裁定; R05]

* [採用中] 評価は JSON / 2D occupancy ベースで行い、Isaac Sim / 3D は主に可視化に使う。
  採用理由: 再現性・実験速度・探索型改善との相性を優先するため。[A/B/C/D/E; R06]

* [採用中] 評価仕様の正本は `eval_v1.json` とする。個別の凍結値は原則として `eval_v1.json` を参照し、本 brief では今回裁定した論点のみ再掲する。
  採用理由: 評価定義を1本化しないと比較結果の意味が揺れるため。[User裁定; R14 と「eval_v1 は凍結済み」]

* [採用中] start / goal はルールベースで自動生成する。bedside goal は、ベッド姿勢に基づく候補生成と側選択で定義し、`goal.offset_m = 0.60` と `choose = closest_to_room_centroid` の wording で統一する。
  採用理由: 恣意性を減らしつつ、過度に単純化した幾何説明を避けられるため。[A/B/C/D/E + User裁定; R09,R11]

* [採用中] Adopt の clearance criterion は `clr_feasible`、閾値は `tau_clr_feasible = 0.10` とする。`clr_min_m = 0.25` は historical proposal として残す。
  採用理由: 現行の canonical definition を 1 本に固定し、旧案は廃止だが参照可能な形で残すため。[User裁定; R14]

* [採用中] `Adopt_core` と `Adopt_entry` は分けて扱い、`C_vis_start` と `OOE_*` を含む entry-side situational awareness を主評価に入れる。
  採用理由: 入口観測込みの採択と、従来型のコア採択を分けて示すことに研究上の意味があるため。[A/B/C/D/E; R15]

* [採用中] 本文・abstract・method・results では entrance / room entry / entry-side situational awareness の wording を使う。operational checkpoint への一般化は discussion / limitations に限る。
  採用理由: 現時点で共通芯として言えるのは entry までであり、一般化は先走らない方が安全だから。[User裁定; R31]

* [採用中] システムは概念的に Generator / Stress / Evaluator / Refiner を分けて扱う。
  採用理由: 生成・入力条件・評価・改善を分離した方が、比較の意味と再現性が明確になるため。[A/C/D/E; R07]

* [採用中] 論文上の主評価 stress taxonomy は `base / usage_shift / clutter / compound` とする。`usage_shift` は既存可動家具の自然な生活ずれ、`clutter` は外部障害物追加、`compound` はその同時発生である。
  採用理由: 自然な乱れを主評価として整理した方が、現場妥当性と論文の分かりやすさが高いから。[User裁定; R22]

* [採用中] `bottleneck / occlusion` は主評価ではなく、補助的な診断評価として保持する。
  採用理由: 狙い撃ちの厳しい試験は有用だが、自然な乱れの代表とは分けて扱う方が整理しやすいため。[User裁定; R22]

* [採用中] stress manifest は benchmark specification の一部とし、状態は `exists / complete / explainable` を分けて管理する。
  採用理由: 「保存されている」と「論文で十分に再現可能」は別物だから。[User裁定; R23]

* [採用中] manifest については、「保存実装があること」と「benchmark-ready であること」を別判定にする。
  採用理由: 中間状態を表せない二値管理では、実装状況を正しく表現できないため。[User裁定; R23]

* [採用中] main の比較条件は `Original / Heuristic / Proposed` とする。`LLM-Refine` は baseline 候補だが、現時点では main comparison に含めない。
  採用理由: 現在の実装・比較実績と、論文の芯の両方に整合するため。[A/B/C/D/E; R24]

* [採用中] 今稿の Proposed は現行凍結版 `proposed_beam_v1.json` を正本とする。`threshold-repair + do-no-harm` は appendix / planned extension として扱う。
  採用理由: 現行凍結版と再設計版の同条件比較がまだなく、main の比較プロトコルを崩さない方が安全だから。[User裁定; R27]

* [採用中] 論文構成は、現時点では `fidelity / validity` と `stress / refinement` の2本柱を基本線とする。
  採用理由: 多くの brief がこの構成で収束しており、スコープ管理にも有利だから。[A/C/D/E; R30]

* [採用中] 現状の比較解釈として、Heuristic は採択率回復に強く、Proposed は連続値改善に強い。
  採用理由: 5本の比較整理で一貫して共有されている現状認識だから。[A/B/C/D/E; R28]

## 3. 保留中の論点

* [保留] `geometry / occupancy / plot` の不整合を完全に解消し、occupancy を唯一の真値として運用できているか。[A/B/C/D/E; R20]
* [保留] 新しい stress taxonomy（`base / usage_shift / clutter / compound` と `bottleneck / occlusion` の役割分離）が、既存 benchmark と実装にどこまで反映済みか。[User裁定 + A/B/C/D/E; R22]
* [保留] `compound` の変更量を 1 本の総量で出すか、家具移動量と追加障害量を分けて持つか。現時点では後者を優先する。[User裁定; R22]
* [保留] manifest の dataset-level QA と `benchmark-ready` 判定が最終的に満たされているか。15/15 DoD の達成状況は要確認。[User裁定; R23]
* [保留] fidelity gate / reconstruction adequacy を、論文でどの最小仕様まで明示するか。[A/B/C/D/E; R29]
* [保留] `LLM-Refine` を今回の論文に入れるか、明示的に外すか。[A/B/C/D/E; R24]
* [保留] 第2タスク `bedside → toilet` を今回含めるか。[A/C/E; R12]
* [保留] 論文内の表記を `G0 / S1 / E1 / R2` と `v0 / v1 / v2` のどちらで統一するか。[A/C/D/E; R08]

## 4. 廃止・不採用となった案

* [廃止] 生成器だけを主提案にする案。
  廃止理由: 見た目生成だけでは「導入可否をどう判断し、どう改善するか」に答えにくい。
  代替: 固定上流生成器を前提に、Evaluator + Refiner を主提案にする。[A/C/D/E; R02]

* [不採用] いきなり重い3D / 実ロボ / 動的人シミュレーションを主実験にする案。
  廃止理由: 実装負荷が高く、再現性と比較実験の回転速度を損なう。
  代替: JSON / 2D occupancy ベース evaluator と stress benchmark を主軸にする。[A/C/D/E; R06]

* [不採用] 施設全体、bath / kitchen / 共用部まで主論文スコープを広げる案。
  廃止理由: 面白いが、個室レイアウト評価・改善という芯が散る。
  代替: 単一個室中心に絞る。[A/C/D/E; R03]

* [不採用] `LLM-Refine` だけを主提案にする案。
  廃止理由: 論文の芯が evaluator-guided refinement から外れてしまう。
  代替: `LLM-Refine` は baseline 候補に留める。[A/C/D/E; R24]

* [廃止] stress 全体を「movable-only の摂動」と一括定義する書き方。
  廃止理由: `clutter = 外部障害物追加` と矛盾し、自然な乱れと診断評価も分けにくい。
  代替: stress 種別ごとに許容操作を分け、主評価と診断評価を分離する。[User裁定; R22]

* [廃止] `bottleneck / occlusion / clutter` をそのまま主評価 taxonomy とみなす案。
  廃止理由: 狙い撃ちの診断評価と、自然な乱れの代表を分けた方が論文として整理しやすい。
  代替: 主評価は `base / usage_shift / clutter / compound`、診断評価は `bottleneck / occlusion` とする。[User裁定; R22]

* [廃止] manifest を「実装済み / 未完成」の二択だけで書く方針。
  廃止理由: ファイルはあるが説明性が足りない、という中間状態を表せない。
  代替: `exists / complete / explainable` を分けて管理する。[User裁定; R23]

* [廃止] Adopt の canonical clearance criterion を `clr_min_m = 0.25` とする案。
  廃止理由: 現行の canonical definition と一致しない。
  代替: `clr_feasible` と `tau_clr_feasible = 0.10` を採用する。[User裁定; R14]

* [廃止] `latest_design_v2_gpt_high.json` を実験再現の canonical upstream config とする案。
  廃止理由: 説明用の latest と、実験再現用の正本を分けた方が再開しやすい。
  代替: `fixed_mode_v2_gpt_high_batch_20260222.json` を experiment canonical、`latest_design_v3_gpt_high_frozen_20260312.json` を latest frozen example config にする。`latest_design_v2_gpt_high.json` は stale schema を含む historical file として扱う。[User裁定; R05]

* [不採用] 本文側まで `operational checkpoint` へ一般化した wording を使う案。
  廃止理由: 現時点で研究の共通芯として確実に言えるのは entrance までである。
  代替: 本文は entrance wording、一般化は discussion / limitations に留める。[User裁定; R31]

## 5. 実装・実験の現状

* **実装済み**

  * 固定上流生成器と、その一括実行パイプラインがある。[A/C/D/E; R04,R05]
  * JSON / 2D occupancy ベース evaluator があり、reachability・clearance・visibility・entry observability・layout change・Adopt を扱える。[A/B/C/D/E; R06,R13,R15]
  * start / goal のルールベース生成があり、評価仕様の正本は `eval_v1.json` に固定する運用で整理されている。[A/B/C/D/E + User裁定; R09,R11,R14]
  * `Original / Heuristic / Proposed` の比較基盤がある。[A/B/C/D/E; R24,R25,R26]
  * 実在レイアウト由来の stress benchmark と、その比較結果の蓄積がある。[A/B/C/D/E; R21,R28]
  * manifest 保存処理の存在を前提に benchmark specification を定義できる段階まで来ている。[A/B/C/E + User裁定; R23]

* **未実装 / 未確認**

  * `threshold-repair + do-no-harm` を main protocol と同条件で再実行した extension 比較。[User裁定; R27]
  * `LLM-Refine` baseline の実装と比較。[A/B/C/D/E; R24]
  * fidelity gate / reconstruction adequacy の最小仕様と表の確定。[A/B/C/D/E; R29]
  * 新しい stress taxonomy（`base / usage_shift / clutter / compound` と診断評価の分離）への全面移行状況。[User裁定; R22]
  * manifest の dataset-level QA と `benchmark-ready` 宣言。[User裁定; R23]
  * 第2タスク `bedside → toilet` の採用判断と実装。[A/C/E; R12]

* **ボトルネック**

  * `geometry / occupancy / plot` の不整合が、評価の信頼性を下げている。[A/B/C/D/E; R20]
  * 現行比較では、Proposed が採択率回復で Heuristic を上回れていない。[A/B/C/D/E; R28]
  * benchmark の難度と taxonomy がまだ論文向けに最終整理され切っていない。[A/B/C/D/E + User裁定; R21,R22,R23]
  * fidelity 側の証拠が不足しており、「その部屋を評価した」とどこまで言えるかが未確定である。[A/B/C/D/E; R29]
  * 命名統一と main / appendix の境界整理が、執筆上まだ残っている。[A/C/D/E + User裁定; R08,R27,R31]

6. 論文主張候補
6.1 現時点で本文主張に使ってよいもの

本研究は、介護居室の単一個室を対象に、ロボット導入可否を評価し、必要に応じて最小変更で改善する deployability evaluation + refinement の枠組みを提示する。
根拠: 5本の brief で最も安定して一致した研究目的である。[A/C/D/E; R01,R03]

上流生成器は固定し、本研究の主提案は独立 Evaluator + Refiner による evaluate-and-refine loopである。
根拠: 生成器改善ではなく、下流の評価・改善を主役とする方針が一貫している。[A/B/C/D/E; R02,R04,R07]

評価は JSON / 2D occupancy ベースで行い、重い 3D シミュレーションではなく、再現性・比較可能性・実験速度を優先する。
根拠: evaluator の位置づけは全 brief で一致している。[A/B/C/D/E; R06]

導入可否は、到達可能性・安全余裕・可視性・入口側の状況把握・変更量を同一の評価系で扱う。
根拠: 指標セットの骨格が共通している。[A/B/C/D/E; R13,R15]

本研究では Adopt_core と Adopt_entry を分け、入口側の situational awareness を deployability 評価に明示的に組み込む。
根拠: entry-aware evaluation は main claim に置ける共通論点である。[A/B/C/D/E + User裁定; R15,R31]

main comparison は Original / Heuristic / Proposed の frozen protocol に基づいて行う。
根拠: 現時点で確定している比較条件である。[A/B/C/D/E + User裁定; R24,R27]

現状の比較では、Heuristic は採択率回復に強く、Proposed は連続値改善に強い。
根拠: これは優劣主張ではなく、現時点の観測結果として一貫している。[A/B/C/D/E; R28]

実験では、実在レイアウト由来の通常ケースと perturbation case を用いる。
根拠: benchmark の存在自体は共通である。[A/B/C/D/E; R21]

6.2 discussion なら使えるが main claim にはまだ弱いもの

stress を base / usage_shift / clutter / compound と bottleneck / occlusion に再編すると、自然な乱れと診断評価を分離できる。
位置づけ: 有力な整理方針だが、全面反映と QA は未完である。[User裁定; R22,R23]

threshold-repair + do-no-harm は、現行 Proposed の有望な次期 spec update 候補である。
位置づけ: main ではなく appendix / planned extension として扱う。[User裁定; R27]

entry 側の考え方は、将来的には operational checkpoint 一般へ拡張できる可能性がある。
位置づけ: 一般化は discussion / limitations に限定する。[User裁定; R31]

施設見学の知見は、評価指標・stress 設計・タスク設定の要求抽出として有効だった。
位置づけ: 設計根拠としては有用だが、一般化主張にはまだ弱い。[A/C/D/E]

fidelity gate を導入すれば、「その部屋を評価した」と言える強さを高められる。
位置づけ: 問題意識としては重要だが、現時点では仕様未完である。[A/B/C/D/E; R29]

LLM-Refine は比較ベースライン候補として有望である。
位置づけ: 将来拡張・planned baseline としては書けるが、main claim には使わない。[A/B/C/D/E; R24]

bedside → toilet は現場要件として重要であり、自然な次段階タスク候補である。
位置づけ: 本稿への採用はまだ未確定である。[A/C/E; R12]

6.3 現時点では主張禁止

Proposed は Heuristic より明確に優れている。
禁止理由: 現時点では採択率回復で Heuristic が強い。[A/B/C/D/E; R28]

本研究は、その部屋を忠実に再現した上で deployability を評価している。
禁止理由: fidelity / reconstruction adequacy は未確定である。[A/B/C/D/E; R29]

本研究は benchmark-ready な stress benchmark を完成させた。
禁止理由: taxonomy 再編と manifest QA がまだ残っている。[User裁定; R22,R23]

manifest は十分完成しており、そのまま論文 benchmark として使える。
禁止理由: exists / complete / explainable の最終 QA が未確認である。[User裁定; R23]

entry observability は、一般的 operational checkpoint 評価として実証済みである。
禁止理由: 本文では entry 限定 wording に固定する裁定になっている。[User裁定; R31]

LLM-Refine を含む4条件比較で提案法を検証した。
禁止理由: 4条件比較は未成立である。[A/B/C/D/E; R24]

bedside → toilet を含む multi-task suitability を本稿で扱っている。
禁止理由: 第2タスクは保留中である。[A/C/E; R12]

geometry / occupancy / plot の不整合は解消済みである。
禁止理由: ここは依然として主要ボトルネックである。[A/B/C/D/E; R20]

本研究は人−ロボット−環境の動的相互作用まで統合的に扱っている。
禁止理由: 動的人シミュレーションは本稿の採用範囲外である。[A/C/D/E]

## 7. 参照索引

* **A–E の対応**
  A = 1本目の brief
  B = 2本目の brief
  C = 3本目の brief
  D = 4本目の brief
  E = 5本目の brief

* **研究主軸（deployability evaluation + minimal-change refinement）**
  A:R01,R02 / B:R01 / C:R01,R02 / D:R01,R02 / E:R01,R02

* **単一個室スコープ**
  A:R03 / C:R03 / D:R03 / E:R03

* **固定上流生成器と下流主役化**
  A:R04 / B:R04 / C:R04 / D:R04 / E:R04

* **JSON / 2D evaluator と entry observability**
  A:R06,R15 / B:R06,R15 / C:R06,R15 / D:R06,R15 / E:R06,R15

* **start / goal と clearance の今回確定分**
  User裁定:R11,R14

* **main stress taxonomy と diagnostic stress の分離**
  User裁定:R22

* **manifest の benchmark QA 方針**
  User裁定:R23

* **main Proposed と extension の境界**
  User裁定:R27

* **experiment canonical config と example config の分離**
  User裁定:R05

* **entrance wording を本文で固定する方針**
  User裁定:R31

* **現状比較の解釈（Heuristic は recovery、Proposed は continuous improvement）**
  A:R28 / B:R28 / C:R28 / D:R28 / E:R28

* **未解決の主要論点**
  geometry整合 = A/B/C/D/E:R20
  fidelity gate = A/B/C/D/E:R29
  第2タスク = A/C/E:R12
  命名統一 = A/C/D/E:R08
  LLM-Refine = A/B/C/D/E:R24

## 8. 次にやるべきこと

* **1. geometry / occupancy / plot の単一化を終える**
  完了条件: 代表ケースだけでなく benchmark 全体で、見た目と評価の矛盾が消え、occupancy を唯一の真値として説明できる。

* **2. stress taxonomy を論文用に再編する**
  完了条件: 各ケースが `base / usage_shift / clutter / compound` または `bottleneck / occlusion` のどれに属するか明記され、主評価と診断評価が混ざらない。

* **3. manifest QA を回し、benchmark-ready 判定を分けて出す**
  完了条件: 各サンプルについて `exists / complete / explainable` が記録され、dataset-level QA report が作成される。

* **4. case-wise 比較表を最終 taxonomy に合わせて作る**
  完了条件: 各ケースについて stress 種別、採否、改善量、変更量、移動家具 / 追加障害が一覧化される。

* **5. fidelity gate の最小仕様を決める**
  完了条件: 5実在レイアウトに対して、「評価対象として妥当か」を示す最低限の表が作られる。

* **6. extension としての `threshold-repair + do-no-harm` を仕様化する**
  完了条件: main protocol と切り分けた spec が書かれ、mini rerun を行うか、今回は未実施と明記するかが決まる。

* **7. `LLM-Refine` を今回入れるか最終判断する**
  完了条件: 実装するなら比較条件が固定され、入れないなら本文で planned baseline / out of scope と明記される。

* **8. 第2タスク `bedside → toilet` の採否を決める**
  完了条件: 本稿に含めるか含めないかが明記され、含めないなら理由が本文に入る。

* **9. 命名を論文内で統一する**
  完了条件: `G0 / S1 / E1 / R2` を使うか、既存表記を注記付きで使うかが決まり、混在しない。

* **10. Method / Experiment / Results の main / appendix 境界を固定する**
  完了条件: main には frozen protocol と現行比較、appendix / planned extension には再設計案と一般化議論を置く構成が確定する。

## 9. 判断変更ログ

* **R27 確定**
  Proposed の main は現行凍結版 `proposed_beam_v1.json`。`threshold-repair + do-no-harm` は appendix / planned extension に移した。
  変更点: 「再設計を main に入れるか」という衝突を、「現行版を main、再設計は extension」で閉じた。

* **R22 確定**
  stress は一枚岩ではなく、主評価 `base / usage_shift / clutter / compound` と、診断評価 `bottleneck / occlusion` に分けた。
  変更点: 以前の「clutter をどう作るか」という局所論点から、「自然な乱れ」と「狙い撃ち試験」の二層構造に変わった。

* **R23 確定**
  manifest は benchmark specification の一部とし、`exists / complete / explainable` を分けて管理する方針に変えた。
  変更点: 以前の「manifest がある / ない」の二値判断から、QA 管理へ変わった。

* **R14 確定**
  clearance criterion は `clr_feasible`、閾値は `tau_clr_feasible = 0.10` とした。`clr_min_m = 0.25` は廃止された旧提案として残す。
  変更点: clearance の canonical prose が 1 本化された。

* **R11 確定**
  bedside goal の説明は、「ベッド姿勢に基づく候補生成と側選択」で統一した。
  変更点: 単純な「ベッド端から 0.60m」とは書かず、幾何学説明の統一を優先した。

* **R05 確定**
  upstream config は `experiment canonical` と `example config` を分ける。
  変更点: `latest` を正本と誤読する余地を消した。

* **R31 確定**
  本文は entrance wording、generalization は discussion / limitations のみ。
  変更点: operational checkpoint への一般化を、本文の主張からは外した。

* **eval_v1 凍結前提を採用**
  評価仕様は凍結済みとして扱い、本 brief では今回裁定した論点だけを再掲し、その他の詳細値は `eval_v1.json` に委ねる。
  変更点: 凍結済み前提で閉じられる衝突を閉じ、不必要な再衝突を避けた。
