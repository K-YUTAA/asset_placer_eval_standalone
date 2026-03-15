# Paper Implementation Inventory (2026-03-16)

## Purpose

この文書は、**2026-03-16 時点の現行コード**を基準に、論文に必要となる実装要素を一覧化したものです。

- 対象: upstream generation, evaluator, stress benchmark, refinement, orchestration
- 基準: **現行実装 + 現在の canonical / frozen config**
- 目的: method / experiment / appendix に書くべき式・パラメータ・実装状況を 1 本で参照できるようにする

## 1. Canonical / Frozen Artifacts

### 1.1 Main evaluation and method freeze

- `experiments/configs/eval/eval_v1.json`
  - frozen evaluation spec
- `experiments/configs/refine/proposed_beam_v1.json`
  - frozen proposed refine spec

### 1.2 Upstream pipeline configs

- `experiments/configs/pipeline/fixed_mode_v2_gpt_high_batch_20260222.json`
  - fixed-mode experiment reproduction の canonical upstream config
- `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`
  - parser-consistent な latest-design frozen snapshot
  - README / user-facing execution example

### 1.3 Main stress config

- `experiments/configs/stress/stress_v2_natural.json`
  - current main stress benchmark config

### 1.4 Extension / protocol-specific configs

- `experiments/configs/refine/clutter_assisted_v1.json`
  - clutter-assisted recovery extension

## 2. System Scope for the Paper

### 2.1 Problem scope

- 対象空間: 単一個室（トイレ付き含む）
- 主タスク: `entrance -> bedside`
- 主目的: fidelity-heavy reconstruction ではなく、**deployability evaluation + minimal-change refinement**

### 2.2 Main pipeline

現行 main pipeline は次の 3 段で構成されます。

1. Step1: OpenAI で room / furniture / text interpretation を行う
2. Step2: Gemini spatial understanding を 3-pass で行う
   - furniture
   - room inner frame
   - openings
3. Step3: rule-based integration で `layout_generated.json` を作る

その後、

4. evaluator (`eval_v1`)
5. stress generator (`stress_v2_natural`)
6. refiner (`original / heuristic / proposed`)

が続きます。

## 3. Upstream Generation Configuration

### 3.1 Fixed-mode canonical upstream config

Source: `experiments/configs/pipeline/fixed_mode_v2_gpt_high_batch_20260222.json`

#### Defaults

- `python_exec = .venv/bin/python`
- `model = gpt-5.2`
- `reasoning_effort = high`
- `text_verbosity = high`
- `image_detail = high`
- `max_output_tokens = 32000`
- `enable_gemini_spatial = true`
- `gemini_model = gemini-3-flash-preview`
- `gemini_temperature = 0.6`
- `gemini_thinking_budget = 0`
- `gemini_max_items = 24`
- `gemini_resize_max = 640`
- `gemini_include_non_furniture = false`
- `eval config = experiments/configs/eval/eval_v1.json`
- `plot enabled = true`
- `bg_crop_mode = none`
- `summary_args.enabled = true`

#### Prompt files

- `prompts/fixed_mode_20260222/step1_openai_prompt.txt`
- `prompts/fixed_mode_20260222/gemini_furniture_prompt.txt`
- `prompts/fixed_mode_20260222/gemini_room_inner_frame_prompt.txt`
- `prompts/fixed_mode_20260222/gemini_openings_prompt.txt`

### 3.2 Latest-design frozen snapshot

Source: `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`

fixed-mode とほぼ同じですが、現行 parser に合わせて次を明示しています。

- `gemini_task = boxes`
- `gemini_label_language = English`
- `gemini_openings_retry_max_retries = 1`
- `gemini_openings_retry_temperature = -1.0`
- `gemini_openings_retry_min_outer_door_width_ratio = 0.72`
- `gemini_openings_retry_max_outer_door_width_ratio = 1.28`
- `gemini_openings_retry_max_outer_center_dist_m = 0.85`
- `gemini_inner_frame_retry_max_retries = 1`
- `gemini_inner_frame_retry_min_coverage_x = 0.88`
- `gemini_inner_frame_retry_min_coverage_y = 0.88`
- `gemini_inner_frame_retry_min_anchor_inside_ratio = 0.55`
- `gemini_inner_frame_retry_temperature = -1.0`

## 4. Evaluator (`eval_v1`) Summary

Source:

- `experiments/configs/eval/eval_v1.json`
- `experiments/src/eval_metrics.py`
- `experiments/src/task_points.py`

### 4.1 Frozen evaluator parameters

| key | value |
| --- | --- |
| `grid_resolution_m` | `0.05` |
| `robot_radius_m` | `0.28` |
| `clr_feasible_max_m` | `2.0` |
| `clr_feasible_tol_m` | `0.01` |
| `clr_feasible_max_iters` | `14` |
| `occupancy_exclude_categories` | `["floor"]` |
| `sample_step_m` | `0.50` |
| `max_sensor_samples` | `10` |
| `tau_R` | `0.90` |
| `tau_clr` | `0.10` |
| `tau_clr_feasible` | `0.10` |
| `tau_clr_astar` | `0.10` |
| `tau_V` | `0.60` |
| `tau_Delta` | `0.10` |
| `lambda_rot` | `0.50` |
| `entry_observability.mode` | `both` |
| `entry_observability.sensor_height_m` | `0.60` |
| `entry_observability.num_rays` | `720` |
| `entry_observability.max_range_m` | `10.0` |
| `entry_observability.tau_p` | `0.05` |
| `entry_observability.tau_v` | `0.30` |
| `adopt.clearance_metric` | `clr_feasible` |
| `adopt.entry_gate.metric` | `OOE_R_rec_entry_surf` |
| `adopt.entry_gate.min_value` | `0.70` |

### 4.2 Occupancy and free space

部屋マスクを `R`、占有セルを `O`、解像度を `\Delta`、ロボット半径を `r` とします。

障害物膨張半径セル数:

\[
\rho = \left\lceil \frac{r}{\Delta} \right\rceil
\]

膨張占有:

\[
O^{(\rho)} = \mathrm{Inflate}(O, \rho)
\]

通行可能セル集合:

\[
F = R \setminus O^{(\rho)}
\]

実装上は `_inflate_occupancy` と `_build_free_mask` で構成されています。

### 4.3 Start / Goal generation

#### Start

Source: `experiments/src/task_points.py`

- `task.start.mode = entrance_slidingdoor_center`
- `task.start.door_selector.strategy = largest_opening`
- `task.start.in_offset_m = 0.50`

手順:

1. door 候補から `largest_opening` を選ぶ
2. door center を `t`
3. room centroid を `c`
4. 室内方向単位ベクトル

\[
u = \frac{c - t}{\|c - t\|}
\]

5. unsnapped start

\[
s_0 = t + 0.50\,u
\]

6. `max_radius_cells = 30` で nearest free cell に snap

#### Goal

- `task.goal.mode = bedside`
- `task.goal.offset_m = 0.60`
- `task.goal.bed_selector.strategy = first`
- `task.goal.choose = closest_to_room_centroid`

ベッド中心を `(x_b, y_b)`、yaw を `\theta`、寸法を `(L, W)` とします。

ローカル軸:

\[
\hat{x} = (\cos\theta, \sin\theta), \quad
\hat{y} = (-\sin\theta, \cos\theta)
\]

bedside 候補は **長辺両側**から生成されます。  
したがって、offset は短辺法線方向に入ります。

\[
d =
\begin{cases}
\frac{W}{2} + 0.60 & \text{if } L \ge W \\
\frac{L}{2} + 0.60 & \text{if } L < W
\end{cases}
\]

候補点:

\[
g_1 = (x_b, y_b) + d\,\hat{n}, \quad
g_2 = (x_b, y_b) - d\,\hat{n}
\]

ここで `\hat{n}` は短辺法線です。

最終候補は room centroid `c` に近い側:

\[
g_{\mathrm{bed}} = \arg\min_{g \in \{g_1, g_2\}} \|g - c\|
\]

その後、start と同様に nearest free cell に snap されます。

### 4.4 Reachability

到達可能セル集合を `\mathrm{Reach}(F, s)` とすると、

\[
R_{\mathrm{reach}} = \frac{|\mathrm{Reach}(F, s)|}{|F|}
\]

実装は 4-neighborhood BFS です。

### 4.5 Path and path-based clearance

start から goal まで 8-neighborhood A* を実行して `P = \{p_i\}` を得ます。

各 path cell で occupied までの距離場を `D_{\mathrm{occ}}(p)` とすると、

\[
\mathrm{clr\_min\_astar}
=
\min_{p \in P} \left( D_{\mathrm{occ}}(p) - r \right)
\]

この最小点が `bottleneck_cell` です。

### 4.6 Feasible clearance

`clr_feasible` は、障害物を追加膨張させてもなお path が存在する最大余裕です。

\[
\mathrm{clr\_feasible}

=
\max \left\{ c \ge 0 \,\middle|\, \exists \text{ path from } s \text{ to } g \text{ in } R \setminus \mathrm{Inflate}(O, \lceil (r+c)/\Delta \rceil ) \right\}
\]

実装は binary search です。

初期上限:

\[
hi = \min \left( D_{\mathrm{occ}}(s)-r,\; D_{\mathrm{occ}}(g)-r,\; \texttt{clr\_feasible\_max\_m} \right)
\]

反復:

\[
mid = \frac{lo + hi}{2}
\]

`robot_radius + mid` で occupancy を再膨張し、path が存在すれば `lo = mid`、なければ `hi = mid` と更新します。

停止条件:

- `hi - lo <= clr_feasible_tol_m`
- または `iterations >= clr_feasible_max_iters`

返値は feasible 側の `lo` です。

### 4.7 Path visibility (`C_vis`)

path 上からの sensor sample 集合を `S` とします。

- `S` は
  - start cell
  - raw/smoothed path 上の等間隔 sample
  から作られます

free cell `f \in F` が、いずれかの sample から line-of-sight 可視なら visible とします。

\[
C_{\mathrm{vis}} = \frac{|\{f \in F \mid \exists s \in S,\ \mathrm{LOS}(s,f)=1\}|}{|F|}
\]

### 4.8 Entry visibility (`C_vis_start`)

start 固定で raw occupancy 上の visible free を数えます。

\[
C_{\mathrm{vis\_start}}
=
\frac{|\{f \in F_{\mathrm{raw}} \mid \mathrm{LOS}(s,f)=1\}|}{|F_{\mathrm{raw}}|}
\]

ここで `F_raw` は inflated ではなく raw occupancy から作る free mask です。

### 4.9 Entry observability (`OOE`)

entry_observability mode は `both` で、first-hit と surface を両方計算します。  
main 指標は `OOE_R_rec_entry_surf` です。

#### First-hit mode

各 ray の first hit object を集計し、物体 `j` の hit 率を

\[
p^{(j)}_{\mathrm{hit}} = \frac{h_j}{N_{\mathrm{rays}}}
\]

とします。

連続値:

\[
\mathrm{OOE\_C\_obj\_entry\_hit}
=
\frac{\sum_j w_j p^{(j)}_{\mathrm{hit}}}{\sum_j w_j}
\]

認識率:

\[
\mathrm{OOE\_R\_rec\_entry\_hit}
=
\frac{1}{N_{\mathrm{obj}}}
\sum_j \mathbf{1}\!\left[p^{(j)}_{\mathrm{hit}} \ge \tau_p \right]
\]

現行値は `\tau_p = 0.05` です。

#### Surface mode

物体 `j` の boundary cell 総数を `B_j`、visible boundary cell 数を `V_j` とすると、

\[
v^{(j)}_{\mathrm{surf}} = \frac{V_j}{B_j}
\]

連続値:

\[
\mathrm{OOE\_C\_obj\_entry\_surf}
=
\frac{\sum_j w_j v^{(j)}_{\mathrm{surf}}}{\sum_j w_j}
\]

認識率:

\[
\mathrm{OOE\_R\_rec\_entry\_surf}
=
\frac{1}{N_{\mathrm{obj}}}
\sum_j \mathbf{1}\!\left[v^{(j)}_{\mathrm{surf}} \ge \tau_v \right]
\]

現行値は `\tau_v = 0.30` です。

### 4.10 Layout change

共通 object 集合を `\mathcal{M}`、そのうち movable objects を対象とします。  
object `i` の面積重み:

\[
w_i = \frac{A_i}{\sum_{k \in \mathcal{M}} A_k}
\]

room 対角長を `d_{\mathrm{room}}`、基準 layout との差を

- 平行移動 `\Delta p_i`
- 回転差 `\Delta \theta_i`

とすると、

\[
\Delta_{\mathrm{layout}}
=
\sum_{i \in \mathcal{M}}
w_i
\left(
\frac{\Delta p_i}{d_{\mathrm{room}}}
+
\lambda_{\mathrm{rot}}
\frac{|\Delta \theta_i|}{\pi}
\right)
\]

現行値は `\lambda_rot = 0.50` です。

### 4.11 Adopt rules

#### Adopt core

現行 canonical definition は `clr_feasible` 基準です。

\[
\mathrm{Adopt}_{\mathrm{core}} = 1
\iff
\left(
R_{\mathrm{reach}} \ge \tau_R
\right)
\land
\left(
\mathrm{clr\_feasible} \ge \tau_{\mathrm{clr\_feasible}}
\right)
\land
\left(
C_{\mathrm{vis}} \ge \tau_V
\right)
\land
\left(
\Delta_{\mathrm{layout}} \le \tau_{\Delta}
\right)
\]

現行値:

- `\tau_R = 0.90`
- `\tau_clr_feasible = 0.10`
- `\tau_V = 0.60`
- `\tau_Delta = 0.10`

#### Adopt entry

\[
\mathrm{Adopt}_{\mathrm{entry}} = 1
\iff
\mathrm{Adopt}_{\mathrm{core}} = 1
\land
\mathrm{OOE\_R\_rec\_entry\_surf} \ge 0.70
\]

#### Validity

\[
\mathrm{validity} = 1
\iff
s \in F \land g \in F
\]

## 5. Current Stress Benchmark (`stress_v2_natural`)

Source:

- `experiments/configs/stress/stress_v2_natural.json`
- `experiments/src/generate_stress_cases.py`

### 5.1 Main taxonomy

Current main taxonomy:

- `base`
- `usage_shift`
- `clutter`
- `compound`

`bottleneck / occlusion` は **main taxonomy ではなく targeted diagnostic** です。

### 5.2 Shared constraints

| key | value |
| --- | --- |
| `same_room_only` | `true` |
| `translation_max_m` | `0.6` |
| `rotation_max_deg` | `30.0` |
| `door_keepout_radius_m` | `0.5` |
| `overlap_ratio_max` | `0.05` |
| `require_validity` | `true` |

fixed categories:

- `bed`
- `toilet`
- `sink`
- `door`
- `window`
- `opening`
- `floor`

### 5.3 Difficulty labels

`stress_v2_natural` では difficulty label は **post-generation only** です。  
生成 objective の一部ではありません。

labels:

- `mild`
- `borderline`
- `hard_recoverable`

### 5.4 Usage shift

`usage_shift` は movable furniture の自然な生活ずれを模擬します。

#### Object count distribution

- 1 object: `0.75`
- 2 objects: `0.25`

#### Eligible / low-priority categories

- eligible:
  - `chair`
  - `coffee_table`
  - `small_storage`
  - `table`
- low priority:
  - `sofa`
  - `storage`
  - `tv_cabinet`
  - `cabinet`

#### Local-frame priors

| category | fb sigma | fb clip | lat sigma | lat clip | yaw sigma | yaw clip |
| --- | --- | --- | --- | --- | --- | --- |
| `chair` | `0.12` | `0.30` | `0.05` | `0.12` | `7.0 deg` | `15.0 deg` |
| `coffee_table` | `0.08` | `0.20` | `0.08` | `0.20` | `5.0 deg` | `10.0 deg` |
| `small_storage` | `0.08` | `0.20` | `0.08` | `0.20` | `5.0 deg` | `10.0 deg` |
| `table` | `0.05` | `0.12` | `0.05` | `0.12` | `3.0 deg` | `6.0 deg` |

生成は candidate pool 方式です。

- pool target: `12`
- max attempts: `48`

評価手順:

1. object count を分布から sample
2. local-frame displacement / yaw を sample
3. constraints を満たす candidate を作る
4. `evaluate_layout(..., base_layout, eval_cfg)` で評価
5. validity=1 の候補のみ pool に入れる
6. pool から **random selection**

つまり main `usage_shift` は **target metric optimization ではなく、constraint-bounded random pool selection** です。

### 5.5 Clutter

`clutter` は既存家具を動かさず、外部障害物を追加します。

#### Count distribution

- 1 object: `0.8`
- 2 objects: `0.2`

#### Catalog

| size_xy_m | height_m | weight |
| --- | --- | --- |
| `[0.35, 0.35]` | `1.0` | `0.6` |
| `[0.60, 0.40]` | `1.0` | `0.4` |

#### Placement bands

| band_id | anchor | shape | parameters | probability |
| --- | --- | --- | --- | --- |
| `entry_staging_band` | `start_point` | `oriented_rectangle` | tangent length `1.0`, normal width `0.6` | `0.45` |
| `path_staging_band` | `start_goal_segment` | `segment_band` | `t in [0.25, 0.75]`, normal half width `0.45` | `0.35` |
| `bedside_staging_band` | `bed_perimeter_goal_side` | `perimeter_band` | offset range `[0.2, 0.8]` | `0.20` |

#### Pose jitter

- `yaw_base_deg = [0, 90]`
- `yaw_jitter_deg = 10`
- second object rule:
  - `prefer_different_band_then_fallback_same_band`

生成は usage_shift と同様に candidate pool 方式です。

- pool target: `12`
- max attempts: `48`

### 5.6 Compound

`compound` は次の recipe を順に適用します。

1. `usage_shift` を 1 object
2. その出力に対して `clutter` を 1 object 追加

config:

- `usage_shift_object_count = 1`
- `clutter_object_count = 1`

### 5.7 Diagnostic stress (not main)

旧 targeted diagnostic は `stress_v1_targeted` に残っています。

- `targeted_bottleneck`
- `targeted_occlusion`

これは main paper taxonomy ではなく、**diagnostic / appendix 扱い**が妥当です。

## 6. Refinement Methods

Source:

- `experiments/src/run_trial.py`
- `experiments/src/refine_heuristic.py`
- `experiments/src/refine_proposed_beam.py`
- `experiments/configs/refine/proposed_beam_v1.json`

### 6.1 Comparison protocols

Current comparison:

- `original`
- `heuristic`
- `proposed`

Extension:

- `clutter_assisted`

### 6.2 Original

`original` は post-refinement を行いません。  
`run_v0_freeze` の出力 layout をそのまま evaluator に通します。

### 6.3 Heuristic refine

#### Search space

- translation neighborhood:
  - `dx, dy \in \{-step_m, 0, +step_m\}`
- rotation neighborhood:
  - current yaw の `-rot_deg, 0, +rot_deg`
  - ただし layout axis alignment prior が有効なら orthogonal candidates を使う

#### Target object selection

優先順位:

1. `bottleneck_cell` があれば、それに最も近い movable object
2. なければ start-goal 線分に最も近い movable object

最大変更物体数 `max_changed_objects` に達した後は、既に changed 済み object だけを再利用します。

#### Hard constraints

- object inside room
- optional wall margin
- optional door keepout
- optional overlap ratio max
- candidate validity = 1
- non-regression:
  - `R_reach(candidate) >= R_reach(current)`
  - `clr(candidate) >= clr(current)`

#### Score

heuristic の score は

\[
\mathrm{Score}_{\mathrm{heuristic}}
=
\alpha C_{\mathrm{vis}}
+
\beta R_{\mathrm{reach}}
+
\eta \,\mathrm{clr}
-
\gamma \Delta_{\mathrm{layout}}
-
\mathrm{extra\_penalty}
-
\mathrm{penalty}
\]

ここで実装既定は

- `\alpha = 1`
- `\beta = 1`
- `\eta = 1`
- `\gamma = 0.5`

追加 penalty:

- `validity = 0` なら `+5`
- `R_reach <= 0` なら `+2`

#### Stop condition

各 iteration で改善がなければ停止します。

#### Default parameters

run-time defaults:

- `refine_max_iterations = 30`
- `refine_step_m = 0.10`
- `refine_rot_deg = 15.0`
- `refine_max_changed_objects = 3`

### 6.4 Proposed refine (beam search)

#### Frozen parameters

Source: `experiments/configs/refine/proposed_beam_v1.json`

| key | value |
| --- | --- |
| `refine_step_m` | `0.1` |
| `refine_rot_deg` | `15.0` |
| `refine_max_changed_objects` | `3` |
| `refine_beam_width` | `5` |
| `refine_depth` | `3` |
| `refine_candidate_objects_per_state` | `2` |
| `refine_eval_budget` | `780` |
| `refine_ooe_primary` | `OOE_R_rec_entry_surf` |
| `refine_use_lexicographic` | `true` |
| `refine_allow_intermediate_regression` | `true` |
| `refine_door_keepout_radius_m` | `0.0` |
| `refine_overlap_ratio_max` | `0.05` |
| `refine_delta_weight` | `0.3` |

#### Candidate object priority

候補 object `i` の中心を `x_i` とすると、

\[
\mathrm{priority}(i)
=
\frac{1}{d_{\mathrm{path}}(x_i) + \varepsilon}
+
0.7\frac{1}{d_{\mathrm{entry}}(x_i) + \varepsilon}
+
0.8\frac{1}{d_{\mathrm{bneck}}(x_i) + \varepsilon}
\]

ここで

- `d_path`: start-goal 線分への距離
- `d_entry`: start への距離
- `d_bneck`: bottleneck への距離

です。

#### Continuous score

threshold `\tau` に対する normalized margin:

\[
\mathrm{margin}(x;\tau)
=
\mathrm{clip}\!\left(\frac{x-\tau}{|\tau|}, -1, 1\right)
\]

continuous score:

\[
\mathrm{Score}_{\mathrm{cont}}
=
1.0\,m_{\mathrm{clr}}
+
1.0\,m_R
+
0.2\,m_{\mathrm{vis}}
+
1.0\,m_{\mathrm{start}}
+
1.0\,m_{\mathrm{ooe}}
-
\lambda_{\Delta}\Delta_{\mathrm{layout}}
\]

with

- `m_clr = margin(clearance_value, tau_clr)`
- `m_R = margin(R_reach, tau_R)`
- `m_vis = margin(C_vis, tau_V)`
- `m_start = margin(C_vis_start, tau_V)`
- `m_ooe = margin(OOE_primary, tau_ooe)`
- `\lambda_{\Delta} = refine_delta_weight`

#### Lexicographic score tuple

現行 frozen proposed は lexicographic ordering を使います。

\[
\mathrm{score\_tuple}
=
(
\mathrm{validity},
\mathrm{Adopt}_{\mathrm{entry}},
\mathrm{Adopt}_{\mathrm{core}},
\mathrm{Score}_{\mathrm{cont}}
)
\]

beam 内の比較はこの tuple の大小で行います。

#### Search loop

1. initial node を評価
2. 各 depth で beam 内各 state から top-k object を選ぶ
3. translation / rotation action を展開
4. constraints と validity を満たす candidate を評価
5. visited state を除外
6. top `beam_width` を次 layer に残す
7. `depth` または `eval budget` に達したら停止

#### Default eval budget

`max_eval_calls <= 0` の場合は

\[
\mathrm{budget}
=
\mathrm{beam\_width}
\times
\mathrm{candidate\_objects\_per\_state}
\times
9
\times
\mathrm{depth}
\]

を使います。  
ここで `9` は translation action 数です。

### 6.5 Clutter-assisted extension

Source: `experiments/configs/refine/clutter_assisted_v1.json`

これは main comparison ではなく extension です。

#### Parameters

| key | value |
| --- | --- |
| `grid_step_m` | `0.2` |
| `candidate_limit_per_object` | `28` |
| `door_keepout_radius_m` | `0.5` |
| `overlap_ratio_max` | `0.05` |
| `delta_weight` | `0.02` |
| `rotation_mode` | `rectangular_clutter_room_axis_only` |
| `square_clutter_rotation` | `false` |
| `square_tolerance_m` | `0.05` |
| `heuristic.max_iterations` | `4` |
| `proposed.beam_width` | `6` |
| `proposed.depth` | `3` |
| `proposed.candidate_limit_per_object` | `20` |

#### Objective order

1. `validity`
2. `Adopt_entry`
3. `Adopt_core`
4. `clr_feasible`
5. `C_vis_start`
6. `OOE_R_rec_entry_surf`
7. `R_reach`
8. `C_vis`
9. `delta_layout_clutter`

## 7. Trial / Batch Orchestration

### 7.1 JSON pipeline batch runner

Source: `experiments/src/run_pipeline_from_json.py`

1. `generate_layout_json.py`
2. `eval_metrics.py`
3. `plot_layout_json.py`

出力:

- `layout_generated.json`
- `metrics.json`
- `debug/`
- `plot_with_bg.png`

batch manifest には少なくとも次が残ります。

- `eval_config_path`
- `eval_config_name`
- `eval_hash`

### 7.2 Trial runner for refine

Source: `experiments/src/run_trial.py`

記録される主要情報:

- `eval_config_path`
- `eval_config_name`
- `eval_hash`
- `method_hash`
- `method_config_path`
- `method_config_sha256`
- `alignment_prior_config_path`
- `alignment_prior_config_sha256`

refine 後には次も出力されます。

- `layout_refined.json`
- `metrics_refined.json`
- `refine_log.json`

## 8. Main vs Extension vs Historical

### 8.1 Main paper-safe items

- upstream fixed pipeline
- `eval_v1`
- `base / usage_shift / clutter / compound`
- `original / heuristic / proposed`
- `Adopt_core`
- `Adopt_entry`
- `C_vis_start`
- `OOE_R_rec_entry_surf`

### 8.2 Extension

- `clutter_assisted`
- layout axis alignment prior

### 8.3 Historical / diagnostic only

- `stress_v1_targeted`
- `targeted_bottleneck`
- `targeted_occlusion`
- stale latest-design v2 config

## 9. What the Paper Can State from the Current Code

### 9.1 Safe to state

- evaluator は `entrance -> bedside` task に対し rule-based start/goal generation を持つ
- clearance の canonical criterion は `clr_feasible`
- entry-side evaluation は `C_vis_start` と `OOE_R_rec_entry_surf`
- main stress taxonomy は `base / usage_shift / clutter / compound`
- proposed は beam search + lexicographic score tuple

### 9.2 Should not be stated as mainline

- `bottleneck / occlusion` を current main stress taxonomy だと言うこと
- `clr_min_astar` を canonical adopt clearance criterion だと言うこと
- `clutter_assisted` を main comparison だと言うこと
- latest-design v2 を現行 parser-consistent config だと言うこと

## 10. Source Index

- `experiments/configs/eval/eval_v1.json`
- `experiments/configs/refine/proposed_beam_v1.json`
- `experiments/configs/refine/clutter_assisted_v1.json`
- `experiments/configs/stress/stress_v2_natural.json`
- `experiments/configs/pipeline/fixed_mode_v2_gpt_high_batch_20260222.json`
- `experiments/configs/pipeline/latest_design_v3_gpt_high_frozen_20260312.json`
- `experiments/src/eval_metrics.py`
- `experiments/src/task_points.py`
- `experiments/src/generate_stress_cases.py`
- `experiments/src/refine_heuristic.py`
- `experiments/src/refine_proposed_beam.py`
- `experiments/src/run_trial.py`
- `experiments/src/run_pipeline_from_json.py`
- `experiments/src/generate_layout_json.py`
