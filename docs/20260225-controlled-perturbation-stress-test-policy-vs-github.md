# Controlled Perturbation実装方針（GitHub比較付き, 2026-02-25）

## 1. 方針の芯

- 「恣意的に壊す」のではなく、現場で起きる乱れ（椅子ずれ・物の仮置き）を上限付きで再現する。
- 生成方法は `evaluation-in-the-loop`（候補生成→評価→採用）で固定する。
- 目的は `validity=1` を保ちながら、狙った運用指標だけを悪化させる stress-test ケースの作成。

## 2. GitHub現状との比較（一次参照: asset_placer_isaac）

- 比較対象:
  - `origin/master` = `d3af84c9f2af08216b1dd27ab91490a3c4295e5b`
  - `origin/exp/eval-loop-v1` = `3944d56f058f5b583579e80038c14df081b4faab`

### 2.1 既にあるもの

- コア評価:
  - `experiments/src/eval_metrics.py`（`R_reach`, `clr_min`, `C_vis`, `Delta_layout`, `Adopt`, `validity`）
- task start/goal 自動解決:
  - `experiments/src/task_points.py`
- trial実行基盤:
  - `experiments/src/run_trial.py`

### 2.2 expブランチにあり、master未統合のもの

- 入口観測系:
  - `C_vis_start`, `OOE_*`, `entry_observability` 設定
  - ファイル: `experiments/src/eval_metrics.py`, `experiments/configs/eval/default_eval.json`

### 2.3 未実装（今回必要）

- `generate_degraded_cases.py`（stress-test生成器）
- `configs/degrade/degrade_v1.json`
- `Adopt_core` / `Adopt_entry` の明示分離
- `Delta_disturb` / `Delta_repair` の分離記録
- `method=proposed` の独立実装（現状は `heuristic` と同経路）

## 3. 乱れ生成の固定仕様（決め打ち）

### 3.1 動かす対象

- movable候補: `chair`, `table`, `small_storage`, `tv_cabinet`
- fixed扱い: `bed`, `toilet`, `sink`, `door/window`, `wall/floor`
- 1ケースあたりの変更数上限:
  - 通常: 1物体
  - clutter: 2物体

### 3.2 摂動バジェット

- 平行移動上限: `d_max_m = 0.6`
- 回転上限: `theta_max_deg = 30`
- 同一 `room_id` 内でのみ移動

### 3.3 reject条件（validity保護）

- OBBが部屋外へはみ出す
- ドア近傍 keep-out 侵入（例: 半径 `0.5m`）
- OBB重なり比が閾値超過（例: `>0.05`）
- evaluator結果 `validity != 1`

### 3.4 壊し過ぎ防止

- 目標は「閾値を少し下回る」状態（ギリギリ悪い）。
- 例: `clr_min ≈ tau_clr - 0.02m` を優先。

## 4. 3種類のstress-test生成ロジック

### 4.1 Bottleneck

- 目標:
  - `validity=1`
  - `path_exists=True`
  - `clr_min < tau_clr`（ただし過度悪化は避ける）
- 手順:
  1. baseline評価で path/狭隘点を取得
  2. 小型 movable を選択
  3. 狭隘点近傍へ候補配置（複数）
  4. 制約を満たし `clr_min` を狙って下げる候補を採用

### 4.2 Occlusion

- 目標:
  - `validity=1`
  - `R_reach >= tau_R` と `clr_min >= tau_clr` を維持
  - `C_vis_start` または `OOE_primary` を悪化
- 手順:
  1. entry視点 `S` を task解決と同規則で取得
  2. `S -> goal` の見通し帯に候補配置
  3. 背の高い movable を優先
  4. 観測系のみ下がる候補を採用

### 4.3 Clutter

- C1（強い乱れ）: `R_reach < tau_R`
- C2（入口限定）: `Adopt_core=1` かつ `Adopt_entry=0`
- 手順:
  - 2手（最大2物体）で探索し、上位候補をbeam状に残す

## 5. 追加ファイル設計

- 追加:
  - `experiments/configs/degrade/degrade_v1.json`
  - `experiments/src/generate_degraded_cases.py`
- 入力:
  - baseline layouts（5件）
  - `eval_v1.json`（凍結）
  - `degrade_v1.json`
- 出力（1ケースごと）:
  - `layout_<baseid>__deg_<type>.json`
  - `disturb_manifest.json`
  - `debug/*`（任意）

## 6. Δlayoutの意味を保つ定義

- 外乱量:
  - `Delta_disturb = Delta(L_deg, L0)`
- 修正量（手法m）:
  - `Delta_repair_m = Delta(L_out_m, L_in)`
- 実装:
  - degrade生成時に `Delta_disturb` 保存
  - trial実行時に `Delta_repair` 保存

## 7. DoD（完了判定）

### 各ケース

- `validity == 1`
- 対象指標が baseline より悪化
  - bottleneck: `clr_min` 低下（経路維持）
  - occlusion: `C_vis_start` または `OOE` 低下（到達系維持）
  - clutter: `R_reach` 低下 または `Adopt_entry` のみ低下

### 全体

- 20ケース（5 normal + 15 degraded）を seed固定で再生成可能
- `Adopt_core=0` ケースを十分確保
- `Adopt_core=1 and Adopt_entry=0` ケースを確保

## 8. 実装順（最短）

1. `eval_v1` 凍結（`Adopt_core`/`Adopt_entry` 仕様固定）
2. `degrade_v1.json` 作成
3. `generate_degraded_cases.py` 実装
4. 5→20ケース生成・JSON凍結
5. `run_trial.py` で 20 x 4条件（Original/LLM/Heuristic/Proposed）実行

## 9. 運用ルール（公平性）

- 乱れ生成は手法非依存で1回だけ作成し、ケースJSONを凍結する。
- 各手法は同じ入力セット・同じseed・同じ評価設定で比較する。


## 10. 追記（詳細実装方針）

これを「恣意的に壊す」のではなく、**現場で起きる“乱れ（椅子がずれる／物が置かれる）”を、上限付きで再現する controlled perturbation（ストレステスト）**として作るのが一番強いです。ここでは *validity=1を保ちながら、狙った運用指標だけを悪化*させる実装方針を決め打ちします。

---

## 0) 結論：乱れた環境の作り方は「評価器つき摂動生成」が最適

あなたの評価が軽い（1回が十分速い）ので、乱れ生成は

* **(A) 乱れの“候補”をたくさん作る（小さな移動・回転）**
* **(B) 評価器でスコアを見て、狙った指標が下がるものを採用する**
* **(C) validityや“壊し過ぎ”を避ける制約で弾く**

という「評価-in-the-loop」でやるのが最も安全で再現性が高いです。
これだと、論文では「現場の乱れを上限付きでモデル化し、運用指標を狙って悪化させる stress-test を生成」と説明できます。

---

## 1) 乱れ生成の基本仕様（これを守れば事故らない）

### 1.1 動かす物を限定（現場らしさ + 再現性）

* **可動（movable）**：`chair`, `table`, `small_storage`, `tv_cabinet` など
* **固定（fixed）**：`bed`, `toilet`, `sink`, `door/window`, `walls`（＝動かさない）
* ルール：**各乱れシナリオで動かすのは最大 1〜2個**（壊し過ぎ防止）

### 1.2 摂動バジェット（上限固定）

configで固定します（例）：

* 動かす個数：`K_disturb = 1`（clutterだけ2）
* 平行移動：`d_max = 0.6 m`（1回の移動上限）
* 回転：`theta_max = 30 deg`
* 変更は **その部屋（room_id）の中だけ**（別室に飛ばさない）

### 1.3 validity を守るための「弾く条件」

候補レイアウトを作ったら、必ず以下で reject：

* 家具中心が room polygon 外（or OBBが外に出る）
* **入口ドアの keep-out zone** を塞ぐ（例：ドア中心から半径0.5m以内に大物が侵入）
* OBB重なりが閾値超え（例：重なり面積比>0.05 など。厳密でなくても良い）
* 評価器が `validity==0` を返す

### 1.4 “壊し過ぎ”を防ぐ（重要）

乱れは「失敗させる」ためじゃなく「現実にあり得る状態で難しくする」ためなので、

* 目標は **“閾値の少し下”**（ギリギリ悪い）に落とす
  例：`clr_min` を `tau_clr - 0.02m` くらいにする
  → こうすると、後段のリファインで「小さな変更で回復」しやすく、研究として綺麗に出ます。

---

## 2) 3種類の乱れ（Bottleneck / Occlusion / Clutter）の作り方

ここからが実装の核です。全部「候補生成→評価→採用」で作ります。

---

### (A) Bottleneck：`clr_min` を落とす（経路は残す）

**狙い**

* `validity==1`
* `path_exists==True`（start→goal の経路はある）
* `clr_min < tau_clr`（最小クリアランスだけを落とす）
* `R_reach` はできれば維持（落ちても少し）

**作り方（実装方針）**

1. baselineで評価して、A*経路（or 近似直線）を取得
2. 経路上で「狭くなる場所」を推定する

   * 既に evaluator が距離変換を持っているなら、`clr_map` と pathから最狭点が取れます
   * 取れないなら、簡易に「通路中心線上に置く」でも十分効果が出ます
3. `chair` or `table` のうち **小さめ**を1つ選ぶ（大物で塞ぐとvalidity事故）
4. その最狭点近傍に “横から寄せる” 候補を複数作る（±方向に 0.2/0.4/0.6m）
5. 全候補を評価して、制約を満たしつつ `clr_min` が最小になるものを採用

**疑似コード**

```python
best = None
for obj in movable_candidates(layout):
    for (dx, dy, dtheta) in bottleneck_action_set(path_min_point):
        L2 = move(layout, obj, dx, dy, dtheta)
        if not quick_valid(L2): 
            continue
        m = eval(L2)
        if m.validity != 1 or not m.path_exists:
            continue
        # 目標: clr_min を tau_clr より少し下へ
        if m.clr_min < tau_clr and m.clr_min > tau_clr - 0.08:
            return L2
        best = argmin(best, key=m.clr_min)
return best
```

---

### (B) Occlusion：入口からの観測（`C_vis_start` / `OOE`）を落とす（到達性は維持）

**狙い**

* `validity==1`
* `R_reach` と `clr_min` は維持（少なくとも閾値以上）
* `C_vis_start` を落とす、または `OOE_primary` を落とす
* 経路上には置かない（わざと動線を壊さない）

**作り方（実装方針）**

1. entry点 `S`（startを0.4m押し込んだ点）を evaluator と同じルールで得る
2. ROIを決める（最短は **ベッドサイドゴール周辺**）
3. 「S→ROI の見通し線」を遮る位置を探す

   * 簡易：S と goal を結ぶ線分の周辺帯（幅0.5m）に “遮蔽物を置く”
4. 置く家具は **背の高いもの（storage系）優先**（OOEの“家具見え率”が落ちやすい）
5. 候補を評価し、`C_vis_start` や `OOE` が落ちるが `R_reach/clr_min` は維持、のものを採用

**疑似コード**

```python
target = "OOE_primary"  # or C_vis_start
best = None
for obj in tall_movable(layout):
    for pose in sample_near_segment(S, goal, band_width=0.5):
        L2 = set_pose(layout, obj, pose)
        if not quick_valid(L2):
            continue
        m = eval(L2)
        if m.validity != 1:
            continue
        if m.R_reach < tau_R or m.clr_min < tau_clr:
            continue
        # 入口観測だけ落ちたものを優先
        best = argmin(best, key=m[target])
return best
```

---

### (C) Clutter：散らかり（R_reach 低下 or Adopt_entry だけ落とす）

**狙いを2パターンに分けると綺麗**

* **C1（強い乱れ）**：`R_reach < tau_R`（到達性そのものが落ちる）
* **C2（入口だけ悪い）**：`Adopt_core=1` だが `Adopt_entry=0`（入口観測を入れる価値を示せる）

**作り方（実装方針）**

* **C1**：入口～室内の連結を阻害するように、椅子/テーブルを「通路の要所」に2つ置く

  * ただしドア開口を完全には塞がない（keep-out）
* **C2**：入口近傍に椅子を寄せて、`C_vis_start`/`OOE` を落とす

  * ただし経路と `clr_min` は保つ（Occlusionと似ているが、より“入口周辺限定”）

**実装のコツ**

* clutterは **2手**（2つ動かす）にして、評価器で「ちょうど悪い」組み合わせを探すと成功率が上がります
* 2手探索は全探索すると増えるので、

  * 1手目で上位N候補（例20）を残し
  * その上で2手目を試す
    という beam 的な生成にします（生成器にもbeamを使うのが安定）

---

## 3) 実装方針（ファイル構成・I/O・ログ）—これで迷いません

### 3.1 追加するファイル（最小）

* `experiments/configs/degrade/degrade_v1.json`
* `experiments/src/generate_degraded_cases.py`

### 3.2 degrade_v1.json（生成器の設定）

例：

```json
{
  "seed": 0,
  "types": ["bottleneck", "occlusion", "clutter"],
  "movable_categories": ["chair", "table", "storage", "tv_cabinet"],
  "fixed_categories": ["bed", "toilet", "sink", "floor"],

  "budgets": {
    "K_disturb_default": 1,
    "K_disturb_clutter": 2,
    "d_max_m": 0.6,
    "theta_max_deg": 30
  },

  "constraints": {
    "door_keepout_radius_m": 0.5,
    "max_overlap_ratio": 0.05
  },

  "targets": {
    "bottleneck": {"metric": "clr_min", "want_below": "tau_clr", "epsilon_m": 0.02},
    "occlusion": {"metric": "OOE_primary", "want_below": "tau_ooe", "epsilon": 0.05},
    "clutter": {"mode": "Adopt_core_to_0_or_Adopt_entry_to_0"}
  }
}
```

### 3.3 generate_degraded_cases.py の入出力（重要）

**入力**

* baseline layouts（5件）：`experiments/baselines/layouts/*.json`
* `eval_v1.json`（凍結済み）
* `degrade_v1.json`

**出力（1ケースごと）**

* `layout_<baseid>__deg_<type>.json`
* `disturb_manifest.json`（これが論文での再現性の根拠になる）

  * moved_objects（id, category）
  * delta_pose（dx, dy, dtheta）
  * budgets / seed
  * baseline_metrics / degraded_metrics
  * `Delta_disturb`（後述）
* `debug/`（可能なら）

  * occupancy, path, entry_visibility など

---

## 4) 「レイアウト変更量（Δlayout）が意味なくなる」問題の解き方

これはあなたの懸念どおりで、**“どこからのΔか”を分ける**だけで完全に解決します。

* **外乱量（乱れの強さ）**
  [
  \Delta_{\text{disturb}} = \Delta(\mathcal{L}^{deg}, \mathcal{L}^{0})
  ]
* **修正量（手法のコスト：論文の主役）**
  [
  \Delta_{\text{repair}}^{(m)} = \Delta(\mathcal{L}^{out}_{m}, \mathcal{L}^{in})
  ]
  ここで (\mathcal{L}^{in}) は “各trialの入力”（通常なら (\mathcal{L}^0)、乱れなら (\mathcal{L}^{deg})）。

**実装としては**

* generate_degraded_cases が `Delta_disturb` を保存
* run_trial が method実行後に `Delta_repair` を計算して保存
  にするだけです。

---

## 5) 生成のDoD（成功条件）—研究が崩れない最低ライン

乱れ生成は「作れた」ではなく、**使える乱れが作れた**が重要なので、DoDを固定します。

### 各 deg ケースで必須

* `validity == 1`
* “狙った指標” が baseline より確実に悪化

  * bottleneck：`clr_min` が閾値を割る（ただし経路は残す）
  * occlusion：`C_vis_start` か `OOE_primary` が閾値を割る（到達性は維持）
  * clutter：`R_reach` が落ちる **or** `Adopt_entry` だけ落ちる

### 全体（20ケース）での目標

* `Adopt_core=0` が **8ケース以上**
* `Adopt_core=1` だが `Adopt_entry=0` が **数ケース**
  → 「入口観測を入れる価値」が実験で出ます

---

## 6) 1番重要な設計判断：乱れ生成は「手法に依存させない」

乱れは、heuristic/proposed の出力を使って作ると不公平になるので、生成器は

* baseline layout + eval_v1
* degrade_v1 の固定ルール

だけで作り、**一度作ったら JSON を凍結**してください（毎回作り直さない）。

---

## 7) まとめ：おすすめの実装プラン（最短）

1. `eval_v1.json` を凍結（Adopt_core/entryを両方出せる設計がベスト）
2. `degrade_v1.json` を作る（動かす物、上限、ターゲット）
3. `generate_degraded_cases.py` を実装

   * 候補生成→評価→採用
   * validity制約＋“壊し過ぎ防止（閾値の少し下）”
4. 5→20ケースを生成して JSON を凍結
5. `run_trial.py` で 20×4条件を回す

---

必要なら次に、あなたの `layout_generated.json` のスキーマ（object_name/room_id/X/Y/Length/Width/rotationZ）に合わせて、
**「候補配置の作り方（sample_near_segment / keepout / inside_polygon 判定）」をそのままコード化できるレベル**で擬似コード（関数設計＋引数＋戻り値）を出します。
