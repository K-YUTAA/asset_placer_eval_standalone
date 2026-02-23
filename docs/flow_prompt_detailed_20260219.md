# フロー詳細とプロンプト詳細（Gemini Spatial 連携）

- 作成日: 2026-02-19
- 対象: `asset_placer_eval_standalone`
- 目的: 各ステップで何をしているか / プロンプトに何を書いているかを明確化

## 1. 全体フロー

実行入口は `experiments/src/run_generate_and_eval.py`。

処理順:
1. `generate_layout_json.py` で JSON 生成（Step1 -> Gemini群 -> Step2）
2. `eval_metrics.py` で評価
3. 必要時のみ `plot_layout_json.py` で背景付き可視化

中核は `generate_layout_json.py` の以下:
1. Step1（GPT）: 画像 + `prompt1` + `LAYOUT_HINTS` から構造化中間JSONを作る
2. Gemini家具検出: inventory 条件付きで家具 bbox を抽出
3. Gemini内枠検出: `room_inner_frame` / `subroom_inner_frame` を抽出
4. Gemini開口部検出（オプション）: `door/sliding_door/window` を抽出
5. Step2（GPT）: Step1 + Gemini出力 + `prompt2` + `LAYOUT_HINTS` を統合して最終JSONを作る

---

## 2. Step1（GPT画像理解）

使用ファイル:
- `prompts/prompt_1_universal_v4_posfix2_gemini_bridge_v1.txt`

### 2.1 Step1でやっていること

- 画像の平面図を解析して、以下を1つのJSONに出す。
  - 面積/外形
  - 部屋ポリゴン
  - 開口部（door/window）
  - 家具・設備オブジェクト（中心、サイズ、bbox、部屋、front_hint）

### 2.2 prompt1に書いている主な指示

- 座標系定義（Isaac Sim, Z-Up）
- 原点は「歩行可能内側領域の左下」
- `LAYOUT_HINTS` は authoritative（特に `SCALE_WIDTH_M` / `SCALE_HEIGHT_M`）
- テキスト文字ではなく、線・フットプリント重視
- object座標は footprint の CENTER
- `front_hint` は `{0,90,180,270}` or `null`
- 出力は「JSON object only（説明文禁止）」
- 必須スキーマを厳密指定

### 2.3 Step1出力の役割

- inventory（カテゴリと個数）の基準
- 部屋トポロジー、開口部、大枠構造の基準
- Step2 での orientation (`front_hint`) の基準

---

## 3. Gemini家具検出（中間ステップ）

実装:
- `experiments/src/spatial_understanding_google.py`
- `generate_layout_json.py` からサブプロセス実行

### 3.1 何をしているか

- Step1 の object count から inventory 文を自動生成（例: `bed x1, chair x1, ...`）
- 画像に対して Gemini を呼び、家具 bbox（必要に応じて mask）を抽出
- `furniture_objects`（中心、サイズ、正規化/px座標）を出力

### 3.2 家具検出プロンプト（boxes時）

`spatial_understanding_google.py` の `_get_2d_prompt(...)` が生成。内容は以下の意図:

- 平面図内の「ラベル付き家具/設備の輪郭線」を tight bbox で囲う
- ラベル文字（"Sink", "Storage" など）自体は囲わない
- テキストは対象特定のためにだけ使い、bboxは実体輪郭を囲う
- 壁/窓/ドアなど建築線は除外
- `box_2d=[ymin,xmin,ymax,xmax]` を 0-1000 正規化整数で返す

### 3.3 生成JSONの使い方

- Step2に `GEMINI_SPATIAL_UNDERSTANDING_JSON` として注入
- Step2側で `center_norm/size_norm` を world meter に変換して利用

---

## 4. Gemini内枠検出（中間ステップ）

### 4.1 何をしているか

- メイン室内枠とサブルーム内枠を抽出
- 目的は「外壁厚を含まない、内側境界」の拘束条件を Step2 に渡すこと

### 4.2 内枠プロンプト（固定）

`generate_layout_json.py` の `DEFAULT_GEMINI_ROOM_INNER_FRAME_PROMPT`:

- `room_inner_frame` / `subroom_inner_frame` を返す
- mask は内側領域を表現
- 家具・テキスト・窓ドア・外側壁厚は除外

### 4.3 後処理のポイント

`spatial_understanding_google.py` 側で内枠カテゴリ判定時:
- mask から単純bboxでなく「内側で最大の軸平行矩形」を採用
- `box_source = mask_largest_inner_rect`

---

## 5. Gemini開口部検出（オプション）

有効化フラグ:
- `--enable_gemini_openings`

### 5.1 何をしているか

- `door/sliding_door/window` を別呼び出しで抽出
- Step2に `GEMINI_OPENINGS_JSON` を注入して開口位置を補強

### 5.2 開口部プロンプト（現行）

`generate_layout_json.py` の `DEFAULT_GEMINI_OPENINGS_PROMPT` は以下を強く指示:

- 対象は建築開口で、ラベルが Door / Sliding Door / Window の文脈
- bbox は「壁がない通行開口（clear opening gap）」のみ
- sliding door は格納ポケット、待避位置、レール延長を含めない
- storage/cabinet/closet 等の家具ドアは除外

### 5.3 後段フィルタ

`generate_layout_json.py` の `_is_opening_category(...)` / `_extract_opening_objects(...)` で:
- 家具ドア系キーワードを deny
- 開口カテゴリのみ残す

---

## 6. Step2（GPT最終統合）

使用ファイル:
- `prompts/prompt_2_universal_v4_posfix2_gemini_bridge_v1.txt`

### 6.1 Step2でやっていること

- Step1の構造情報と Gemini座標情報を統合し、最終 schema に出力
- 位置とサイズは Gemini を優先、inventory と向きは Step1 を優先

### 6.2 prompt2に書いている主な指示

- Gemini正規化座標 -> world meter の明示変換式
- merge policy の優先順位
  1. 部屋形状/トポロジー: Step1 + LAYOUT_HINTS優先
  2. object inventory: Step1 authoritative（欠落禁止）
  3. 位置サイズ: Gemini優先（不足時はStep1 fallback）
  4. 回転: Step1 `front_hint` 優先
- rotationに応じた Length/Width の割当規則
- 最終出力スキーマを固定
- floorは必ず出力、wall objectは出力しない

### 6.3 Step2に実際に差し込んでいる追加ブロック

`generate_layout_json.py` が prompt2 の末尾に追加:
- `STEP1_JSON`
- `GEMINI_SPATIAL_UNDERSTANDING_JSON`（家具）
- `GEMINI_ROOM_INNER_FRAME_JSON`（内枠）
- `GEMINI_OPENINGS_JSON`（開口部, 有効時）
- 各JSONの統合ポリシー文（priority/fallback）

---

## 7. 出力ファイル対応（runディレクトリ）

代表例: `experiments/runs/e2e_gemini3_openings_case2`

- Step1:
  - `step1_output_raw.json`
  - `step1_output_parsed.json`
- Gemini家具:
  - `gemini_spatial_output.json`
  - `gemini_spatial_output_plot.png`
- Gemini内枠:
  - `gemini_room_inner_frame_output.json`
  - `gemini_room_inner_frame_output_plot.png`
- Gemini開口部:
  - `gemini_openings_output.json`
  - `gemini_openings_output_plot.png`
- Step2最終:
  - `layout_generated.json`
  - `generation_manifest.json`
- 評価:
  - `metrics.json`

---

## 8. チューニング時に触るべきポイント

1. 家具輪郭が文字に引っ張られる場合:
- `_get_2d_prompt(...)` の「ignore text glyph」部分を強化

2. 内枠が外壁厚を含む場合:
- 内枠プロンプト + `mask_largest_inner_rect` 適用カテゴリを確認

3. 開口部で収納ドアを拾う場合:
- `DEFAULT_GEMINI_OPENINGS_PROMPT` と `_is_opening_category(...)` deny語彙を強化

4. Step2で反映が弱い場合:
- `GEMINI_*_POLICY` の優先順位文を強める

