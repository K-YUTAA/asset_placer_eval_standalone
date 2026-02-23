# 実装報告書（2026-02-19, gemini-spatial-20260219）

## 目的
家具配置精度を上げるため、既存の `prompt1 -> prompt2` フローに対して、Gemini Spatial Understanding を前処理として導入し、特に以下を改善することを目的に実装・検証を実施。

1. 家具の中心座標・サイズ（bbox）精度
2. 部屋の内枠（壁内側）検出
3. 開口部（door / window）検出の安定化
4. 可視化とUSD出力の整合性

## 本日実施した主な変更

### 1. Gemini Spatial Understanding の分割実行方式を確立
1. 家具検出
2. 部屋内枠検出
3. 開口部検出

単一プロンプトで同時検出も比較したが、精度低下が確認されたため、3分割方式を正式採用。

### 2. 家具検出プロンプトの強化
家具は「文字ではなく、家具外形線をタイトに囲う」方針に修正。

- 使用方針
  - ラベル文字領域そのものを囲わない
  - 家具の実体輪郭のみを bbox 化
  - 壁・窓・扉など建築線を除外
- 温度
  - `0.60` をベースに比較（`0.55`, `0.65` も試行）
- モデル
  - `gemini-3-flash-preview` で精度向上を確認

### 3. 開口部検出ロジックの修正（スライディングドア問題）
「引き込み部まで door と誤認識」する問題に対して、開口部（壁が切れている部分）のみを採用する制約を強化。

- `Door`/`Sliding Door` の文言は door として統一解釈
- ただし geometry は「開口幅のみ」を優先
- Storage 扉など非対象扉の混入を抑制

### 4. 部屋内枠（壁内側）検出の反映
Gemini が検出した `room_inner_frame` 系情報を使い、内側基準での部屋形状推定を強化。

- サブルーム内枠は比較的安定
- メインルームは「最大内枠矩形」優先の制約を追加

### 5. Step2 のルールベース化
Step2 の座標決定を LLM依存からルールベースへ移行。

- 家具・部屋は Gemini の bbox を一次ソースとして強制採用
- 窓・ドアのみ追加解釈（開口としての変換）
- 曖昧ケースのみ将来的に LLM fallback 可能な設計方針を整理

### 6. 機能的向き（rotationZ）と bbox 解釈の分離
重要仕様として確定。

- bbox は「平面上の大きさ（world-axis寸法）」として扱う
- `rotationZ` は「機能的正面（アクセス方向）」として扱う
- bbox 自体を回転させる処理は廃止

### 7. 可視化・評価の更新
`plot_with_bg` を含む各可視化で、以下を確認。

- 家具 bbox は概ね高精度
- 機能的向きも改善
- 開口部の door 中心ずれ問題は最終的に改善

### 8. USD / USDA 出力改善
`experiments/src/layout_json_to_usda_bbox.py` を更新。

- `defaultPrim = "LayoutBBox"` を明示
- `size_mode=world` で geometry 回転を抑止
- `functionalRotationZ` を userProperties に保持
- 外壁/内壁 bbox 生成オプションを追加・運用

### 9. 3D地面への評価画像オーバーレイ
要望に対応し、床への画像貼り付けを追加。

- 追加オプション
  - `--overlay_images`
  - `--overlay_z_base`
  - `--overlay_z_step`
- 画像を出力ディレクトリへコピーし、USD Material (`UsdPreviewSurface + UsdUVTexture`) で床面メッシュへバインド
- `plot_with_bg_innerframe` の余白影響により位置ずれが発生したため、以下を実施
  - 単純トリム版
  - 床色領域の最大連結成分を使う floor-crop 版

## 主要な変更ファイル

- `experiments/src/generate_layout_json.py`
- `experiments/src/step2_rule_based.py`
- `experiments/src/layout_tools.py`
- `experiments/src/plot_layout_json.py`
- `experiments/src/run_generate_and_eval.py`
- `experiments/src/layout_json_to_usda_bbox.py`
- `docs/implementation_report_gemini_step1_bbox_decouple_20260219.md`
- `docs/implementation_report_gemini_spatial_20260219_addendum.md`

## 本日生成・確認した代表出力

- 代表run
  - `experiments/runs/e2e_gemini3_openings_case2_rule_step1gemini_bboxfix2_doorfix/`
- 3D出力（bbox）
  - `layout_bbox_worldaxis.usda`
  - `layout_bbox_worldaxis.usdz`
  - `layout_bbox_worldaxis_walls.usda`
  - `layout_bbox_worldaxis_walls_with_inner.usda`
- オーバーレイ出力
  - `layout_bbox_worldaxis_overlay_bgonly.usda`
  - `layout_bbox_worldaxis_overlay_bgtrim.usda`
  - `layout_bbox_worldaxis_overlay_floorcrop.usda`

## 現在の到達点

1. 家具 bbox 精度は実用域まで改善
2. 開口部（特にスライディングドア）の誤検出は大幅改善
3. 機能的向きと bbox の分離仕様を確定
4. USD出力と可視化パスは運用可能

## 残課題

1. `plot_with_bg` 画像をそのまま床貼りした場合の「描画余白・軸情報」起因の幾何ずれ
2. 内枠へ完全一致させるには、描画画像ではなく `room_inner_frame` の px bbox / world bbox を直接使った UV マッピングが必要

## 運用ルール（本日合意）

1. APIコストがかかる再実行は明示指示時のみ実施
2. 長時間処理は TTY でストリーミングログ確認
3. 完了判定はログだけでなく生成ファイル存在確認まで行う

