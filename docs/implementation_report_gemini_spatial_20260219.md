# 実装報告書（Gemini Spatial 連携）

- 作成日: 2026-02-19
- 対象リポジトリ: `asset_placer_eval_standalone`
- 作業ブランチ: `gemini-spatial-20260219`
- 安定スナップショット: commit `21ebbd2`

## 1. 目的

家具配置JSONの生成精度向上のため、既存の `prompt1 -> prompt2` フローの中間に Gemini Spatial Understanding を導入し、家具の中心座標・サイズ・部屋内枠・開口部（door/window）を補助情報として Step2 に注入する。

## 2. 今回実装した内容

- `inputs_isaac` 内の `prompt*.txt` を `prompts/` 配下へ移動。
- `Zone.Identifier` 系ファイルを削除。
- GPT reasoning の実行デフォルトを `medium`（`middle` 指定も `medium` に正規化）へ変更。
- Gemini中間ステップを e2e パイプラインへ統合。
- `spatial-understanding` オリジナル実装を参照し、`experiments/src/spatial_understanding_google.py` を整備。
- Gemini家具検出を `2D bbox` ベースへ調整。
- 家具検出プロンプトを「文字領域ではなく家具輪郭を囲む」方針に修正。
- ラベル言語を `English` に統一。
- 部屋内枠検出を追加し、内枠カテゴリではマスクから「内側で最大の軸平行矩形」を採用。
- Gemini開口部検出（door/sliding_door/window）を追加。
- 開口部プロンプトを「通行可能な開口ギャップのみ」に厳格化。
- `run_generate_and_eval.py` へ Gemini オプションの引き回しを追加。
- 生成マニフェストに Gemini 各中間出力（家具/内枠/開口部）のパスを出力。

## 3. 主要変更ファイル

- `experiments/src/generate_layout_json.py`
- `experiments/src/run_generate_and_eval.py`
- `experiments/src/spatial_understanding_google.py`
- `prompts/prompt_1_universal_v4_posfix2_gemini_bridge_v1.txt`
- `prompts/prompt_2_universal_v4_posfix2_gemini_bridge_v1.txt`
- `docs/implementation_policy_gemini.md`

## 4. 実行・検証結果（代表）

- e2e（Gemini 3 + openings）:
  - run: `experiments/runs/e2e_gemini3_openings_case2`
  - `C_vis=0.8689710610932476`
  - `R_reach=0.9557877813504824`
  - `clr_min=0.3656854249492381`
  - `Adopt=1`
  - `validity=1`
- 開口部単体（厳格化後）:
  - run: `experiments/runs/gemini_openings_gaponly_t06`
  - 検出: `window x2`, `sliding_door x2`, `door x0`
  - 収納ドア誤検出は抑制。
- 内枠単体（最大内接矩形適用）:
  - run: `experiments/runs/gemini_room_inner_frame_t06_rerun_maxrect`
  - `room_inner_frame`, `subroom_inner_frame` の `box_source` が `mask_largest_inner_rect` で出力。

## 5. 現状の問題点

- `door` を厳しく絞ると `door x0` になりやすい。
- `sliding_door` は改善したが、ケースによっては開口以外（格納側）を広めに含む可能性が残る。
- `Gemini 3` は `mask` を PNG data URL ではなく座標配列で返す場合があり、既存の mask decode 前提処理が効かないケースがある。
- 内枠の検出結果は run ごとに揺れがあり、`main room` 側が崩れるケースがある。
- Step2（GPT最終生成）は待ち時間とコストが大きく、再試行コストが高い。

## 6. 今後の対応方針（提案）

- 開口部で `sliding_door` を最終JSON上は `door` に正規化する。
- 開口部抽出に「壁線の切断点ベース」の後処理ルールを追加し、ギャップのみを幾何的に再切り出しする。
- `Gemini 3` のポリゴン mask 形式を `spatial_understanding_google.py` で正式サポートする。
- 内枠は `main/sub` を別プロンプトで分離し、main崩れ時のフォールバックルールを定義する。
- 高コスト区間（Step2）の再実行を避けるため、Gemini単体評価用の固定スクリプトを継続運用する。

## 7. 参照先（今回の成果物）

- 安定ブランチ: `stable/gemini-spatial-20260219`
- PR URL: `https://github.com/K-YUTAA/asset_placer_eval_standalone/pull/new/stable/gemini-spatial-20260219`
- 代表run:
  - `experiments/runs/e2e_gemini3_openings_case2`
  - `experiments/runs/gemini_openings_gaponly_t06`
  - `experiments/runs/gemini_room_inner_frame_t06_rerun_maxrect`


## 8. 追加検証: Gemini「分割3回」 vs 「単発統合1回」

### 8.1 検証目的

- 問い: 家具・内枠・開口部を1回のGeminiプロンプトに統合した場合、分割3回より精度が下がるか。

### 8.2 比較条件

- 同一画像を使用: `inputs_isaac/oizumi-gakuen_C_24_5.19*4.63_v2.png`
- モデル: `gemini-3-flash-preview`
- 温度: `0.6`（一部再試行で `0.0`）
- 比較対象:
  - 分割3回: `e2e_gemini3_openings_case2`（家具/内枠/開口部を別呼び出し）
  - 単発統合: `gemini3_singlepass_*` 系 run

### 8.3 結果サマリ

- 分割3回（基準）:
  - run: `experiments/runs/e2e_gemini3_openings_case2`
  - 検出カテゴリ: 家具 + 内枠 + 開口部が安定
  - 代表内訳: `bed1 chair1 sink1 storage2 toilet1 tv_cabinet1 room_inner_frame1 subroom_inner_frame1 sliding_door2 window2`

- 単発統合（探索）:
  - `experiments/runs/gemini3_singlepass_explore_s1_t06`: JSON崩れで失敗
  - `experiments/runs/gemini3_singlepass_explore_s1_t00`: JSON崩れで失敗
  - `experiments/runs/gemini3_singlepass_explore_s2_t06`: `detection_count_raw=16` だが `invalid_detection_count=11`（有効5件のみ）
  - `experiments/runs/gemini3_singlepass_all_custom2_t06`: 有効14件だが `storage` 過検出（+1）かつ `sliding_door` が過大bbox化

### 8.4 定量比較（代表）

- 分割3回の総トークン（Gemini 3呼び出し合計）: `4398`
- 単発統合の総トークン（1呼び出し）: `1728`
- コストは単発統合が低いが、精度・安定性は分割3回が優位。

### 8.5 結論

- 現時点では **分割3回（家具/内枠/開口部）を継続** が妥当。
- 単発統合は、出力安定性（JSON整形）と開口部・一部カテゴリ精度で劣後する。

