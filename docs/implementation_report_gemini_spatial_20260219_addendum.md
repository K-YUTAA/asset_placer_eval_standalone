# 実装報告書 追補（Gemini Spatial）

- 作成日: 2026-02-19
- 対象リポジトリ: `asset_placer_eval_standalone`
- 対象期間: 前回報告書 `docs/implementation_report_gemini_spatial_20260219.md` 作成以降の追加分

## 1. 追加で実施したこと（要約）

- Gemini 3系で「分割3回（家具/内枠/開口部）」と「単発統合1回」の比較検証を実施。
- 検証結果を踏まえ、運用方針を分割3回に固定。
- 開口部プロンプトを「壁のない開口ギャップのみ」へ厳格化。
- `outer_polygon` を Gemini 内枠（`room_inner_frame`）優先で最終出力へ反映する方針を実装。
- Step1 の `front_hint` 推定ルールを強化（カテゴリ別機能的向きルールを追加）。
- APIコスト運用ルールを `AGENTS.md` に明文化。
- ユーザー提供JSONをそのまま評価・可視化するフローを複数回実施。

## 2. 分割3回 vs 単発統合1回（追加検証）

### 2.1 結果

- 分割3回（基準）:
  - run: `experiments/runs/e2e_gemini3_openings_case2`
  - カテゴリ整合と安定性が最良。
- 単発統合（複数パターン探索）:
  - JSON崩れで失敗するケースが発生。
  - 成功ケースでも誤検出（例: `storage` +1）や `sliding_door` 過大bboxが発生。

### 2.2 代表値

- 分割3回（Geminiトークン合計）: `4398`
- 単発統合1回（代表）: `1728`
- 単発統合は安価だが、精度・安定性が不足。

### 2.3 結論

- 当面の運用は **分割3回を継続**。

## 3. 開口部（door/sliding_door/window）改善の追加

- 開口部プロンプトを「clear opening gap（壁欠損の通行部）のみ」に寄せた。
- `sliding_door` では、パネル収納ポケット/退避領域/レール延長の除外を明記。
- 家具ドア（storage/cabinet/closet）を除外するルールを保持。
- 改善効果:
  - 収納ドア誤検出は抑制。
- 残課題:
  - 一部ケースで `sliding_door` が依然として開口以上に広く出る。

## 4. outer boundary を内側壁ライン優先にする追加実装

変更ファイル:
- `experiments/src/generate_layout_json.py`
- `prompts/prompt_2_universal_v4_posfix2_gemini_bridge_v1.txt`

実装内容:
- Gemini内枠（`room_inner_frame`）から main room の world boundary ヒントを算出。
- Step2 に `GEMINI_MAIN_ROOM_INNER_BOUNDARY_WORLD` を注入。
- `GEMINI_OUTER_BOUNDARY_POLICY` を追加し、`outer_polygon` は内側壁ライン優先で採用する指示を追加。

## 5. front_hint 強化（Step1）

変更ファイル:
- `prompts/prompt_1_universal_v4_posfix2_gemini_bridge_v1.txt`

追加内容:
- `front_hint` の角度定義（0/90/180/270）を明示。
- カテゴリ別の機能的向きルールを追加。
  - `bed`, `chair/sofa`, `tv`, `sink`, `toilet`, `storage/cabinet`, `door/sliding_door`
- 推定手順を追加。
  - 形状・記号優先
  - 文字向きに引っ張られない
  - 不確実なら `null`

観測された変化（Step1比較）:
- 旧 run (`e2e_gemini3_openings_case2`) から、新 run (`e2e_gemini3_openings_case2_front_hint_v2`) の `front_hint` が複数カテゴリで更新されたことを確認。

## 6. 実行運用ルールの明文化

追加ファイル:
- `AGENTS.md`

追記ルール:
- 長時間実行は `TTY` 付きで起動し、ストリーミングログを逐次確認。
- 完了判定は「ログ完了」かつ「期待出力ファイル存在」の両方で行う。
- APIコストが発生する再実行は、ユーザー明示指示がある場合のみ実施。

## 7. ユーザー提供JSONの評価・可視化（追加分）

主に以下ケースを作成:
- `experiments/runs/from_user_json_case`
- `experiments/runs/from_user_json_case2`
- `experiments/runs/from_user_json_case3`
- `experiments/runs/from_user_json_case4`

`from_user_json_case4` で実施:
- 入力JSON保存: `layout_generated.json`
- 評価: `metrics.json`
- 可視化: `plot.png`, `plot_with_bg.png`
- デバッグ可視化: `debug/` 一式

代表メトリクス（case4）:
- `C_vis=0.9370`
- `R_reach=0.9772`
- `clr_min=0.1000`
- `Adopt=0`
- `validity=1`

## 8. 現時点の判断

- パイプライン構成は引き続き **分割3回** が妥当。
- 向き推定は改善傾向あり（Step1ルール強化の効果あり）。
- 次の主要改善対象は `sliding_door` の開口幅過大化抑制。

## 9. 次アクション（提案）

- 開口部後処理に「壁境界クリップ」を追加して `sliding_door` bbox を開口ギャップへ強制縮退。
- Step2で opening 幅は Gemini開口部を優先する明示度をさらに上げる。
- front_hint のカテゴリ別検証をケース分けしてログ化（bed/toilet/storageで誤り傾向を分離）。

