# 実装方針: Step2 ルールベース化（必要時のみLLMフォールバック）

- 作成日: 2026-02-19
- 対象ブランチ: `feature/step2-rule-based`
- 目的: Step2（最終JSON生成）を原則ルールベースで置換し、再現性・精度・コストを改善する

## 1. 背景と狙い

現状の課題:
- Geminiで bbox は高精度に取れているが、Step2（LLM）が再解釈してズレることがある
- 特に outer boundary（内側壁線）と開口部（door/window）の整合で揺れが出る
- APIコストと実行安定性の観点で Step2 LLM の依存を下げたい

狙い:
- 家具・部屋は Gemini 座標を優先して「決定的に」反映
- door/window のみ幾何変換（壁スナップ）
- LLMは原則使わず、曖昧ケースのみ任意で呼ぶ構成にする

## 2. 方針の結論

- Step2は **Rule-based をデフォルト** とする
- LLMは **fallback_mode=off** を基本運用
- どうしても曖昧で解けないケースだけ `fallback_mode=llm` を許可（明示フラグ）

## 3. 入出力設計

入力:
- `step1_output_parsed.json`
- `gemini_spatial_output.json`（家具）
- `gemini_room_inner_frame_output.json`（内枠）
- `gemini_openings_output.json`（開口部）
- `LAYOUT_HINTS`

出力:
- 現行と同一スキーマの `layout_generated.json`

## 4. ルールベース変換の中核ルール

### 4.1 境界（outer/rooms）

1. `outer_polygon`
- `room_inner_frame`（main）の world 変換ポリゴンを最優先採用
- Step1外周と矛盾しても、内側壁線を優先

2. `rooms[].room_polygon`
- Step1 room polygon をベース
- outer_polygon の内側へクリップ（必要時）
- room間重複禁止、共有境界は一致させる

### 4.2 家具（area_objects_list）

1. inventory
- カテゴリと個数は Step1 を正とする

2. 位置・サイズ
- `bed/chair/sink/storage/tv...` は Gemini bbox を world変換して強制採用
- Gemini欠落時のみ Step1値へフォールバック

3. 向き
- `rotationZ` は Step1 `front_hint` を優先
- `front_hint=null` の場合のみ簡易補完（長辺方向・壁向き）

4. search_prompt
- Step1 で生成された `objects[].search_prompt` をそのまま利用
- 未設定時のみカテゴリ別テンプレートを使用

### 4.3 開口部（door/window）

- door/window だけは bbox をそのまま使わない
- 最寄り壁セグメントへ中心を投影（wall snap）
- 幅は壁方向の成分で再計算
- sliding_door は `door` 正規化オプション可
- 収納ポケット側を含む過大幅は、壁欠損区間へクリップ

## 5. あいまいケースのフォールバック設計

## レベルA: ルールで解ける（デフォルト）
- 近傍一致、距離最小、面積/比率閾値、壁スナップ
- すべて deterministic に解決

## レベルB: ルールで曖昧（fallback_mode=rule_strict）
- 不確実対象のみ保守的に採用
- 例: 候補複数なら「未採用」にして Step1へフォールバック

## レベルC: どうしても曖昧（fallback_mode=llm）
- 最小入力で LLM に問い合わせ（対象オブジェクト限定）
- 全体再生成はしない
- 返答は JSON 部分のみ受け取り、対象フィールドだけ上書き

推奨運用:
- まずは `fallback_mode=off` または `rule_strict`
- 検証で必要なときだけ `llm` を有効化

## 6. 実装ステップ

1. `experiments/src/step2_rule_based.py` 新設
- `build_layout_rule_based(...)` を実装

2. `experiments/src/generate_layout_json.py` 更新
- `--step2_mode {llm,rule}` を追加
- `rule` のとき Step2 LLM呼び出しをスキップ

3. 変換関数群を実装
- `bbox_norm_to_world`
- `select_main_room_inner_frame`
- `assign_objects_by_category_and_distance`
- `snap_openings_to_walls`
- `validate_and_fix_layout`

4. fallback制御を追加
- `--step2_fallback_mode {off,rule_strict,llm}`
- `--step2_fallback_llm_model`（必要時のみ）

5. 既存互換を維持
- `--step2_mode llm` を残して比較可能にする

## 7. 検証観点

- outer_polygon が常に内側壁線に一致するか
- 家具中心・サイズが Gemini bbox と一致するか
- door/window が壁上にあり、幅が開口区間と一致するか
- 向き（rotationZ）が Step1 front_hint と整合するか
- 最終メトリクス（C_vis, R_reach, clr_min, Adopt）

## 8. リスクと対策

リスク:
- ルールが厳しすぎて object/drop が発生
- 開口部スナップで room接続の誤割当

対策:
- しきい値を設定ファイル化（ケースごとに調整可能）
- 中間デバッグJSONを必ず保存
- fallbackは対象限定で使い、全体再生成を避ける

## 9. 最終判断

- 本件は Step2 をルールベースに置換する価値が高い
- LLMは「全体生成」ではなく「限定フォールバック」に用途を縮小する
- まずは Rule-only 実装を完成させ、必要なら限定LLMを後付けする

