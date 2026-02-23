# Full Session Report (2026-02-20)

## 1. 本日の目的

- 家具配置精度の向上
- Step1/Step2 の間に Gemini Spatial Understanding（前処理）を導入して品質改善
- 最終的に安定運用できるフローへ整理（再現性・設定管理・可視化）

## 2. 実施内容（時系列）

1. 既存フロー整理と運用前提の確認
- `prompt*.txt` の配置整理（`inputs_isaac` → `prompts`）
- `entry_observability` は常時有効化
- `Zone.Identifier` 系ファイルは除去方針で統一
- JSON生成と実験ループを分離する運用を維持

2. Gemini Spatial Understanding の単体導入
- `spatial-understanding` オリジナル実装を参照して `experiments/src/spatial_understanding_google.py` を再実装
- マスク中心から 2D bbox 中心へ方針変更（精度改善）
- 家具抽出プロンプトを「文字ではなく家具外形を囲う」内容に調整
- 温度を比較（0.55 / 0.60 / 0.65）し、0.60 を採用
- `gemini-3-flash-preview` を検証し、家具bbox精度が大きく改善

3. 中間抽出（家具・内枠・開口部）の品質改善
- 家具検出・内枠検出・開口部検出を分割3ステップで実行（単発統合より安定）
- 開口部は「壁の欠損部のみ」を検出する制約を強化
- 収納扉など家具ドアを除外
- Sliding door の収納部まで拾う誤検出を抑制（開口部中心に寄せる）

4. Step2 のルールベース化
- Step2での幾何復元をルールベースへ移行
- 家具/部屋は Gemini bbox を優先して強制利用
- 窓/ドアのみ専用変換ロジックで整合
- 曖昧ケースだけ LLM フォールバック可能な設計を検討・整理
- Search prompt は Step1 生成へ寄せ、Step2（ルール）に受け渡す構成へ変更

5. 向き・座標・可視化の整合修正
- `rotationZ` と bbox 形状の責務を分離
- bboxは上面サイズ、`rotationZ` は機能的正面として扱う方針に修正
- plot系で bbox が回転して見える不整合を修正
- 最終 `plot_with_bg` の背景合わせ（inner frame基準）を改善

6. ルーム内枠（room_inner_frame/subroom_inner_frame）強化
- `dimensions.txt` から部屋数・タイプを取り込み、Gemini inner-frameプロンプトへ制約注入
- main room を「内壁線に沿う最大矩形」として抽出する制約を強化
- subroom を可視化しやすい描画へ変更（中心点＋枠）
- subroom を main room 側へスナップする処理を追加
- 家具が辺間に存在する場合はスナップ抑制
- スナップ後にサブルームドア中心を最寄り壁へ再投影
- `snap_threshold` を 0.8 に調整

7. 特殊ケース対応（walk-in closet / toilet）
- walk-in closet の「部屋/家具」二重認識問題を解消方向へ調整
- toilet は「部屋名でもあり設備でもある」例外を許容
- closet に吸われて sink/storage が崩れる問題を緩和

8. 3D出力（USD BBox）と可視化拡張
- JSON から USD BBox を生成
- world座標基準での配置整合を修正
- 外壁に加えて内壁も生成
- 2D評価画像を3D地面に重ねる可視化を試行
- 余白影響を減らすため背景クロップを改善

9. 安定版運用とドキュメント化
- 安定版としてブランチ運用・コミット・プッシュを実施
- 実装報告書/方針書を複数作成し、履歴を文書化
- `AGENTS.md` に長時間実行時のTTY監視・生成物確認・再実行制約を追記

10. 最新設計向けコード整理（本ターン）
- `experiments/src/run_pipeline_from_json.py` を運用向けに整理
- JSONから環境変数注入（`defaults.env` / `cases[].env`）を追加
- generate/eval/plot 各ステージでログ＋生成物存在確認を追加
- `eval_args.enabled` で評価ON/OFFをJSON制御
- 最新一括設定 `experiments/configs/pipeline/latest_design_v2_gpt_high.json` を追加
- `README.md` をJSON駆動実行前提に更新

## 3. 今回達成できたこと

- Gemini前処理（家具・内枠・開口部）を実用レベルで統合
- 家具bboxの精度向上（特に文字囲い誤検出の削減）
- 開口部検出の意味づけ改善（開口部のみ）
- Step2をルールベース化し、再現性・コスト制御を改善
- 内枠ベース可視化と最終配置の整合性を改善
- JSON設定で再実行しやすい運用基盤を整備

## 4. まだ改善余地がある点

- 一部ケースで main room inner frame の取り切り不足が再発し得る
- ドア向き/開口長の安定性はケース依存で追加改善余地あり
- walk-in closet のような複合意味ラベルは、さらに汎化ルール強化の余地あり

## 5. 現在の運用推奨

- 3分割Gemini抽出（家具・内枠・開口部）を継続
- Step2はルールベースを標準
- 実行は `run_pipeline_from_json.py` + 設定JSONで統一
- API再実行は明示指示時のみ
