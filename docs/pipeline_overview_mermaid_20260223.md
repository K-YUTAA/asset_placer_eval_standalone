# End-to-End Pipeline Overview (Mermaid)

最終更新: 2026-02-23  
目的: Step2単体ではなく、画像入力から評価可視化までの全体フローを mermaid で把握する。

---

## 1. 全体実行フロー（ケース単位）

```mermaid
flowchart TB
    U[User / Config JSON] --> R[run_pipeline_from_json.py]

    subgraph LOOP[Case loop]
        R --> G[generate_layout_json.py]
        G --> E[eval_metrics.py]
        E --> P[plot_layout_json.py]
    end

    P --> O[outputs per case<br/>layout_generated.json<br/>metrics.json<br/>plot_with_bg.png]
    R --> BM[batch_manifest.json]
```

---

## 2. `generate_layout_json.py` の内部フロー

```mermaid
flowchart TB
    A1[input image]
    A2[dimensions_*.txt]
    A3[prompt_1]
    A4[prompt_2]
    A5[pipeline config]

    A1 --> B0
    A2 --> B0
    A3 --> B0
    A4 --> B0
    A5 --> B0[prepare runtime args/env]

    B0 --> B1[Step1: LLM理解<br/>OpenAI or Gemini]
    B1 --> S1RAW[step1_output_raw.txt]
    B1 --> S1PARSED[step1_output_parsed.json]

    B0 --> C0{enable_gemini_spatial?}
    C0 -->|No| C9[skip Gemini bridge]
    C0 -->|Yes| C1[Gemini furniture bbox]
    C0 -->|Yes| C2[Gemini room_inner_frame bbox]
    C0 -->|Yes| C3[Gemini openings bbox]

    C1 --> GS[gemini_spatial_output.json]
    C2 --> GR[gemini_room_inner_frame_output.json]
    C3 --> GO[gemini_openings_output.json]

    B0 --> D0{step2_mode}
    D0 -->|rule| D1[Step2 rule transform align schema]
    D0 -->|llm| D2[Step2 llm prompt2 generation]

    S1PARSED --> D1
    GS --> D1
    GR --> D1
    GO --> D1

    S1PARSED --> D2
    A4 --> D2

    D1 --> LAYOUT[layout_generated.json]
    D2 --> LAYOUT
    LAYOUT --> MANI[generation_manifest.json]
```

---

## 3. Step2(rule) の中身（全体内での位置づけ）

```mermaid
flowchart LR
    I1[Step1 parsed]
    I2[Gemini furniture]
    I3[Gemini inner frames]
    I4[Gemini openings]
    I5[inner boundary hint]

    I1 --> T1[座標統一<br/>norm->world->local]
    I2 --> T1
    I3 --> T1
    I4 --> T1
    I5 --> T1

    T1 --> T2[整合処理<br/>room構築 openingマッチ 壁再投影]
    T2 --> T3[スキーマ整形<br/>rooms windows area_objects_list]
    T3 --> O1[layout_generated.json]
```

---

## 4. 評価・可視化フロー

```mermaid
flowchart TB
    L[layout_generated.json] --> M[eval_metrics.py]
    M --> MJ[metrics.json]
    M --> DBG[debug/*]

    L --> P[plot_layout_json.py]
    MJ --> P
    DBG --> P
    BG[input image] --> P
    IF[gemini_room_inner_frame_output.json] --> P

    P --> PNG[plot_with_bg.png]
```

---

## 5. 生成物依存関係（Artifact Graph）

```mermaid
flowchart LR
    IMG[input image] --> STEP1
    DIM[dimensions txt] --> STEP1
    P1[prompt_1] --> STEP1
    STEP1[step1_output_parsed.json] --> STEP2

    IMG --> GEMF[gemini_spatial_output.json]
    IMG --> GEMR[gemini_room_inner_frame_output.json]
    IMG --> GEMO[gemini_openings_output.json]

    GEMF --> STEP2[step2 rule_or_llm]
    GEMR --> STEP2
    GEMO --> STEP2
    P2[prompt_2 llm mode only] --> STEP2

    STEP2 --> LAYOUT[layout_generated.json]
    LAYOUT --> MET[metrics.json]
    LAYOUT --> PLOT[plot_with_bg.png]
    MET --> PLOT
```

---

## 6. 実運用での読み方

- 失敗切り分けは `2 -> 3 -> 4` の順に見る。
- 幾何ズレは `Gemini outputs` と `Step2(rule)` の間で発生しやすい。
- 見た目ズレは `plot_layout_json.py` 側（背景貼り合わせ・inner frame適用）も確認する。
