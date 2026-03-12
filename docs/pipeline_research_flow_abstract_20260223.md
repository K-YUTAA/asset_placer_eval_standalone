# Research oriented pipeline flow abstract version

Updated: 2026-02-23

This document is an abstract research view of the pipeline.  
It focuses on:
1. what kind of data is input
2. how data is transformed
3. what kind of outputs are produced

No implementation file names or concrete json keys are used.

---

## 1. Data processing pipeline abstract

```mermaid
flowchart TB
    classDef data fill:#e8f4ff,stroke:#2b6cb0,color:#1a365d,stroke-width:1px;
    classDef proc fill:#fff5e6,stroke:#b7791f,color:#744210,stroke-width:1px;
    classDef note fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-width:1px;

    subgraph LEGEND[legend]
        L1[data]
        L2[process]
        L3[explanation]
    end

    subgraph INPUT[input layer]
        I1[floor plan image data]
        I2[spatial constraints and scale hints]
        I3[task policy and semantic priors]
        I4[inference settings]
    end

    subgraph CORE[processing layer]
        P1[semantic interpretation of rooms and objects]
        P2[geometric extraction of object boundaries]
        P3[geometric extraction of room inner boundaries]
        P4[geometric extraction of architectural openings]
        P5[cross source consistency fusion]
        P6[coordinate normalization and wall alignment]
        P7[scene representation assembly]
    end

    subgraph OUTPUT[output layer]
        O1[structured scene data]
        O2[placement ready object set]
        O3[evaluation indicators]
        O4[human interpretable visualization]
    end

    I1 --> P1
    I2 --> P1
    I3 --> P1
    I4 --> P1

    I1 --> P2
    I1 --> P3
    I1 --> P4
    I2 --> P3
    I2 --> P4

    P1 --> P5
    P2 --> P5
    P3 --> P5
    P4 --> P5
    P5 --> P6 --> P7

    P7 --> O1
    P7 --> O2
    O1 --> O3
    O1 --> O4
    O3 --> O4

    N1[semantic and geometric evidence are separated then fused]
    N2[final outputs support both machine use and human inspection]
    N1 -.-> P5
    N2 -.-> O4

    class L1,I1,I2,I3,I4,O1,O2,O3,O4 data;
    class L2,P1,P2,P3,P4,P5,P6,P7 proc;
    class L3,N1,N2 note;
```

---

## 2. Research view data transformation map

```mermaid
flowchart LR
    classDef data fill:#e8f4ff,stroke:#2b6cb0,color:#1a365d,stroke-width:1px;
    classDef proc fill:#fff5e6,stroke:#b7791f,color:#744210,stroke-width:1px;
    classDef note fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-width:1px;

    subgraph LEGEND2[legend]
        LL1[data]
        LL2[process]
        LL3[explanation]
    end

    D1[raw visual evidence]
    T1[semantic abstraction]
    D2[semantic object room hypotheses]
    T2[geometry focused detection]
    D3[geometric primitives boxes and masks]
    T3[constraint aware fusion]
    D4[consistent spatial state]
    T4[normalization and projection]
    D5[layout level representation]
    T5[evaluation and reporting]
    D6[metrics and visual reports]

    D1 --> T1 --> D2
    D1 --> T2 --> D3
    D2 --> T3
    D3 --> T3 --> D4 --> T4 --> D5 --> T5 --> D6

    N3[data moves from ambiguous to constrained and interpretable]
    N4[evaluation closes the research loop]
    N3 -.-> D4
    N4 -.-> T5

    class LL1,D1,D2,D3,D4,D5,D6 data;
    class LL2,T1,T2,T3,T4,T5 proc;
    class LL3,N3,N4 note;
```

---

## 3. Reading guide for first time readers

1. Start from `input layer` and confirm what data types are available.
2. Read `processing layer` as three phases:
- semantic estimation
- geometric extraction
- fusion and normalization
3. Confirm that outputs are dual purpose:
- machine consumable scene structure
- human verifiable visual evidence

