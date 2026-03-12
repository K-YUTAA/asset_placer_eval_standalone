# Research flow with GPT and Gemini focus points

Updated: 2026-02-23

Goal:
1. Keep model step names explicit: GPT and Gemini
2. Show what data each step consumes
3. Show what each step focuses on during processing
4. Show what data each step outputs

---

## 1. Step focus matrix

| Step | Model or method | Main input data | Processing focus | Main output data |
|---|---|---|---|---|
| Step1 | GPT semantic parser | floor plan image, dimensions | room and object semantics, category intent, functional orientation hints, opening intent | semantic scene hypothesis |
| Step2A | Gemini furniture geometry | image, dimensions, Step1 inventory guidance | fixture outline contour, text exclusion, tight furniture bounding geometry | furniture geometry set |
| Step2B | Gemini room inner frame | image, dimensions, Step1 anchor guidance | inside wall boundary, maximal inner rectangles for main and sub rooms | room inner boundary set |
| Step2C | Gemini openings geometry | image, dimensions, Step1 opening intent reference | opening gap detection, sliding door gap span, exclusion of storage pocket area | opening geometry set |
| Step3 | Rule based fusion | Step1 semantic hypothesis, Gemini geometry outputs | coordinate unification, wall alignment, consistency resolution, final schema assembly | structured layout representation |

---

## 2. Data flow with explicit Step1 Step2 Step3

```mermaid
flowchart TB
    classDef data fill:#e8f4ff,stroke:#2b6cb0,color:#1a365d,stroke-width:1px;
    classDef proc fill:#fff5e6,stroke:#b7791f,color:#744210,stroke-width:1px;

    subgraph LEGEND[legend]
        L1[data]
        L2[process]
    end

    I1[floor plan image]
    I2[dimensions]

    subgraph S1[Step1 GPT semantic parsing]
        direction TB
        P1[semantic interpretation]
        O1[semantic scene hypothesis]
        P1 --> O1
    end

    subgraph S2[Step2 Gemini geometry extraction]
        direction TB
        P2A[furniture geometry branch]
        O2A[furniture geometry set]
        P2B[room inner frame branch]
        O2B[room inner boundary set]
        P2C[openings geometry branch]
        O2C[opening geometry set]

        P2A --> O2A
        P2B --> O2B
        P2C --> O2C
    end

    subgraph S3[Step3 rule based fusion]
        direction TB
        P3[coordinate normalization and consistency fusion]
        O3[structured layout representation]
        P3 --> O3
    end

    I1 --> P1
    I2 --> P1

    O1 -- inventory guidance --> P2A
    I1 --> P2A
    I2 --> P2A

    I1 --> P2B
    I2 --> P2B
    O1 -- anchor guidance --> P2B

    O1 -- opening intent reference --> P2C
    I1 --> P2C
    I2 --> P2C

    O1 --> P3
    O2A --> P3
    O2B --> P3
    O2C --> P3

    class L1,I1,I2,O1,O2A,O2B,O2C,O3 data;
    class L2,P1,P2A,P2B,P2C,P3 proc;
```

---

## 3. Focus points by step

| Step | Focus point |
|---|---|
| Step1 GPT | semantic meaning of rooms and objects, functional orientation hints, opening intent |
| Step2A Gemini furniture | contour of real fixtures, exclusion of text glyph regions |
| Step2B Gemini inner frame | inside wall boundary and maximal room coverage |
| Step2C Gemini openings | opening gap span, exclusion of sliding panel storage region |
| Step3 rule fusion | normalize coordinates and enforce wall consistent placement |

---

## 4. Practical reading order

1. Follow `Step1 GPT` to understand semantic control signals.
2. Check three Gemini branches separately:
- furniture geometry
- room inner boundary
- openings geometry
3. Confirm how `Step3 rule based fusion` resolves all branches into one consistent spatial representation.
