# Q-AB — Taxonomy of report-extractor errors against the 58 gold studies

Claude asked: of the cells the extractor gets wrong against the 58 gold studies, what fraction fall into a named, re-extractable construction, and what fraction are irreducible reader disagreement?

## Method

- Gold positives: image-read label `== 1` in `train.csv`.
- Extracted positives: `report_labels_v2.csv` value `> 0.5` (YES grades `0.68 / 0.82 / 0.94`).
- The 58 gold-annotated studies in `train.csv` give **58 × 12 = 696 cells**.
- One study is missing from `report_labels_v2.csv`; its five gold-positive cells are counted as missed/pipeline errors.
- For each of the **145 disagreeing cells**, the report text was read and classified into a named construction.
- Classification was done by keyword/phrase matching followed by manual review of ambiguous cases. Rates are over the 145 disagreements, not over the 696 cells.

## Denominator

| | count |
| :-- | ---: |
| Studies with all 12 gold labels | 58 |
| Cells | **696** |
| Extractor-gold disagreements | **145** |
| Disagreement rate over cells | **20.83%** |

## Taxonomy of the 145 disagreements

| construction | count | fraction of disagreements | re-extractable? |
| :-- | ---: | ---: | :-- |
| **Hedged negation / negative report read as positive** | 75 | 51.7% | **Yes** |
| Finding not mentioned in report | 30 | 20.7% | No |
| Report negative, gold positive | 20 | 13.8% | No |
| Explicit finding, reader/criteria disagreement | 9 | 6.2% | No |
| Severity / quantity threshold | 6 | 4.1% | **Yes** |
| Missing extracted row (pipeline defect) | 5 | 3.4% | Yes, not a parser problem |
| **Total re-extractable** | **81** | **55.9%** | |
| **Total irreducible / pipeline** | **64** | **44.1%** | |

### Re-extractable constructions (81 / 145 = 55.9%)

#### 1. Hedged negation or negative report read as positive — 75

The extractor fires YES on language that is negative, uncertain, or describes an intact structure. This is the dominant error mode.

Verbatim fragments (representative):

- *"ACL is intact. PCL is preserved. The MCL is intact. Medial meniscus is not torn."* — multiple labels denied by gold but called positive by extractor.
- *"No hay alteraciones de señal significativas de la médula ósea. Ligamentos cruzados y colaterales dentro de límites normales."* (Spanish: no significant bone-marrow signal changes; cruciate and collateral ligaments within normal limits) — ligament labels over-called.
- *"The FCL, popliteus and biceps femoris tendons are preserved."* — structure described as preserved, label called positive.
- *"Medial meniscus appears to be normal."* — called positive.
- *"No fracture is seen."* — fracture called positive.

This category alone is **larger than every other category combined**. A parser that respected explicit negation and intact/preserved language would recover most of it.

#### 2. Severity / quantity threshold — 6

The report uses a size or severity qualifier that the extractor ignores.

Verbatim fragments:

- *"Leve derrame articular"* (Spanish: mild joint effusion) — effusion called positive, gold negative.
- *"Minimalni izljev"* (Croatian: minimal effusion) — effusion called positive, gold negative.
- *"Some joint effusion"* — effusion called positive, gold negative.
- *"Transformación quística parcial del tercio proximal del LCA"* (Spanish: partial cystic transformation of proximal ACL) — ACL called positive, gold negative.
- *"Hondromalacija II. stupnja lateralnog platoa tibije"* (Croatian: grade II chondromalacia of lateral tibial plateau) — lateral OA called positive, gold negative.

These are re-extractable by adding severity-aware rules, but the thresholds must come from the literature or a separate annotation set, not from the 58 gold studies.

### Irreducible constructions (59 / 145 = 40.7%)

#### 3. Finding not mentioned in report — 30

The report text contains no term for the finding, yet the gold label is positive. The extractor either correctly says NO or says YES based on unrelated language.

Verbatim fragments:

- Spanish report discussing iliotibial band friction, Hoffa's fat pad impingement, and patellar chondropathy — **synovitis** gold positive, no mention of synovium/inflammation.
- Greek report describing degenerative meniscal tear — **synovitis** gold positive, no synovial language.
- Report listing ACL tear, MCL partial tear, meniscal tears, effusion, bone contusions — **fracture** gold positive, no fracture term.

These cannot be fixed by re-extraction; the information is not in the report.

#### 4. Report negative, gold positive — 20

The report explicitly states the finding is absent or intact, but the radiologist annotated it positive.

Verbatim fragments:

- *"The meniscus is not torn and there is no focal chondrosis or chondral injury."* — **synovitis** gold positive.
- *"ACL normal. MCL normal. Medial meniscus tear. Lateral meniscus normal. OA of medial compartment."* — **PF OA** gold positive.
- *"No significant bone-marrow signal abnormality."* — **contusion** gold positive.

This is genuine report-vs-radiologist disagreement, not extraction error. Re-extraction would make it worse by reinforcing the report's negative statement.

#### 5. Explicit finding, reader/criteria disagreement — 9

The report clearly states the finding, but the gold annotation disagrees.

Verbatim fragments:

- *"Rotura de menisco interno y lateral"* (Spanish: tear of medial and lateral meniscus) — **medial meniscus** gold negative.
- *"Rotura parcial del LCM"* (Spanish: partial tear of MCL) — **MCL** gold negative.
- *"Baker's cyst present"* — **Baker's** gold negative.
- Turkish report describing grade III degeneration and horizontal tear of medial meniscus — **lateral meniscus** gold negative.

These are radiologist criteria disagreements or laterality/compartment mismatches, not extraction defects.

### Pipeline defect (5 / 145 = 3.4%)

#### 6. Missing extracted row — 5

One study (`1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220`) is absent from `report_labels_v2.csv`. Its report describes ACL tear, joint effusion, suprapatellar bursitis with synovium hypertrophy, and bone-marrow signal abnormality. Five gold-positive cells are therefore missed:

- ACL, PF OA, Effusion, Synovitis, Contusion.

This is a pipeline/data-loading defect, not a parser error.

## Per-label breakdown of re-extractable vs irreducible

| label | re-extractable | irreducible / pipeline | total disagreements |
| :-- | ---: | ---: | ---: |
| ACL | 9 | 1 | 10 |
| MCL | 8 | 1 | 9 |
| Medial Meniscus | 5 | 3 | 8 |
| Lateral Meniscus | 10 | 3 | 13 |
| Medial OA | 8 | 1 | 9 |
| Lateral OA | 8 | 1 | 9 |
| PF OA | 9 | 4 | 13 |
| Effusion | 12 | 5 | 17 |
| Synovitis | 6 | 11 | 17 |
| Baker's | 3 | 3 | 6 |
| Contusion | 2 | 21 | 23 |
| Fracture | 1 | 10 | 11 |

The re-extractable fraction varies sharply by label. **Contusion, Synovitis, and Fracture** are dominated by irreducible disagreement or missing information. **ACL, MCL, Lateral Meniscus, Medial OA, Lateral OA, and Effusion** are dominated by re-extractable over-calls.

## Implications for `errors_dense` and the label-quality axis

- **If re-extractable errors are fixed first**, the residual noise rate drops from ~21% of cells to ~9% of cells (irreducible only), and the remaining errors are concentrated in contusion/synovitis/fracture.
- **Pruning** (`errors_dense`) treats all 5,360 flagged cells as equally suspect. A parser fix targeted at the 75 hedged-negation cases would be far more precise and would not require a GPU arm.
- **The cheapest next step** is therefore a rule-based re-extraction pass that handles:
  1. Explicit negation (`no`, `not`, `intact`, `preserved`, `normal`, `within normal limits`).
  2. Severity/quantity thresholds (`small`, `trace`, `mild`, `minimal`, `some`).
- Only after that pass should a second cleanlab/pruning arm be run, because the current flagging mixes re-extractable and irreducible cells.

## Guardrails honored

- The 58 gold studies were used only to **bound and classify** errors, not to tune thresholds.
- All rates are stated over the 145 disagreements or the 696 cells; no rate is over a subset selected to make a construction look larger.
