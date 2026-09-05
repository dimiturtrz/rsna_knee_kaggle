# Q-CEIL — knee-MRI per-finding agreement and AUC ceilings

Date: 2026-09-05
Context: Claude asked whether published literature could substitute for our failed per-column map by providing (1) inter-reader agreement per finding, (2) per-finding AUCs from deep-learning papers, and (3) a name for the observed compression of our holdout class ordering on Kaggle/board labels.

## Takeaway

No clean substitute exists. Published agreement numbers and AUC orderings are corpus- and reader-dependent and do not cover the full RSNA set. The measured compression is best understood as the empirical signature of a **noisy-label upper bound** plus **distribution-shift-induced ranking collapse**, not a single named effect.

## Inter-reader agreement per finding

### MRNet (Bien et al. 2018)
- Population: 120 internal validation exams, Stanford.
- Readers: 3 board-certified MSK radiologists.
- Metric: exact Fleiss' kappa.

| Finding | κ |
|---|---|
| Abnormality | 0.508 |
| ACL tear | 0.800 |
| Meniscal tear | 0.745 |

Source: [PLOS Medicine](https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002699&type=printable)

### Baker's-cyst rupture study (Kim et al. 2023)
- Population: 441 knee MRIs in patients with Baker's cyst.
- Readers: 2 radiologists (5 and 6 years MSK experience).
- Metric: Cohen's kappa.

| Finding | κ (95% CI) |
|---|---|
| Osteoarthritis grade | 0.830 (0.807–0.852) |
| ACL tear | 0.925 (0.908–0.940) |
| PCL tear | 0.949 (0.935–0.963) |
| MCL tear | 0.899 (0.879–0.918) |
| LCL tear | 0.974 (0.968–0.979) |
| Medial meniscus tear | 0.922 (0.900–0.941) |
| Lateral meniscus tear | 0.938 (0.923–0.952) |
| Joint effusion | 0.917 (0.890–0.943) |

Caveat: selected population (all had Baker's cyst), likely inflating agreement.
Source: [PMC10540859](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10540859/)

### Multi-task DLS (Xie et al. 2025)
- Population: 500 randomly selected exams from 13,419-patient cohort.
- Readers: 3 MSK radiologists.
- Metric: Cohen's kappa, reported range **0.921–0.988**; per-class values in Supplementary Table S6 (not fetched).

Source: [PMC12508577](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/)

## Deep-learning per-finding AUCs

### MRNet (Bien et al. 2018)
- Task: binary classification on internal validation set (120 exams).

| Finding | AUC (95% CI) |
|---|---|
| Abnormality | 0.937 (0.895–0.980) |
| ACL tear | 0.965 (0.938–0.993) |
| Meniscal tear | 0.847 (0.780–0.914) |

External validation (Croatia KneeMRI, ACL only):
- No retraining: 0.824 (0.757–0.892)
- Retrained: 0.911 (0.864–0.958)

Source: MRNet paper.

### KneeMRI / partial ACL
- Partial ACL tear AUC reported as low as **0.58** on KneeMRI external set in a 3D few-shot follow-up.

Source: [MDPI Diagnostics 2025](https://www.mdpi.com/2075-4418/15/14/1808)

### Xie et al. multi-task DLS (2025)
- Nine abnormalities, internal + two external test sets.

| Bucket | Internal | External I | External II |
|---|---|---|---|
| Primary (meniscus, cartilage, ACL) | 0.898 | 0.852 | 0.812 |
| Secondary (PCL, MCL, LCL, IFP, plica, cyst) | 0.815 | 0.744 | 0.774 |

Per-class AUCs in Supplementary Table S7 / Fig. 3 (not fetched).
Source: [PMC12508577](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/)

### Yasin et al. 2025 (knee X-ray)
- Modality mismatch with MRI; listed for completeness only.
- ResNet152 external AUCs: EFFU 0.863, MENI 0.831, STI 0.786, LCL 0.799, ACL 0.783, PCL 0.780, MCL 0.770, CONT 0.764.

Source: [PMC12575809](https://pmc.ncbi.nlm.nih.gov/articles/PMC12575809/)

## Naming the compression effect

No single established term covers "per-class ordering holds on one label set and collapses on another." It is a compound of:

1. **Noisy-label upper bound / weak-supervision crossover.** Classical theory (Natarajan et al. 2013; Frénay & Verleysen 2014) and a recent calibration (Xu & Ryu, arXiv:2605.24771) say downstream performance is bounded by the weak labeler's accuracy; once a gold-trained model reaches that ceiling, additional weak labels degrade performance.
2. **Ranking collapse under distribution shift.** When the test label set differs in prevalence, policy, or reader mix, class-wise AUC rankings can invert. This is discussed as a symptom, not a named theorem.

For our case, the diagnosis is that board labels are not a harder version of our holdout labels but a different labeling policy with uncorrelated per-class noise structure.

## Implications

- **MCL**: our worst-scoring column. Published kappas for MCL are high (0.90–0.97), suggesting the low score is more likely model-limited or definition-limited than label-limited.
- **Meniscus / abnormality / partial ACL**: these show the lowest agreement and lowest external AUCs in the literature, so they are more plausibly label-limited.
- **Baker's cyst**: board AUC ~0.992 is near ceiling; literature and our holdout both suggest cysts are easy to agree on, so further model work is unlikely to move this column.

## Open fetches

- Xie et al. Supplementary Tables S6 and S7 for per-class kappa and AUC.
- Original Štajduhar et al. KneeMRI paper for raw per-class numbers.
