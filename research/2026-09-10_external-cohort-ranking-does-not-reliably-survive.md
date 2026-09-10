# 2026-09-10 — External-cohort generalisation: the internal ranking does NOT reliably survive

Field synthesis on whether an internal model ordering transfers to an external cohort,
how big the internal→external drop is, what has been measured to buy external AUC, and
the protocol-vs-population decomposition. Conclusion up front: **the ranking does not
reliably survive — two clean flips and one 10-model benchmark with the internal winner
collapsing.** Our holdout ordering of 251 members is an in-distribution statement; a
member ranked #12 internally could be #4 or #30 on the board. [CLF] = classifier
endpoint, [SEG] = segmentation, [ASSERTED] = claimed without ablation.

## Ranking flips — quoted

- **Klontzas et al. 2024, Eur Radiol** — hip AVN staging on MRI, the nearest MSK-MRI
  analogue [CLF]. Internal (UHH, n=104): InceptionResNetV2 0.997 > InceptionV3 0.993 >
  **VGG-16 0.973 (worst individual)**. External (TUM, 49 patients, different country):
  **VGG-16 the highest individual AUC at 78.9%** — internal worst became external best.
  Their ensemble was best internally (tied 0.997) AND best externally (0.855). Small
  external n, wide CIs — direction of truth, not magnitude.
- **NCCID multi-centre COVID CXR benchmark** (PMC11148230): 10 models × 3 domains
  [CLF]. Train NCCID (20,515 exams, 25 UK centres). Internal (n=824): SSL-AM #1 (0.74),
  XceptionNet #2 (0.75), MAG-SD #3, CORONET #4. National external (n=11,204):
  XceptionNet #1, **CORONET #7** (0.70). International external (COVIDGR Spain, n=852):
  XceptionNet #1, MAG-SD #2, **SSL-AM #4** (0.67), **CORONET last, 10th** (0.60; recall
  halved 0.52→0.26). Internal #1 fell to #4 abroad; internal #4 collapsed to last; the
  whole field compresses to 0.59–0.71 internationally.
- **Roberts et al. 2021, Nat Mach Intell** — the most-cited flip: models trained on
  non-COVID CXR (CheXpert) beat COVID-trained models on external COVID test sets —
  the reverse of the internal ordering. Numbers paywalled, direction multiply
  corroborated [flagged: not verified this session].

## Ranking HELD, for balance

Velarde 2024 mammography replication: internal-best stayed best on both externals
(0.88→0.90/0.84), the other three stayed poor. Klontzas ensemble: rank preserved AND
drop buffered. So flips are real, but every clean one sits on a small or extreme-shift
external set. Clean null for meta-reviews: Yu/Mohajer/Eng 2022 quantifies drops per
algorithm but never asks whether relative order changed — **nobody has measured ranking
stability at field level.**

## The size of the drop — paired internal→external numbers

| study | task / corpus | internal | external | drop |
|---|---|---|---|---|
| **Xie 2025** (eClinicalMedicine) | multi-label knee abnormalities, 13,419 pts, 5 Chinese centres — closest analogue to us | 0.898 primary | 0.852 ext-I, 0.812 ext-II | −0.046 / −0.086 (grows with distance); secondary 0.815→0.744→0.774 |
| **Rizk 2021** | meniscal tear, 11,353 Swiss exams → external MRNet n=120 | 0.84–0.93 | 0.83 | −0.05…−0.10 (crosses vendor + field strength + label source simultaneously) |
| **MRNet** | Stanford → Rijeka, ACL | 0.937 | 0.824 | −0.11 (triple shift) |
| **Pooch 2020** | four CXR corpora, same DenseNet-121 | — | — | cross-dataset −0.04…−0.12; every test set's best model was its own-domain model (8/8 per-label for three of four corpora) |
| **Zech 2018** | CXR, pooled internal → IU | 0.931 | 0.815 | −0.116 |
| **Yu 2022** systematic review (86 algorithms) | field base rate | — | — | **81% of algorithms drop on external validation, 24% substantially** |

Expected drop for a single-corpus knee model: **0.03–0.10 AUC.**

## Measured external gains (internal alongside)

- **External fine-tuning** — the biggest lever, measured twice on knee: MRNet
  0.824→0.911 on Rijeka after training there; Rizk 0.83→0.89 on MRNet. [CLF]
- **Contrastive domain-adaptive transfer on the exact MRNet↔KneeMRI pair**
  (Electronics 2024): 0.845→0.870 (+0.025) and 0.846→0.881 (+0.035) [CLF]. FLAG —
  the harmful-direction instance: plain contrastive learning WITHOUT source filtering
  DECREASED transfer in the KneeMRI→MRNet direction. Identical internal setup,
  opposite external sign.
- **Cross-architecture ensembling** (Klontzas): internal margin +0.024, **external
  margin +0.066 — the ensemble's edge GROWS externally** [CLF, single study, small n].
- **No internal-only lever was found to buy external AUC without an external-side
  component:** Xie's attention-module ablation (0.730–0.966 → 0.687–0.953 without it)
  is internal-test-only; no external delta published. TTA on external cohorts: no
  clean pair found. That silence is itself a finding.

## Protocol vs population decomposition

- **Zech 2018 is the cleanest experiment in the field** [CLF]: CNNs read hospital
  system at 99.95–99.98% — the acquisition confound is directly decodable.
  Prevalence-matched training → external performance consistent (p=0.88); a 10×
  prevalence imbalance introduced BETWEEN training sites IMPROVED internal AUC
  (p<0.001) and that gain **failed to transfer externally** (p<0.001). Internal
  numbers cannot distinguish the two axes; external ones can.
- **Xie 2025**: on the most-external set, age-subgroup AUC gaps exceeded 0.15
  (meniscal tear 0.844 >40y vs 0.572 ≤40y) — the population axis is measurable on the
  external leg.
- **Rizk 2021** as incidental protocol experiment: crossed vendor + field + sequence,
  lost only ~0.05–0.10 on a coarse task.
- Travelling-subject/harmonisation literature (travelling-heads Sci Data 2025; ComBat
  validated on travelling subjects; Neuroharmony; Saponaro 2022 — harmonisation
  UNCOVERS discrimination masked by site effects; M&Ms 4-vendor cardiac challenge)
  isolates the protocol axis by design [ASSERTED for effect sizes]. Silent: **no
  knee-MRI travelling-subject or vendor-crossed study exists at all.**

## Flagged unverifiable (do not quote without the primary)

Roberts 2021 numbers (paywalled); Levy 2021 adversarial+kneeMRI external table;
Ting 2017 external per-dataset AUCs; AlBadawy 2018 Dice pairs; M&Ms cross-vendor
deltas; Yu 2022 primary table (81%/24% read via RSNA News).

Source letter: D277 (asks 1–4, flips, paired-drop table, external-gain interventions,
Zech decomposition, flagged list).
