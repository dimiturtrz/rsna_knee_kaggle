# The data-volume thread: power-law pricing of +0.010 and the labelled knee-MRI corpus enumeration

Researcher lane, 2026-09-11. Sources for letter D328 (main's M333 Q1/Q2). All citations verified at
fetch time.

## Q1 — the exponent, the denominator, and the price of +0.010 from 0.944 at N=4,407 studies

### Verified anchors and their denominators

| Source | x-axis | y-axis | Exponent | Usable for a study-denominated corpus? |
|---|---|---|---|---|
| Zhang et al., arXiv:2210.02189 (Radiology:AI 2023, COVID CXR) | **patients** (100–2000 fit, 6000 validated) | AUC | k = −0.21…−0.25 on AUC = a·N^k + b | **YES — the only one** |
| Mahmood et al., arXiv:2207.01725 (CVPR 2022) | images | accuracy/mAP | NOT reported (no θ tables) | no medical task at all |
| Hestness et al., arXiv:1712.00409 | samples | error rate | −0.07…−0.35 | y is error; no medical task |
| Sharma et al., arXiv:2509.12818 (Sept 2025, radiology FMs) | PRETRAINING image-report pairs (30k–3.5M) | AUPRC | k = 0.012 (MI2) / 0.004 (RAD-DINO) | NO — pretraining axis, downstream N fixed |
| ELIXR, arXiv:2308.01317 | images (~2.2k → ~22k) | mean AUC | implied β ≈ 0.02 | images axis, frozen encoder — conservative bound |
| Sellergren et al., Radiology 2022, PMID 35852426 | 45 images (!) | AUC 0.95 external TB | none fitted | marginal response extremely shallow with strong pretraining |

Mahmood's portable content is methodological, not numeric: four regressors (power/arctan/log/root),
and with saturating forms the data-requirement estimate is pessimistic, with power/log optimistic;
single-round extrapolation from small subsets misestimates by 2–10×. No exponents tabulated, no
medical tasks.

Zhang's measured points (HF internal test): 100→0.732, 400→0.786, 800→0.794, 2000→0.808,
**6000→0.818 (measured, not extrapolated)** — 3× bought +0.011 at baseline ≈0.81.

### The arithmetic (pure deficit d(N) = a·N^(−β), anchor d(4407)=0.056, target 0.046)

required multiple k = (0.056/0.046)^(1/β) = 1.2174^(1/β):

| β | multiple | absolute studies |
|---|---|---|
| 0.05 | 51× | ≈226,000 |
| 0.10 | 7.2× | ≈31,500 |
| 0.21–0.25 (Zhang) | 2.7×–2.2× | ≈9,700–11,800 |
| 0.35 (Hestness upper) | 1.8× | ≈7,700 |

With irreducible deficit c (ceiling AUC = 1−c), k = [(0.056−c)/(0.046−c)]^(1/β), reachable at any
multiple iff c < 0.046:

| c | ceiling | β=0.10 | β=0.20 | β=0.25 |
|---|---|---|---|---|
| 0 | 1.000 | 7.2× | 2.7× | 2.2× |
| 0.01 | 0.990 | 11.6× | 3.4× | 2.7× |
| 0.02 | 0.980 | 25.9× | 5.1× | 3.7× |
| 0.04 | 0.960 | ≈18,000× | ≈135× | ≈51× |
| ≥0.046 | ≤0.954 | unreachable at any multiple | | |

### The caveat that dominates

Pure-deficit overstates responsiveness. Zhang's own measured points violate it: β≈0.22 pure-deficit
predicts 2000→6000 should give 0.849; measured 0.818. A two-parameter refit of his Table 3 (100/2000/
6000) at β=0.23 gives ceiling b ≈ 0.885 — c ≈ 0.115 for HIS task (our arithmetic on his numbers,
flagged as derived). Mahmood's thesis: the ceiling parameter, not the exponent, decides whether the
target exists, and the ceiling is badly identified from small-N fits. **No published medical study
measures the ceiling at baseline AUC ≥0.90** (Zhang tops at 0.82; ELIXR's 0.89→0.90 is images-axis).
ELIXR's implied β≈0.02 would make +0.010 cost ~13,000× — the opposite bound, from a frozen
language-aligned encoder.

### Saturation

No universal wall exists in the published record. Sharma: log-linear AUPRC still rising at 3.5M
pretraining images; explicit benchmark saturation on small public holdouts (VinDR) — the ceiling there
is benchmark/label noise, not corpus size. Zhang: gains "rapidly become marginal" past ~800 patients —
fitted-ceiling flattening, not a wall.

### Synthesis with the audit (Q3)

If the ceiling c is set by residual label error (Northcutt: ~6% flips close rankings), then more data
under the SAME label process buys nothing past the ceiling — volume pays only if label quality scales
with it. The audit's residual bound and the power law's ceiling term are the same unknown wearing two
names.

## Q2 — labelled knee-MRI corpora at scale (all verified against source pages)

### Corpora with diagnostic labels

| Corpus | n | Columns of our 12 | Licence verdict |
|---|---|---|---|
| OAI | ~4,796 subjects | **MedOA/LatOA/PFOA clean + ~3 partial (MOAKS: effusion-synovitis pooled, meniscal damage, BML)** | ❌ DUA: research/education only, §4(c) no redistribution — a Kaggle kernel is neither |
| CoPAS (Qiu et al. Nat Commun 2024, PMC11368947) | 1,748 | ~6–7 (closest schema in print: ACL, MCL, LCL, meniscus, cartilage, contusion, effusion, popliteal cyst ≈ Baker's) | ❌ private, MTA, non-commercial — data availability verbatim: "restricted access… formal material transfer agreement" |
| MOST | ~3,026 | same OA profile as OAI (MOAKS) | ❌ NIA biobank, research-gated |
| fastMRI+ | 1,172 | 4 clean (ACL, MCL weak-125-pos, Effusion) + Contusion/Fracture pooled; OA proxies only; NO synovitis | ⚠️ annotations MIT but images NYU DUA — blocked for shipping |
| MRNet | 1,370 | 1.5 (ACL + meniscus pooled over Med/Lat) | ❌ Stanford RUA non-commercial, no derivatives |
| Rijeka KneeMRI | 917 | 1 (ACL) | ❌ CC BY-NC-ND 4.0 |
| SKM-TEA | 155 | up to 4–5 (unique: med/lat meniscus separable via tissue seg) | ❌ "may not create derivative works" — strongest anti-shipping clause |
| AH-Knee / GreenRFM | 3,662 | 13 LLM-extracted labels | ❌ private, unreleased |

### Segmentation/auxiliary (0 of 12 diagnostic columns)

SKI10 (150, OAI-derived DUA), iMorphics (proprietary — killed), OAI-ZIB/OAIZIB-CM (507, inherits OAI),
K2S (300, CC BY-NC-ND registry), KMAR-50K (1,444 scans, CC-BY — SSL/augmentation only, on disk),
TCIA Soft-tissue-Sarcoma (51, **CC BY 3.0 — the only genuinely shippable MSK-MRI licence**, tumour
labels only), TCIA TCGA-SARC (261, CC BY 4.0), RIDER Knee (**RETIRED — gone from TCIA/IDC; do not
build on it**), M4Raw (brain, irrelevant).

### Grand-challenge tasks

SKI10 (seg), KNOAP2020 (OAI-derived, inherits DUA), K2S (recon+seg). **No grand-challenge task with
diagnostic multi-label knee-MRI findings exists.**

### Searched-but-nonexistent (threads closed)

- MRNet-v2 / any Stanford follow-up — does not exist.
- VinDr-knee — VinDr is X-ray/CT only.
- Iranian/Korean public labelled knee-MRI sets — none; the Chinese multi-label sets (CoPAS, AH-Knee,
  OrthoDiffusion's 15,948-MRI corpus) are all private.
- **Public knee MRI + paired radiology reports annotatable by our own extractor — none found. The RSNA
  2026 corpus's MRI+report pairing is unique; the only annotatable-with-reports knee corpus is our
  own.** The highest-value null.
- Any public corpus with a Synovitis label — none (OAI/MOST effusion-synovitis is pooled and blocked).
- Any public corpus with a true Baker's/popliteal-cyst label — none.

## Bottom line

No corpus carrying labels overlapping our twelve permits competition use. OAI is the OA-column corpus
at scale and is licence-closed; CoPAS is the schema twin and is private. Data volume at scale does not
exist for purchase — which makes the power-law price academic unless the owner's thread is about
continued self-labelling of our own 4,407 + unlabelled SSL pools (KMAR-50K, TCIA CC-BY), where the
ceiling term is exactly what the audit prices.
