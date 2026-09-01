# Missing-sequence handling in multi-contrast knee MRI

Claude's Q-Q asks what the field has already measured about absent MRI
sequences, so we do not re-derive it on a GPU card. This note collects the
relevant literature and translates it to the knee-MRI setting.

## 1. Taxonomy: what published models do when a sequence is absent

| Strategy | Mechanism | Representative work |
|---|---|---|
| **Naive zero-fill** | Missing sequence channel is set to zeros at inference; model trained only on complete inputs. | nnU-Net baseline in Chrysochoou et al. 2026; "Filling" in Zhang et al. 2024 |
| **Modality / input-level dropout** | Randomly zero entire sequence channels during training so the network learns to function with any subset. | Grøvik et al. 2021 (brain mets); Chrysochoou et al. 2026 (pediatric brain tumors) |
| **Copy substitution** | Replace missing sequence with its closest physical counterpart (e.g. FLAIR ← T2w, T1-pre ← T1-post). | Chrysochoou et al. 2026 |
| **Generative synthesis** | Train a GAN/diffusion/transformer to synthesize the missing sequence from the available ones, then feed the completed set to a standard segmentation/classification model. | ResViT synthesis in Chrysochoou et al. 2026; many BraTS-oriented synthesis papers |
| **Learned missing embedding / prompt** | Use a learnable vector or prompt that encodes "this modality is absent" and distill knowledge from a multi-modal teacher. | Zhang et al. 2024 (WSI + clinical data) |
| **Per-availability submodels** | Train a dedicated model for each observed subset of modalities. | Alluded to in AAAI/BraTS missing-modality surveys; rarely used in clinical MRI because the number of subsets explodes |

For segmentation with one or two sequences missing, the two best-measured
contenders are **modality-dropout training** and **generative synthesis**;
zero-fill is the baseline, copy substitution is a cheap heuristic, and learned
prompts are better studied in multimodal tabular/image settings than in
multi-sequence MRI.

## 2. Measured comparison: modality-dropout vs zero-fill

### 2.1 Grøvik et al. 2021 — brain metastasis segmentation

- Task: automatic detection and segmentation of brain metastases on
  multisequence MRI.
- Design: an input-level dropout (ILD) DeepLab V3 was trained on all
  combinations of four MRI sequences; the test cohort was missing one sequence.
- The comparison model was a standard DeepLab V3 trained only on the three
  sequences present in the test set (i.e., the zero-fill / limited-input
  baseline).
- Results:
  - AUC: **0.989 vs 0.989** (no difference).
  - Dice: **0.795 vs 0.774** (Δ +0.021, p = 0.017).
  - IoU: **0.561 vs 0.492** (p < 0.001).
  - False positives ≥ 10 mm³: **3.6/patient vs 7.0/patient** (p < 0.001).

Interpretation: training with random sequence dropout matched the AUC of a
model trained on the exact available sequences but produced cleaner segmentations
and fewer false positives. The test-set missingness here is **institution-level**
(all test cases lack the same sequence), not mixed availability within a test
set.

### 2.2 Chrysochoou et al. 2026 — pediatric brain tumor segmentation

- Task: tumor subregion segmentation (whole tumor, enhancing tumor,
  non-enhancing tumor) and downstream survival risk stratification.
- Cohorts: 715 patients from CBTN/BraTS-PEDs; external 43-patient PNOC003/007
  longitudinal cohort.
- Dropout: nnU-Net trained with FLAIR and T1w-pre each zeroed with p = 0.4.
- Baseline: standard nnU-Net with missing sequences zeroed at test time.
- Key segmentation numbers under missing FLAIR (median Dice, CBTN / PNOC):
  - Dropout: **0.87 / 0.74**
  - Synthesis: **0.85 / 0.68**
  - Copy: worse
  - Baseline (zero-fill): worse still
- Under missing T1w-pre (median Dice, CBTN / PNOC):
  - Dropout: **0.79 / 0.78**
  - Synthesis: **0.71 / 0.74**
- Average per-patient Dice drop from complete-input to missing-input was never
  more than **0.03–0.04** for Dropout across all regions and missingness
  patterns.

This is the cleanest mixed-availability segmentation comparison found:
modality-dropout training dominates zero-fill and is more robust than a
state-of-the-art generative imputation pipeline.

### 2.3 Zhang et al. 2024 — axillary lymph-node metastasis prediction

- Task: predict axillary lymph-node metastasis from whole-slide images (WSI)
  plus tabular clinical data.
- Their "Filling" baseline replaces missing clinical features with zeros; it
  degrades heavily as the missing rate rises and can fall below a single-modal
  model.
- Their bidirectional-distillation method reaches **AUC 0.861** with complete
  modalities and **0.842 at 80% missing clinical data**.

The transfer to MRI is indirect: it shows that a learned missing prompt can
preserve prediction quality under high missing rates, but the missing data are
clinical features, not imaging sequences.

## 3. Generative imputation: downstream task gain vs image quality

Generative models synthesize visually plausible missing sequences. The question
is whether that improves the actual diagnostic task.

- Chrysochoou et al. 2026 report synthesis SSIM **> 0.90** for both FLAIR and
  T1w-pre, with median PSNR ~20–27 dB. The synthetic scans look anatomically
  reasonable and sometimes remove artifacts.
- Despite the high image-similarity scores, **synthesis did not beat dropout on
  downstream segmentation** in any missing scenario; Dropout achieved higher
  median Dice and narrower interquartile ranges in the clinically important
  cases (NET with missing FLAIR, ET with missing T1w-pre).
- Downstream survival analysis under missing FLAIR:
  - Dropout C-index **0.65**, tumor-volume HR 1.57 (p = 0.0045).
  - Synthesis C-index **0.63**, tumor-volume HR 1.44 (p = 0.0082).
  - Baseline zero-fill C-index **0.61**, tumor-volume HR 1.29 (p = 0.106).

So the only systematic medical-imaging comparison located confirms Claude's
prior: generative imputation wins on image-similarity metrics but does not win
on the downstream task; modality-dropout training does.

## 4. Availability as a feature: is sequence availability correlated with pathology?

This is the leak-shaped question. The answer is nuanced.

- **No published knee-MRI cohort was found that quantifies a correlation between
  which sequences are present and the underlying pathology.**
- However, the broader MRI literature shows that sequence selection is
  **indication-driven**, i.e., missing-not-at-random:
  - Denck et al. 2022 (automated MRI protocoling) note that "the responsible
    radiologist selects the appropriate protocol depending on the clinical
    indication, the ordered procedure, prior imaging exams ...". They also
    compute overlap coefficients of 0.04–0.57 between admitting diagnoses and
    assigned protocols, showing that diagnosis heavily influences, but does not
    fully determine, protocol choice.
  - Brown et al. 2018 (sequence-level automated MRI protocoling in
    neuroradiology) trained a gradient-boosting model to select individual MRI
    sequences from free-text clinical indications, reaching 95.1% accuracy.
    This only works because indications strongly predict which sequences are
    ordered.
  - Parkar et al. 2024 (ESR knee-MRI recommendations) state that "some imaging
    centres vary their knee MRI protocols according to the clinical query"
    while still recommending a base protocol with both T1-weighted and
    fluid-sensitive images.

Implication for a model: a binary "T1 present / absent" indicator is likely to
carry information about the clinical indication and therefore about the
pathology distribution. Feeding that indicator directly into the classifier is a
potential leak unless the indication is itself part of the intended input
(e.g., a known referral flag). It is safer to treat availability as a channel
mask inside the network (as in modality dropout) than as a learnable feature
for the final predictor.

## 5. What this means for the current corpus

- Fill rates in our data: fluid-sensitive slots are 93–100% present; T1 slots
  are only 55–60% present.
- CoPAS external validation shows that withholding T1W/T2W drops mean AUC from
  **0.812 to 0.721/0.726**, with bone-contusion AUC falling from 0.821 to
  ~0.698. So T1/T2 carry genuine signal for some labels.
- The literature therefore points toward **modality-dropout training** as the
  most evidence-backed way to exploit the 60% of studies that have T1 without
  discarding the 40% that do not.
- Generative T1 synthesis is not recommended as a first move: it would add a
  large training and inference cost and the only direct comparison shows it
  underperforms dropout on the downstream task.
- If an availability indicator is used, it should be treated as a leak risk,
  because sequence selection is known to be indication-driven.

## References

- Grøvik E, Yi D, Iv M, et al. Deep learning enables automatic detection and
  segmentation of brain metastases on multisequence MRI with missing data.
  *npj Digital Medicine*. 2021;4:1. PMCID: [PMC7900111](https://pmc.ncbi.nlm.nih.gov/articles/PMC7900111/).
- Chrysochoou D, Gandhi DB, Adib S, et al. AI-powered segmentation and
  prognosis with missing MRI in pediatric brain tumors. *npj Precision
  Oncology*. 2026;10:63. PMCID: [PMC12894653](https://pmc.ncbi.nlm.nih.gov/articles/PMC12894653/).
- Zhang S, Zheng S, Shui Z, Li H, Yang L. Multi-modal Learning with Missing
  Modality in Predicting Axillary Lymph Node Metastasis. arXiv:2401.01553
  [cs.CV], 2024. https://arxiv.org/abs/2401.01553
- Denck J, Guehring J, Maier A, et al. Automated protocoling for MRI
  examinations — challenges and potential solutions. *Insights into Imaging*.
  2022;13:178. PMCID: [PMC9582071](https://pmc.ncbi.nlm.nih.gov/articles/PMC9582071/).
- Brown AD, Marotta TR. Using machine learning for sequence-level automated
  MRI protocol selection in neuroradiology. *J Am Med Inform Assoc*.
  2018;25(5):568–571.
- Parkar AP, Maas M, Feydy A, et al. ESR essentials: MRI of the knee —
  practice recommendations by ESSR. *Insights into Imaging*. 2024;15:44.
  PMCID: [PMC11399221](https://pmc.ncbi.nlm.nih.gov/articles/PMC11399221/).
- Qiu J, Chen Y, Chen S, et al. CoPAS: A composite backbone based on
  plug-and-play modules for accurate knee MRI abnormality diagnosis.
  *Nature Communications*. 2024;15:6424. PMCID: [PMC11368947](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)
  (external-validation numbers cited from the supplement).
