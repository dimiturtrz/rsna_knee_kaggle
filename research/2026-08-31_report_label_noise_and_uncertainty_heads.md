# Report-derived label noise and uncertainty heads — what the field has already measured

External synthesis, read 2026-08-31, prompted by the owner's question "would it help if our heads returned
both label score and confidence separately?" and the follow-up "how would we collapse them".

Everything here is someone else's measurement. Our own numbers appear only where they collide with one.

## 1. A separate confidence head: the field has benchmarked it, and it loses

*Benchmarking Uncertainty and its Disentanglement in multi-label Chest X-Ray Classification*
(arXiv 2508.04457, MIMIC-CXR-JPG, 13 uncertainty-quantification methods, ResNet and ViT). It extends
Evidential Deep Learning, HetClass NNs and Deep Deterministic Uncertainty to the multi-label setting —
which is our setting, twelve independent binary findings.

Findings that bear on us:

- **Ensembles win.** "Ensemble-based methods (D-Ens, S-Ens) consistently perform best across
  architectures." Evidential DL trails on OOD detection.
- **Evidential heads are badly calibrated.** EDL at ECE ~0.4, described as "strongly ill-calibrated for
  both models".
- **The aleatoric/epistemic split is degenerate in practice.** "Almost all methods show significant
  correlations between EU and AU", and many "fail to disentangle uncertainty types, with AU estimates
  performing comparably or better" on epistemic tasks. The authors advise applying the
  information-theoretic framework "with caution".
- **A gap in their work, not in ours:** they never report whether adding an uncertainty head changes
  AUROC/AUPRC against a plain BCE baseline. The whole evaluation is calibration, OOD and correctness
  prediction. So the published record does not even claim a classification gain.

**What this does to knee-qrtx.** Our 12.7%/8.3% reducible-uncertainty read uses the same
information-theoretic decomposition this paper cautions about. Two things keep it standing: ours is
computed over a *deep ensemble* of separately trained members, which is the method class that won their
benchmark, and it is used as an upper bound on what a new member can buy rather than as a per-cell
signal. But the caveat is real and belongs in that interpretation: the EU/AU correlation they measure
means our "87.3% aleatoric" should be read as "at most 12.7% is addressable by more members", not as a
clean partition of the world into ambiguity and ignorance.

**Consequence for the head question.** A single-model confidence head is a worse-identified version of
the ensemble disagreement we already compute, its ensemble use is capped by the 8.3% we measured, and its
training-time use is self-referential. Filed against the confidence-head bead as published evidence, not
as an argument.

## 2. Rep-GLS — our labelling scheme, published, with the gain on the axis we do not have

*Clinical Expert Uncertainty Guided Generalized Label Smoothing* / *Rep-GLS: Report-Guided Generalized
Label Smoothing for Robust Disease Detection* (arXiv 2508.02495). MIMIC-CXR-JPG, 14 diseases.

Mechanism, which is close enough to ours to be worth stating precisely:

1. An LLM (Qwen-3 4B, few-shot) parses each free-text report into structured JSON mapping each finding to
   its uncertainty keyword (`"Pneumonia": "suspicious for"`).
2. A **Rate Generation Network** maps keywords to a continuous smoothing rate `r` in `(-1, 1)` via tanh.
   A clinical expert hand-assigned ground-truth rates to 19 uncertainty terms, from **-0.2 for "positive"**
   to **0.9 for "cannot be evaluated"**; the RGN is pretrained on those.
3. Per-class adaptive loss: `L_i^(k) = (1 - r_i^(k)) * L_CE + r_i^(k) * L_uniform`.
   `r > 0` regularises uncertain cells toward uniform. `r < 0` **sharpens** confident cells — it
   subtracts the uniform term, pushing the model to be *more* decisive than plain CE. Unmentioned
   findings get `r = 0`.

Results, mean AUC over 14 diseases:

| | mean AUC |
|---|---|
| BCE baseline | 79.63 |
| Rep-GLS end-to-end (no expert RGN pretraining) | 80.27 |
| Rep-GLS, no sharpening (`r >= 0` only) | 81.56 |
| Rep-GLS full (sharpening enabled) | **84.14** |

**The ablation is the finding.** Softening uncertain cells buys +1.93. **Sharpening confident cells buys
a further +2.58** — more than half the total, and the larger half. Expert-supervised two-stage training
beats end-to-end by 3.87.

**Where we stand relative to it.** We implement the softening half and nothing else. `Supervision.applied`
maps reader certainty to measured agreement `0.08 / 0.28 / 0.68 / 0.82 / 0.94` — the top of that range is
0.94, so a maximally confident report is still softened relative to a hard 1.0. `weight_floor` gives
`0.25 + 0.75*conf`, which only ever *down*-weights. There is no term anywhere that pushes a confident cell
past plain BCE. On Rep-GLS's ablation that is the larger of the two levers, and we have none of it.

## 3. Why we cannot simply add sharpening — and what that implicates instead

Rep-GLS's sharpening assumes a confident report statement is true. The expert rate table encodes exactly
that assumption: "positive" gets `r = -0.2`.

Our own measurement says otherwise:

- `supervision_gap`: where the report says YES, the radiologist agrees **65.3%** of the time.
- `noise_binding`: **108 confident false positives** over the 58 annotated studies, all verdict YES,
  **15.8% of every annotated cell**, against 32 misses.

Sharpening drives the model harder onto those 108 cells. On our data the published lever points the wrong
way, and it points wrong precisely where it is largest.

**The asymmetry is the lead.** Their reports are radiologist-written and LLM-read once. Ours are LLM-read
from report text too, and the RSNA challenge page states report-derived labels agree only **~82%** with
ground truth — which our own reader read brackets (78.9% binarised at 0.5, 84.2% at the best threshold,
mean rank AUC 0.866 over 58 x 12 = 696 cells). So the question is *whose* error the 65.3% is: the
radiologist who wrote the report, or the extractor that read it.

The negation literature makes that a live hypothesis rather than a guess. Negation errors — "no evidence
of ACL tear" read as a positive — are the classic report-labeler failure mode, alongside left/right and
interval-change errors; and recent work reports that **LLMs are specifically weak at detecting negated
expressions** relative to purpose-built models (CAN-BERT beat medspaCy on negation; several studies find
Llama-3 / GPT-4 poorer at negated expressions than at assertion).

A false positive produced by a missed negation is *confident by construction* — the extractor saw the
finding named in the text and had no reason to hedge. That is exactly the signature of our 108: all
verdict YES, all high confidence.

**This is checkable without a GPU and without a submission slot.** Take the 108 confident FPs, pull the
report sentence each was extracted from, and classify: missed negation, laterality error, hedge dropped,
temporal/prior-study reference, or genuine radiologist-vs-radiologist disagreement. The split decides the
whole label lane:

- If a material share are extraction failures, the fix is the **extractor**, and it is corpus-wide —
  4,407 studies, not 58. That is a bigger lever than any loss term, and it is upstream of every member
  ever trained here.
- If they are genuine reader disagreement, then Rep-GLS-style sharpening stays contraindicated for us and
  the softening half we already have is the correct half.

Either answer is worth having, which is what makes it an instrument rather than a bet.

## 4. The CheXpert taxonomy, for naming what we do

CheXpert's uncertainty-label strategies are the standard vocabulary and we should use it when describing
our scheme: **U-Ignore** (mask uncertain cells), **U-Zeros** / **U-Ones** (map to 0 or 1),
**U-MultiClass** (uncertainty as a third class), **U-SelfTrained** (train on certain cells, then relabel
uncertain ones from the model's own thresholded predictions, iterate).

We are none of these exactly: we are a **soft-label** variant, mapping certainty to a measured agreement
probability. The nearest published relative is Rep-GLS.

Two things worth carrying:

- CheXpert found the best strategy is **pathology-dependent** (U-Ones helps Atelectasis, U-MultiClass wins
  Cardiomegaly). That sits in tension with our `finding_calib_dense` result, where per-finding calibration
  read FLAT. The tension is informative rather than contradictory: they vary the *strategy* per pathology,
  we varied a *scalar offset* per finding. Ours was the weaker knob, which is a reason the flat result may
  not generalise to the stronger one.
- **U-SelfTrained** is confident learning's cousin and the published precedent for the detector currently
  running (knee, CPU, 2026-08-31): relabel from model predictions rather than from reader-side text. The
  difference that matters is that ours uses strictly out-of-fold predictions, so it is not circular in the
  way self-training on in-sample predictions is.

## 5. Field-wide, on noisy-label method families

Recent surveys and benchmarks converge on: neither loss reweighting nor sample selection dominates, and
the combination (label correction + reweighting, or selection + robust loss) is what holds up at higher
noise rates. Small-loss sample selection has a provable clean-sample identification result behind it. No
single method transfers as a drop-in, which argues for gating any of it on our own gold-58 read rather
than adopting a published recipe.

## Sources

- [Benchmarking Uncertainty and its Disentanglement in multi-label Chest X-Ray Classification (arXiv 2508.04457)](https://arxiv.org/abs/2508.04457)
- [Rep-GLS / Clinical Expert Uncertainty Guided Generalized Label Smoothing (arXiv 2508.02495)](https://arxiv.org/html/2508.02495)
- [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://www.researchgate.net/publication/330552948_CheXpert_A_Large_Chest_Radiograph_Dataset_with_Uncertainty_Labels_and_Expert_Comparison)
- [A Large Language Model to Detect Negated Expressions in Radiology Reports](https://link.springer.com/article/10.1007/s10278-024-01274-9)
- [Learning to estimate label uncertainty for automatic radiology report parsing (arXiv 1910.00673)](https://arxiv.org/pdf/1910.00673)
- [RSNA Knee Abnormality Detection AI Challenge (2026)](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- [Efficiency and safety of automated label cleaning on multimodal retinal images (npj Digital Medicine)](https://www.nature.com/articles/s41746-024-01424-x)
- [Automatic Deep Learning-assisted Detection and Grading of Abnormalities in Knee MRI Studies (Radiology: AI)](https://pubs.rsna.org/doi/abs/10.1148/ryai.2021200165)
- [MRNet — Deep learning to assist clinicians in reading knee MRI](https://stanfordmlgroup.github.io/projects/mrnet/)
