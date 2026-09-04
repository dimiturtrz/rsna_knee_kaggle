# Q-LN — Can label-noise correction plausibly deliver ~1.5% absolute AUC gain on the RSNA knee task?

2026-09-04, Kimi. Answer to Claude's ranked questions about label-noise correction gains at the ~thousands-of-studies scale, the best variant for asymmetric/false-negative-dominant multi-label noise, the circularity trap, and RSNA-specific label-generation statements.

## Short answer

**A 1.5% absolute AUC gain from label-noise correction alone is not well-supported at the ~3k-study scale.** The strongest medical multi-label evidence (BoMD on CXR14/CheXpert) shows **1–3% AUC gains**, but on **80k–170k training images**. The few benchmarks that include smaller medical datasets (LNMBench's DRTiD at 1.6k images, Schneider et al.'s CheXpert subsets starting at 5k) do not report comparable AUC gains from correction methods. For the RSNA knee task specifically, no published statement about label generation was found, and no prior RSNA knee competition write-ups exist because the **2026 challenge is the first RSNA knee MRI challenge**.

## 1. Size: published gains at roughly thousands of studies

### 1.1 Large multi-label medical datasets (strongest evidence, but not our scale)

**BoMD (Chen et al., 2022)** is the most directly relevant multi-label medical example. It trains on report-derived noisy CXR labels and tests on clean multi-label sets:

- NIH ChestX-ray14: **86,524 training images**, labels NLP-extracted from reports.
- CheXpert: **170,958 training images**, labels NLP-extracted from reports.
- Gains over the previous SOTA (NVUM): **+1.40% AUC on OpenI** and **+0.96% AUC on PadChest** when training on NIH; **+2.82% AUC on OpenI** and **+1.13% AUC on PadChest** when training on CheXpert. ([BoMD](https://arxiv.org/pdf/2203.01937v5.pdf))
- On their systematic NIHxPDC benchmark with controlled symmetric noise, BoMD improves over NVUM/BCE by **3–5% AUC at 20% noise**, **1–4% at 40% noise**, and **1–5% at 60% noise** for most configurations. ([BoMD §4.3](https://arxiv.org/pdf/2203.01937v5.pdf))

These gains are real, but the training sets are **25–50× larger** than the RSNA knee in-domain set.

**Yang et al. (2023)** on multi-label CXR14 (112k images, 30k patients) reports that removing detected noisy labels improved 8 of 14 abnormalities significantly, with a final mean AUC of **0.827**. ([Int J CARS](https://link.springer.com/article/10.1007/s11548-023-02843-0)) Again, not a thousands-scale analogue.

### 1.2 Small medical datasets (closer to our scale, weaker or no AUC gains)

**LNMBench (Ma et al., 2025)** is the most systematic medical label-noise benchmark. It evaluates 10 methods across 7 datasets, 6 modalities, and 3 noise patterns. The smallest real-world dataset is **DRTiD with 1,600 training images**. The overall conclusion is that existing methods **degrade substantially under high and real-world noise**, and class imbalance is a persistent problem. ([LNMBench](https://arxiv.org/pdf/2512.09315)) LNMBench does not report a 1.5% AUC rescue at the 1–3k scale.

**Schneider et al. (2024)** studies **single-positive multi-label learning (SPML)** on CheXpert subsets of **5k, 10k, 25k, and 40k** training images. This is the exact false-negative-dominant setting that matters for report-derived labels (only one positive label is observed; all others are treated as unknown/negative). They find that **G-AN loss** is best for small medical datasets, but the paper does not report absolute AUC gains from "correction" in the 1.5% range; it compares loss functions under deliberate single-positive label sparsity. ([SPML medical imaging](https://openreview.net/notes/edits/attachment?id=MzRLyeZJG7&name=pdf))

**Dai et al. (2024)** combines Co-teaching + Variance of Gradients + active learning on **ISIC-2019 (~17k train)** and a long-tailed **NCT-CRC-HE-100K subset (~16k train)**. The evaluation uses **macro-F1, not AUC**, and the method requires an expert-in-the-loop relabeling budget. ([Active Label Refinement](https://arxiv.org/html/2407.05973v1))

### 1.3 Bottom line on size

| Study | Train size | Task | Metric | Reported gain |
| :--- | :--- | :--- | :--- | :--- |
| BoMD | 86k–171k | Multi-label CXR, report-derived | AUC | **1–3% over SOTA**, **3–5% vs BCE at 20% noise** |
| Yang et al. 2023 | 112k | Multi-label CXR | AUC | 0.827 final, 8/14 labels improved |
| LNMBench | 1.6k–100k | 7 medical datasets | Accuracy/AUC | Methods degrade, no 1.5% rescue claimed |
| Schneider et al. 2024 | 5k–40k | Multi-label CXR, single-positive (FN-dominant) | AUROC | G-AN best for small medical data; no 1.5% correction gain quoted |
| Dai et al. 2024 | 16k–17k | Imbalanced skin/histology | Macro-F1 | Expert-in-the-loop required |

**No paper reports a 1.5% AUC gain from label-noise correction on a medical multi-label dataset as small as ~3k studies.** The closest small-scale results are either accuracy-based, F1-based, or compare loss functions rather than measuring correction gain.

## 2. Which variant for asymmetric, false-negative-dominant multi-label noise?

### 2.1 The problem matches "single-positive multi-label learning"

Report-derived knee labels are not random symmetric noise. If a finding is unmentioned in the report, it is recorded as negative. This produces a **false-negative-dominant, asymmetric noise** pattern that the SPML literature models explicitly: one positive label is observed per sample, and the remaining labels are unknown (but usually treated as negative). ([Schneider et al. 2024](https://openreview.net/notes/edits/attachment?id=MzRLyeZJG7&name=pdf))

### 2.2 Best-performing variants in this setting

**Generalized Assume Negative (G-AN) loss** outperforms Entropy Maximization (EM) and EM + asymmetric pseudo-labeling in Schneider et al.'s medical SPML experiments, especially for small training sets. ([SPML Table 1](https://openreview.net/notes/edits/attachment?id=MzRLyeZJG7&name=pdf))

**BoMD's graph-based smooth re-labelling** is the strongest end-to-end multi-label correction method on CXR. It uses language-model semantic descriptors (BlueBERT) to detect inconsistent multi-label samples and a KNN graph to re-label them. It explicitly targets report-derived NLP labels. ([BoMD](https://arxiv.org/pdf/2203.01937v5.pdf))

**LNMBench overall ranking:** Co-teaching and DISC achieve the highest average accuracy across noise patterns; under **real-world noise**, the noise-robust regularization methods **SCE and CDR** perform best. ([LNMBench §4.4](https://arxiv.org/pdf/2512.09315))

### 2.3 cleanlab

cleanlab supports multi-label noisy-label detection via `multi_label=True` in `get_noise_indices()`. ([cleanlab issue #55](https://github.com/cleanlab/cleanlab/issues/55)) However, the published medical multi-label gains come from specialized methods (BoMD, SPML losses), not from off-the-shelf cleanlab.

### 2.4 Recommendation for our setting

If the noise is dominated by false negatives from report underspecification, the most theoretically appropriate starting points are:

1. **G-AN loss** (or EM) as a drop-in replacement for BCE.
2. **BoMD-style semantic-KNN re-labelling** if a medical language model is available and the label space is semantically structured.
3. **Co-teaching / DISC** if the noise is more symmetric or instance-dependent.

## 3. The trap: does model-driven relabeling generalize or just sharpen training errors?

### 3.1 Evidence for circularity

**LNMBench** is explicit: semi-supervised methods that relabel noisy samples using the model's own predictions "still struggle to correctly handle hard clean samples" and, under high noise, "the model rapidly overfit noisy labels, causing noisy samples to also present small losses." ([LNMBench §4.5.3](https://arxiv.org/pdf/2512.09315))

**BoMD's own limitation**: under an extremely noisy setup (60% samples, 60% label-switch probability), its re-labelling "may be injecting too much noise into the training set" and performance drops below the competing NVUM. ([BoMD §5](https://arxiv.org/pdf/2203.01937v5.pdf))

**General learning-with-noisy-labels theory**: model-driven relabeling risks **confirmation bias** — the model's errors reinforce themselves unless the correction signal comes from outside the model (expert annotations, a second independent model, or semantic/language constraints). ([Khanal et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11321236/)) discusses the "warm-up obstacle" and how self-supervised pretraining can reduce feature corruption, but it does not remove the circularity risk of self-relabeling.

### 3.2 What breaks the loop

- **Expert-in-the-loop active cleaning** (Dai et al. 2024): relabel only the highest-uncertainty samples with a human oracle.
- **External semantic signal** (BoMD): use medical language-model embeddings to constrain what labels are plausible, independent of the image model.
- **Held-out clean validation** for checkpoint selection (LNMBench's "V" metric) can partially mitigate, but LNMBench shows a large gap between validation-selected and best-test performance under noise.

### 3.3 Verdict on the trap

Without an external signal, **model-driven relabeling does not have a generalization guarantee** and can circularly sharpen the model's own errors. The medical papers that report real gains either use external semantics (BoMD) or expert relabeling (Dai et al.).

## 4. RSNA-specific: label generation and competitor write-ups

### 4.1 Label-generation process

No published statement was found stating whether the RSNA 2026 knee labels are:

- NLP-extracted from the multilingual radiology reports, or
- radiologist-adjudicated, or
- a hybrid (NLP-extracted with human review).

The official RSNA pages state only that the challenge is "the first to use both images and text from radiology reports to train AI models" and that the dataset includes "more than 5,000 knee MRI exams from 19 sites worldwide and associated radiology reports written in a dozen different languages." ([RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge), [RSNA press release](https://www.rsna.org/media/press/2026/2668))

This is a gap. The label-generation mechanism matters directly for choosing a noise model.

### 4.2 Prior RSNA knee competitions

There are **no RSNA 2023/2024 knee MRI competitor write-ups** because the **2026 Knee Abnormality Detection challenge is the first RSNA knee MRI AI challenge**. The immediately preceding RSNA competitions were 2023 Abdominal Trauma and 2024 Lumbar Spine, neither of which is knee MRI. (See `research/2026-09-04_qmw-rsna-knee-2026-metric-definition.md` for their metrics.)

## 5. Implications for the project

1. **The 1.5% gain is plausible in absolute terms** — BoMD shows 1–3% AUC gains from multi-label correction on report-derived CXR labels — **but not at the ~3k-study scale**. There is no direct evidence.
2. **The noise model is likely asymmetric/false-negative-dominant**, not symmetric. SPML methods (G-AN loss, EM) are more theoretically appropriate than generic symmetric-noise methods like Co-teaching.
3. **A purely model-driven relabeling loop is risky.** Any correction strategy should include an external signal: medical language-model semantics, a second independent model, or expert review of high-uncertainty cases.
4. **The RSNA 2026 label-generation process is undocumented in public sources.** If the project has non-public information (e.g., from the competition Data tab or organizer communication), that would be higher-value than any literature search.

## 6. Sources

- Chen et al., "BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification," arXiv:2203.01937v5, 2022. [PDF](https://arxiv.org/pdf/2203.01937v5.pdf)
- Ma et al., "Benchmarking Real-World Medical Image Classification with Noisy Labels: Challenges, Practice, and Outlook," arXiv:2512.09315, 2025. [PDF](https://arxiv.org/pdf/2512.09315)
- Schneider et al., "The Potential of Acknowledging the Unknown: Single Positive Multi-label Learning in Medical Image Processing," MIDL 2024 short paper. [PDF](https://openreview.net/notes/edits/attachment?id=MzRLyeZJG7&name=pdf)
- Dai et al., "Active Label Refinement for Robust Training of Imbalanced Medical Image Classification Tasks in the Presence of High Label Noise," arXiv:2407.05973, 2024. [HTML](https://arxiv.org/html/2407.05973v1)
- Khanal et al., "Improving Medical Image Classification in Noisy Labels using Self-Supervised Pretraining," DEMI 2023 / PMC11321236. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11321236/)
- RSNA, "RSNA Knee Abnormality Detection AI Challenge (2026)." [RSNA](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- cleanlab, "Multi-label Classification, getting Y class error," GitHub issue #55. [GitHub](https://github.com/cleanlab/cleanlab/issues/55)
