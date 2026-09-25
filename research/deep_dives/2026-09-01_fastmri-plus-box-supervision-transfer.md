# fastMRI+ box supervision for RSNA knee study-level classification — research note

> Context: fastMRI+ provides per-slice bounding-box annotations (and a few study-level labels) on knee MRI. We want to know whether using that localized supervision on corpus A can improve a binary study-level classifier on our RSNA knee corpus B, with a ~1.5% absolute AUC bar.

## 1. Training formulation: which mechanism has published gains on a *different* target corpus?

The most directly comparable cross-corpus experiment I found is **van Sonsbeek et al., “Probabilistic Integration of Object Level Annotations in Chest X-ray Classification”** (WACV 2023 / arXiv:2210.06980) [[paper](https://arxiv.org/abs/2210.06980)]. Their mechanism is best described as **attention/CAM-like supervision against an object-level map** derived from bounding boxes (and eye-gaze maps). They train a variational classifier in two stages:

1. Train on a large corpus with only image-level disease labels.
2. Freeze early layers and condition the latent prior on pixel-level box/gaze maps from a small richly annotated subset.

This is *not* an auxiliary detection head, MIL, or plain pretrain-then-finetune. On the **same** corpus, adding object-level annotations gives consistent gains of roughly **+0.01–0.03 AUC** (e.g., Chest X-ray14 DenseNet121: 0.772 → 0.809 with in-corpus bounding boxes; MIMIC-CXR DenseNet121: 0.807 → 0.836 with in-corpus gaze maps).

For the **cross-corpus** setting, they train the base model on one chest-X-ray corpus and apply object-level annotations from the other corpus in stage 2. The result is **not a reliable gain**:

| Base corpus | Annotation source | Backbone | Base AUC | Cross-corpus AUC | Δ |
|---|---|---|---|---|---|
| MIMIC-CXR | Chest X-ray14 bounding boxes | DenseNet121 | 0.807 | 0.798 | −0.009 |
| MIMIC-CXR | Chest X-ray14 bounding boxes | ResNet50 | 0.804 | 0.788 | −0.016 |
| Chest X-ray14 | MIMIC-CXR subsets | DenseNet121 | 0.772 | 0.789 | +0.017 |
| Chest X-ray14 | MIMIC-CXR subsets | ResNet50 | 0.763 | 0.757 | −0.006 |

I did **not** find published cross-corpus gains for an auxiliary detection head, MIL with box-derived instance labels, or plain pretrain-then-finetune specifically using medical bounding-box supervision.

## 2. Graded vs binary supervision under corpus/domain shift

The ordinal-regression literature argues that preserving rank structure (soft labels, metric-aware losses, cumulative links) is better than collapsing ordered grades to one-hot/binary labels when the downstream task respects that order. The standard deep-learning reference is **Díaz & Marathe, “Soft Labels for Ordinal Regression”** (CVPR 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html)].

In medical imaging, **MORF (Meta Ordinal Regression Forest for Medical Image Classification With Ordinal Labels)** showed that using the ordinal “unsure” nodule grade in LIDC-IDRI improves the final benign-vs-malignant binary classification on the same corpus: ResNet-18 accuracy rose from ~0.855 to 0.886 when unsure nodules were included via ordinal training [[paper](https://www.ieee-jas.net/article/doi/10.1109/JAS.2022.105668?viewType=HTML&pageType=en)].

However, **I found no medical-imaging paper that directly compares ordinal vs binary supervision for a binary target under corpus or domain shift**. So the claim that graded ACL/MCL labels will beat binary labels on our RSNA corpus is not supported by a published transfer result.

## 3. Honest effect-size range on a shifted corpus

From van Sonsbeek et al. Table 3, the cross-corpus effect of object-level box/gaze supervision is roughly **−0.02 to +0.02 AUC**, with several backbones showing small negative transfer. The in-corpus gains are larger but still only **+0.01–0.03 AUC**.

So the honest published range under domain shift is **about ±0.005 AUC, not the ~0.015 (1.5%) absolute AUC bar you set**. If anything, the cross-corpus expectation from the closest precedent is near zero or slightly negative.

## 4. Slice decimation: bounding-box supervision on a thinned slice stack

There is limited MRI-specific precedent. The closest precedent is **CT**, where the **Brain Hemorrhage Extended (BHX)** dataset took manually drawn bounding boxes on thick-slice head CT and extrapolated them to thin-slice CT by matching slice positions via the DICOM `Image Position (patient)` tag, propagating the same 2D box to nearby slices [[PhysioNet](https://physionet.org/content/bhx-brain-bounding-box/)]. The authors explicitly note that “nearest z position” matching plus interpolation is the expected fix when the extrapolation is imperfect.

For your case (~12 of ~35 slices retained):

- **Standard move**: keep the original-slice → kept-slice index map and place each annotated box on the nearest kept slice. If you want to be slightly smoother, linearly interpolate the box center and size between the two kept slices that bracket the annotated original slice.
- **Better move**: do not decimate slices that carry bounding-box annotations. Keep a dense local window around every annotated slice (say ±2 slices) and only decimate the unannotated portions of the stack. Alternatively, run the box-supervision loss on the original slice grid, and decimate only the backbone feature maps / attention pooling, not the label grid.

The fastMRI+ annotations are stored as original slice indices (Zhao et al., *Scientific Data* 2022) [[paper](https://pubmed.ncbi.nlm.nih.gov/35383186/)], so the mapping must be applied at ingestion; otherwise the localization signal is silently dropped.

## 5. Dual-T4 inference in Kaggle kernels (optional)

People do use Kaggle’s `nvidia-t4-x2` runtime for inference. For example, the **ROGII** winning solution ran a neural ensemble across two T4 GPUs during inference [[discussion](https://www.kaggle.com/c/rogii-wellbore-geology-prediction/discussion/733135)], and **CZII Cryo-ET** top entries split predictions across the two GPUs [[discussion](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561401)]. A BirdCLEF write-up reports ~1.8× throughput with `nn.DataParallel` on T4×2 [[notebook](https://www.kaggle.com/code/karnakbaevarthur/s124-g124-reverse-engineered)]. The practical benefit depends on model size and communication overhead; many teams avoid the complexity unless the model is small enough that the second GPU is genuinely useful.

---

## Bottom-line recommendation

Using fastMRI+ bounding-box labels as auxiliary supervision is **unlikely to deliver the ~1.5% absolute AUC lift** on our RSNA study-level classifier. The only closely related cross-corpus experiment (chest-X-ray attention/box conditioning) shows **~0 AUC transfer**; ordinal labels have **no demonstrated transfer benefit**; and slice decimation forces an annotation-mapping step that is only precedented in CT. If you still want to test the idea, the cheapest experiment is to add a slice-index-resolved attention-loss branch on fastMRI+ and measure zero-shot or fine-tuned transfer on a small held-out RSNA fold before committing to a full run.

## Sources consulted

- van Sonsbeek, Zhen, Mahapatra, Worring. “Probabilistic Integration of Object Level Annotations in Chest X-ray Classification.” WACV 2023 / arXiv:2210.06980. https://arxiv.org/abs/2210.06980
- Díaz & Marathe. “Soft Labels for Ordinal Regression.” CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html
- Meta Ordinal Regression Forest for Medical Image Classification With Ordinal Labels. IEEE/CAA JAS 2022. https://www.ieee-jas.net/article/doi/10.1109/JAS.2022.105668?viewType=HTML&pageType=en
- Reis et al. “Brain Hemorrhage Extended (BHX): Bounding box extrapolation from thick to thin slice CT images.” PhysioNet 2020. https://physionet.org/content/bhx-brain-bounding-box/
- Zhao et al. “fastMRI+, Clinical pathology annotations for knee and brain fully sampled magnetic resonance imaging data.” *Scientific Data* 2022. https://pubmed.ncbi.nlm.nih.gov/35383186/
- ROGII winning-solution discussion (dual-T4 inference). https://www.kaggle.com/c/rogii-wellbore-geology-prediction/discussion/733135
- CZII Cryo-ET 4th-place discussion (two T4 GPUs for parallel inference). https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561401
- BirdCLEF reverse-engineered notebook (~1.8× DataParallel speedup on T4×2). https://www.kaggle.com/code/karnakbaevarthur/s124-g124-reverse-engineered
