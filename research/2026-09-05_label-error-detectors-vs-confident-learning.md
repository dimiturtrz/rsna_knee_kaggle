# Label-error detectors that beat confident learning, and the missing k

Context: Claude asked two questions about our label-noise detector (cleanlab-style confident learning over out-of-fold predictions from 20 checkpoints).

1. **Q-K:** Is there a published measurement of k, the relative cost of a false correction versus a missed error, under sample re-weighting or soft re-labelling?
2. **Q-DET:** What beats confident learning at *ranking* label errors on the same out-of-fold predictions without new supervision?

## Q-K — no measured k found

I found no paper that defines and measures k = cost(false correction) / cost(missed error) under sample re-weighting or soft re-labelling, with downstream accuracy plotted against detector precision at fixed budget.

- **Bernhardt et al. (2022)** frames budget-constrained active label cleaning and reports cleaning-efficiency AUC, but maximises corrected samples under budget; it does not price false corrections separately.
- **LEMoN / Zhang et al. (ICML 2024)** plots downstream test error vs % data retained — the right curve shape — but does not extract k.
- **Das et al. (ICML 2025)** proves consensus-based retraining helps; it avoids relabelling entirely, implicitly treating false corrections as catastrophic.
- **Srikanth et al. (2023)** measures retrained-model AUC vs amount removed, but again does not model the false-correction / missed-error trade-off.

Conclusion: k is assumed, not measured, in the literature I could find.

## Q-DET — methods measured to beat CL

| Method | Family | Key result vs CL |
|--------|--------|------------------|
| **SimiFeat** (Zhu et al., ICML 2022) | pretrained-embedding kNN + TTA | Beats CL, TracIn, CORES on CIFAR-10/100 synthetic and human noise; ~+20% F1 on CIFAR-10 human noise. |
| **SEMD** (Srikanth et al., 2023) | linear-probe CLIP + CL + TTA | Beats CL on 6/8 CIFAR noise settings; CL -0.206 F1 on CIFAR-10 symmetric, -0.156 on CIFAR-100 symmetric. |
| **AUM** (Pleiss et al., NeurIPS 2020) | training-dynamics margin | ≥90% precision/recall in high-noise synthetic settings; beats Data Parameters / DY-Bootstrap / INCV. |
| **LSTM noise detector** (Li et al., NeurIPS 2022) | learned training-dynamics features | Beats AUM on mAP / ROC AUC / Precision@95; transfers across datasets without retraining. |
| **GEX / IGEX** (Kim et al., NeurIPS 2023) | geometric-ensemble influence | Beats CL on AUC/AP for noisy-label detection on CIFAR-10/100, CIFAR-N and ImageNet. |
| **AQuA benchmark** (Goswami et al., NeurIPS 2023) | benchmark over 17 datasets | SimiFeat best, CINCER second, AUM third, **CL worst** across noise types and modalities. |

Data Maps / cartography (Swayamdipta et al., 2020) identifies a hard-to-learn region that often contains label errors, but I found no direct AUC/precision-at-budget comparison to CL.

## Caveat

The published comparisons use **synthetic noise** or **crowd-annotated** labels (CIFAR-N). None report separation AUC or precision-at-budget against a **human-expert adjudicated panel** comparable to our 58-study radiologist set. So the methods give a strong prior that CL is not the ceiling, but the exact gain on our panel must be measured.

## Multi-label extension (Claude's Q-MULTI)

Claude pointed out that our data are multi-label (12 findings per study, one CELL = (study, finding)), while SimiFeat, SEMD and AQuA are mostly single-label.

- **Cleanlab multi-label extension** (Thyagarajan et al., arXiv:2211.13895): the canonical approach. Run CL independently for each of K binary labels using the K-th predicted probability; flag the union of examples with any estimated error; pool per-class self-confidence scores into an example-level score. Their proposed **EMA pooling over sorted scores** (α=0.8) outperforms min/mean/median/softmin/log/cumulative-average pooling on synthetic multi-label data and CelebA.
- **SEMD** uses the same idea under the name "Per-Label Removal" for CheXpert/METER-ML and finds it comparable to an oracle per-class grid search.
- **Per-column independent kNN** is therefore the natural SimiFeat analogue, but I found no paper measuring the precision degradation from correlated neighbour orderings across columns.
- **Label-vector distance** (Hamming/Jaccard among neighbours) has no published attribution rule for mapping vector disagreement back to individual cells.
- **Per-finding embeddings** would be ideal but are not argued as necessary in the multi-label CL literature; a jointly trained multi-label classifier with per-column probabilities appears sufficient.
- **MDPI 2026 chest-X-ray paper** ("Mislabel Detection in Multi-Label Chest X-Rays via Prototype-Weighted Neighborhood Consistency in CoAtNet Embedding Space") is the closest published work to our exact shape. It is training-free neighbourhood consistency on multi-label chest X-rays, evaluated with patient-disjoint splits and a top-20% review budget. I could not fetch the full text due to MDPI's access challenge.

## The missed-finding / false-negative half (Claude's Q-MISS)

Our CL detector reads misses at AUC 0.6903 and precision 0.08–0.11, far below its performance on all disagreements (AUC 0.7628, precision 0.720).

- Radiology literature confirms false negatives are a major failure mode: Noguchi et al. report false-negative error rates of **13–90%** in diagnostic imaging, and 0.8–5% of daily errors.
- **No detector family is published specifically for ranking existing negatives as probable missed findings.** Positive-unlabelled (PU) learning is the right conceptual framing, but it is used to train classifiers under incomplete annotations, not to rank label errors.
- Relevant PU references: Bekker & Davis survey (ML 2020); Zhao et al. on PU learning for incomplete cell annotations in histopathology (MELBA 2023); Jain et al. on PU performance estimation (AAAI 2017).
- The CL / AUM / SimiFeat family is built on confident disagreement with the given label, which structurally favours false positives. Attacking the miss half would likely require anomaly detection on negatives, PU instance-level scoring, or missing-label-aware contrastive learning — all training-centric or new builds rather than literature drop-ins.

## The missed-finding / false-negative half — updated after direct inspection (Claude's Q-MISS)

Claude's measured separation AUCs on the identical gate:
- confident learning: 0.7628 / 0.7580 / 0.6903 (all / confident FP / misses)
- 10-NN vote, 6 trunks: 0.6830 / 0.7714 / 0.3588
- rank-average: 0.7511 / 0.8020 / 0.5161

The 10-NN vote is below chance on misses because a false negative agrees with the negatives around it by construction. This is not a tuning failure; it is structural.

**Direct evidence that standard detectors are extraneous-label detectors**
- Anttonen 2024 evaluated Confident Learning on real MIMIC-IV medical codes against a keyword-search ground truth. CL found erroneous extra labels with high precision but was "too unreliable" for missing labels. [Aaltodoc thesis](https://aaltodoc.aalto.fi/bitstreams/ddc4c95d-ab7c-45f6-9631-357f95be4b77/download)
- Deep k-NN (Bahri et al., ICML 2020) flags a sample when its label disagrees with its k-NN neighbours in representation space. This again detects extraneous labels; false negatives are neighbours with the same (wrong) negative label. [Deep k-NN for Noisy Labels](http://proceedings.mlr.press/v119/bahri20a/bahri20a.pdf)
- Ortega Vázquez 2023 treats PU data and one-sided label noise as "practically equivalent," but the PU literature trains classifiers or estimates class priors; it does not typically produce a spendable ranking. [Lirias](https://lirias.kuleuven.be/retrieve/734064)

**Closest published false-negative detector**
Karimi et al. 2020 (ISBI), "Learning to detect brain lesions from noisy annotations," is a medical-imaging method that directly targets false-negative annotations. It iterates: train a CNN, use CNN predictions + Monte-Carlo dropout uncertainty + lesion statistics (size, T1/T2/FLAIR intensity, distance to nearest annotated lesion) to classify candidate lesions as true missed positives or false positives, add the accepted ones to labels, and retrain. On TSC brain MRI it improved lesion-count F1 from 0.747 to 0.819. It is per-lesion/region and domain-specific, not a per-cell ranking score. [NIH/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7456674/)

**Conclusion for Q-MISS**
No published detector family outputs a per-cell ranking score for one-sided false-negative label noise with usable precision in general classification. PU learning is the right conceptual framing, but its typical output is a classifier or class-prior estimate. A drop-in miss ranker would be a new build, not a literature import.

---

## MRI-CORE checkpoint details (Claude's Q-MRICORE)

Downloaded `MRI_CORE_vitb.pth` from the official release and inspected the state dict. The file is a DINOv2-pretrained ViT-B/16 backbone, not a finished SAM `image_encoder` state dict.

- **URL**: `https://drive.google.com/file/d/1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW/view?usp=sharing`
- **Optional SAM weights** (prompt/mask decoder): `https://drive.google.com/file/d/1oWoVGdwXwNTVr1oTLV_dRDGEJZyKJLnJ/view?usp=sharing`
- **Top-level key**: `{'teacher': OrderedDict(...)}`
- **Prefixes inside teacher**: `backbone.*` (151 keys), `dino_head.*` (8 keys)
- **Real keys**: `teacher.backbone.patch_embed.proj.weight`, `teacher.backbone.pos_embed`, `teacher.backbone.blocks.0.0.attn.qkv.weight`, ...
- **Position embedding**: `[1, 197, 768]` → 14×14 patch grid + cls token → **224×224 pretrain read size**, 16×16 patches
- **Depth**: 12 layers stored as 4 groups × 3 sub-blocks (`blocks.G.s`)
- **Params**: ~85.8 M backbone, ~108.9 M with DINO head
- **Normalization**: `[0, 1]` FloatTensor, per README feature-extraction example
- **License**: Apache 2.0, stated to cover code and models

To use as a 1024 px SAM encoder, the loader must remap `teacher.backbone.*` → `image_encoder.*` and interpolate `pos_embed` from 14×14 to 64×64. Their `models.sam.sam_model_registry` does this.

## Practical order for us

Given we already store per-epoch out-of-fold predictions:
1. **AUM** is the cheapest add — needs only the logits we already have.
2. **LSTM/sequence detector on dynamics** is strongest if we train a small meta-detector on synthetic noise.
3. **SimiFeat-style kNN on trunk embeddings** is strongest if we have good embeddings and can afford TTA. For multi-label, run it per-column and pool scores (EMA-style) across the 12 findings.
4. **Missed-finding ranking** is a research gap; PU scoring of "absent" cells is the most plausible first experiment.
