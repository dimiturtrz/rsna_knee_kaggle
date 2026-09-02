# Q-AL — trunk quality vs readout quality on medical volumes

Question (Claude, 2026-09-02): find published evidence that separates trunk quality from
readout quality on medical volumes. Specifically:

- (a) same frozen trunk under global-pooled linear probe and patch-level / attention-pooled probe;
- (b) 2.5D-vs-3D comparison on knee MRI or comparable volumetric task where a learned depth
  aggregation is measured against max/mean pooling;
- (c) whether DINOv3 ViT-L distilled has a linear-probe classification number vs DINOv2-L.

## (a) Same frozen trunk, different readout

### Strongest evidence — chest radiography (2D, not volumetric)

Muthyala et al. 2026, *Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in Chest
Radiography* ([arXiv:2606.11606](https://arxiv.org/abs/2606.11606)), probes five frozen ViTs
(RAD-DINO, DINOv2-B/14, DINOv3 ViT-7B, BiomedCLIP, MedSigLIP) on ChestX-Det10 (n=3,543) and a
small-lesion perturbation panel. All three pooling modes come from the **same forward pass**:

| Readout | AUC on small-lesion panel | Notes |
|---|---|---|
| CLS token (global) | **0.500–0.524** (chance floor) | global aggregation discards small-lesion signal |
| Patch-mean (all final-layer patch tokens) | similar to CLS on fine lesions, rises with CLS on larger blur footprints | still global-ish |
| Patch-local (bounding-box-restricted patch tokens) | **~1.0** | per-model mean improvement **+0.412 to +0.488** |

This is a direct measurement of the readout gap on the same frozen trunk: the trunk encodes the
lesion; the global readout hides it. The rank of usable signal flips from "none" to "near-perfect"
when the readout is made local.

### Volumetric evidence — AnyMC3D

Liu et al. 2026, *Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification*
([arXiv:2512.12887](https://arxiv.org/abs/2512.12887)), adapts frozen 2D FMs to 3D volumes. They
explicitly compare slice-embedding aggregation strategies on the **same DINOv3 frozen backbone**:

| Fusion method | AUC on T5 | Params (M) |
|---|---|---|
| Average pooling | 0.958 | 0 |
| **Max pooling** | **0.946** | 0 |
| Median pooling | 0.944 | 0 |
| LSTM | 0.903 | 5.5 |
| Transformer | 0.950 | 7.0 |
| **Query-based attention pooling** | **0.962** | 0.001 |

Gap from max pooling to query-based attention pooling: **+0.016 AUC**. The paper also states that,
on at least one earlier baseline setup, replacing average pooling with their adaptation gives
**ΔAUC = +0.11** with the same DINOv3 backbone.

This is medical-volume evidence that the readout choice moves the number on the same trunk.

## (b) 2.5D/3D comparison with learned depth aggregation vs max/mean pooling

### MRNet lineage

Bien et al. 2018, *MRNet* ([PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)),
uses an AlexNet-style 2D trunk per slice and **max pooling across slices** to obtain a
256-dimensional series vector. No learned aggregation is ablated in the published paper; the max
pool is the design choice.

### AnyMC3D (abdominal trauma / CT, comparable volumetric classification)

Same Table 6 above. The relevant comparison is:

- Max pooling (0.946) vs query-based attention pooling (0.962): **+0.016**
- Average pooling (0.958) vs query-based attention pooling (0.962): **+0.004**
- LSTM sequential fusion (0.903) is worse than all pooling methods, suggesting ordered modeling is
  not the win; the win is a **learned, permutation-invariant attention query** over slice embeddings.

The RSNA-Kaggle winning solutions summarized in the same paper use 2.5D stacks + LSTM/GRU or
attention-based MIL for slice fusion, consistent with the "learned aggregation beats simple pooling"
pattern, but they do not report a clean max-pooling ablation.

### KneeMRI-specific work

A Frontiers 2026 paper on MRNet-style ACL detection reports a full model at **0.919 ± 0.003 AUROC**
and notes that removing a multi-scale attention aggregator and reverting to mean pooling degrades
discrimination and calibration, but the exact delta is not given in the accessible text.

## (c) DINOv3 ViT-L distilled linear-probe classification vs DINOv2-L

DINOv3 authors did **not** release standard ImageNet-1k linear-probe heads for the smaller distilled
checkpoints; they report only ImageNet-ReAL top-1. Berreby et al. trained standard ImageNet-1k linear
probes for the distilled checkpoints ([GitHub](https://github.com/yberreby/dinov3-in1k-probes),
reported in [CanViT](https://arxiv.org/abs/2603.22570)):

| Model | IN1k val top-1 | IN-ReAL val top-1 |
|---|---|---|
| DINOv3 ViT-L/16 distilled | **87.44%** | 90.42% |
| DINOv2 ViT-L/14 (with registers) | ~86.7% | — |
| DINOv2 ViT-L/14 (no registers) | ~86.3% | — |

So on ImageNet-1k linear probing, DINOv3-L is **+0.7 to +1.1 percentage points** better than
DINOv2-L. It is not worse.

AnyMC3D Table 8 compares DINOv2 and DINOv3 as frozen 2D backbones for 3D abdominal-trauma
classification with their own adaptation (LoRA + query-based attention pooling):

| Task | DINOv2 AUC | DINOv3 AUC |
|---|---|---|
| T1 | 0.956 | 0.954 |
| T2 | 0.914 | 0.922 |
| T3 | 0.988 | 0.984 |
| T4 | 0.957 | 0.953 |

They are effectively tied. The AnyMC3D authors explicitly note that DINOv3's Gram-anchoring
improvement is for **dense/patch-level features**, not for image-level classification.

## Interpretation for our readings

1. **DINOv3-L is not globally weaker than DINOv2-L.** On ImageNet linear probe it is slightly
   better; on 3D medical classification with proper adaptation it is tied. Your measured
   DINOv3-L 0.7446 vs DINOv2-L 0.7852 (-0.0406) on the frozen knee-MRI ridge probe is therefore
   anomalous as a statement about trunk quality and consistent with a readout mismatch.

2. **The readout gap can be large.** In chest radiography the same trunk goes from chance to
   near-perfect AUC when the readout is made local. In volumetric tasks, learned attention pooling
   over slice embeddings can beat max/mean pooling by up to +0.11 AUC (reported) and +0.016 AUC
   (Table 6) on the same frozen 2D FM.

3. **Depth ordering is not the win.** AnyMC3D's query-based pooling is permutation-invariant and
   outperforms LSTM/Transformer sequential fusion. This matters because the project already has
   `DepthAttention`; the missing piece may be making it the **default** aggregation for the trunk
   probe rather than treating `max` as the canonical readout.

## Recommendation

The trunk lane should be re-read with a **patch-level or attention-based probe** on the frozen caches
you already hold (dinov2-large, v3large, ortholarge). The literature gives no reason to expect
DINOv3-L to underperform DINOv2-L on a fair readout, and several reasons to suspect the current
max-pooled ridge probe is the confound.
