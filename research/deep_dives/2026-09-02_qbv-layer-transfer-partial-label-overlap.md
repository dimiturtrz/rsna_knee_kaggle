# Q-BV — Layer-wise transfer and partial-label overlap: does MRNet pretrain help all 12 RSNA labels or only the 3 it knows?

Claude's question (2026-09-02): for a domain-matched *supervised* pretrain → fine-tune on a
different labelled corpus, what does the literature say about which layers carry the transfer
and how much of the gain survives when the label sets only partially overlap?

Context: MRNet provides 1,370 knee MRI exams labelled for abnormal / ACL tear / meniscal tear
(3 of our 12 RSNA findings). The alternative to the already-tried corpus-concatenation lane is
to pretrain the encoder on MRNet and then fine-tune on RSNA. The decision hinges on whether
the gain lives in the representation (helps all twelve labels) or in the head/last-block
specialisation (helps only the three overlapping labels, capping the mean-AUC gain at roughly
+0.0075).

## 1. The classic layer-transferability finding

Yosinski et al. (NeurIPS 2014, *How transferable are features in deep neural networks?*)
measured transferability layer by layer on ImageNet → random splits of ImageNet classes:

- **Lower layers are general; upper layers are task-specific.** Features transition from
  general to specific as depth increases.
- **Transferability drops with depth.** The first two layers transfer almost perfectly; the
  drop grows for deeper layers.
- **Fine-tuning rescues fragile co-adaptation.** When source and target tasks differ, frozen
  transfer of middle layers underperforms because the features are co-adapted; fine-tuning
  those layers together recovers most of the gain.

Source: [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)

This is the baseline expectation: even when labels differ, lower and middle layers carry
shared visual primitives, while the head and deepest blocks must be retasked.

## 2. Medical imaging: the gain is concentrated in low-level / early features

Raghu et al. (NeurIPS 2019, *Transfusion: Understanding transfer learning for medical imaging*)
showed that transfer from ImageNet to large medical datasets (CheXpert, retinal fundus)
provided surprisingly small performance benefits. Their analysis found that much of the small
benefit came from low-level feature statistics and weight scalings rather than sophisticated
high-level feature reuse. Lightweight randomly-initialised models could match ImageNet-pretrained
ones on the large target datasets.

Source: [arXiv:1902.07208](https://arxiv.org/abs/1902.07208)

Neyshabur, Sedghi & Zhang (NeurIPS 2020, *What is being transferred in transfer learning?*)
separated feature reuse from low-level statistics by training on block-shuffled images. They
found that **some of the benefit of transfer learning comes from low-level statistics**, not
just feature reuse, and that pretrained weights keep the model in the same loss-landscape basin.

Source: [arXiv:2008.11687](https://arxiv.org/abs/2008.11687)

Matsoukas et al. (CVPR 2022, *What makes transfer learning work for medical images*) added the
layer-wise picture specifically for ViTs and CNNs:

- **ViTs depend strongly on feature reuse**, and the reuse is **concentrated in early layers**.
- **CNNs reuse features more uniformly throughout the network**.
- The benefit of transfer increases with **smaller target datasets** and **smaller domain
  distance** between source and target.

Source: [arXiv:2203.01825](https://arxiv.org/abs/2203.01825)

For MRNet → RSNA, the source and target are domain-matched (both are clinical knee MRI) and the
target dataset is small (~4,400 studies). Both factors push toward positive transfer. The
mechanism, however, is expected to be low-level / early-to-mid feature reuse, not late-block
specialisation.

## 3. Scale changes which layers are reusable

Azizi et al. (2021, *Supervised transfer learning at scale for medical imaging*) trained BiT
models on JFT-300M and transferred to chest X-ray, mammography and dermatology. Their CKA and
weight-movement analysis showed that with larger pretraining datasets the model is able to
**reuse deeper features** on the medical target — earlier layers move more during fine-tuning,
later layers move less. This is the opposite of the small-ImageNet regime where only early
features are reusable.

Source: [arXiv:2101.05913](https://arxiv.org/abs/2101.05913)

The implication for MRNet is mixed. MRNet is domain-matched but tiny (1,370 exams) compared with
the multi-million-image pretraining in Azizi et al. It is unlikely to learn reusable *deep*
features. The transferable value is therefore expected to remain in low-level and early/mid
knee-MRI statistics — Gabor-like textures, contrast patterns, slice geometry — rather than in
high-level pathology detectors.

## 4. Partial label overlap: the multi-task learning view

Feng & Chen (2022, *Learning multi-tasks with inconsistent labels by using auxiliary big task*)
studied exactly the partially-overlapping-labels setting: each task has a small dataset and a
label set that only partially overlaps with the others. Their DAMTL framework uses a large
auxiliary task whose label space covers all target labels, plus adaptive pruning to construct
task-specific sub-networks. The key insight is that **a shared trunk is still valuable even when
individual heads differ**, because co-occurrence and shared visual structure let the trunk learn
about labels it has never explicitly seen.

Source: [arXiv:2201.02305](https://arxiv.org/abs/2201.02305)

Mapping this to MRNet → RSNA: MRNet does not cover OA, synovitis, Baker's, fracture or contusion,
but a trunk trained on MRNet still learns knee anatomy. That anatomy is a prerequisite for *all*
twelve RSNA labels. The question is not whether the trunk can learn about unseen labels — it
can, indirectly — but how much the extra 1,370 exams improve on the DINOv2/DINOv3 trunk that was
already pretrained on far more images.

## 5. What the project has already measured

Two results bound the answer:

1. **OrthoFoundation-L was declined.** It was continued-pretrained on 1.25M knee images (MRI +
   X-ray) with labels overlapping our task, yet its frozen probe (0.7387) was weaker than its own
   DINOv3-L init (0.7446), and the fitted member (0.8452) lost to the DINOv2-base control
   (0.8502). Knee-specific supervised continuation can **subtract** signal, not just add it.
   Source: `research/deep_dives/2026-09-02_external_data_and_foundation_model_status_quo.md`.

2. **The merged-corpus seat axis closed flat/down.** Concatenating external studies into the
   fine-tuning cache produced at best +0.006 and at six external seats -0.004. External labels
   alone are not a large lever.

These two results do not directly answer Q-BV, but they set priors. If 1.25M knee images with
multi-label supervision can hurt overall, 1,370 MRNet images with only three labels is a much
weaker bet.

## 6. Synthesis

| Mechanism | Literature expectation | Affected RSNA labels | Plausible mean-AUC contribution |
|---|---|---|---|
| Low-level statistics / early-layer feature reuse | Positive, especially for ViTs and small target data | All 12 (anatomy/texture primitives) | Small; hard to separate from DINOv2 init |
| Mid-level feature reuse | Positive if source/target domain close | All 12, but strongest for labels that share local patterns (ACL, meniscus, effusion) | Modest |
| Late-block / head specialisation | Task-specific; must be retasked | Mostly the 3 overlapping labels | Capped at ~+0.0075 on mean AUC |
| Catastrophic forgetting of strong natural-image init | Real risk, shown by OrthoFoundation-L | Can hurt all 12 | Negative if pretrain overfits MRNet's label distribution |

**Answer to Claude's question.** The literature says the gain from a domain-matched supervised
pretrain with partial label overlap comes **primarily from representation (lower/mid layers), not
from head/last-block specialisation**. The head and deepest blocks are task-specific and must be
relearned for the 12-label task. Therefore, in principle, MRNet pretraining could help labels it
has never seen, because all twelve labels share knee anatomy and imaging statistics.

**However**, the magnitude is bounded by two things:

1. **MRNet is too small to learn deep reusable features.** Its 1,370 exams can refine low-level
   and some mid-level knee-MRI statistics, but not the high-level pathology detectors that would
   dominate the 12-label task. Azizi et al. needed millions of images and large models to reuse
   deep features.

2. **The starting point is already strong.** DINOv2/DINOv3 provide excellent general visual
   features. The marginal value of MRNet's domain-matched low-level statistics over that init is
   likely small, and the OrthoFoundation-L decline shows that knee-specific continuation can
   overwrite useful general features.

So the representation argument wins on mechanism, but the **gain is unlikely to be large enough
to justify a dedicated pretraining arm** unless the experiment is designed to test it cheaply
(e.g. a small dense arm with LP-FT, L2-SP, and heavy early-layer regularisation to prevent
forgetting). The per-label cap of ~+0.0075 for the three overlapping labels is a reasonable
pessimistic floor; a broader but still modest gain across all twelve is the optimistic ceiling.
