# Q-EO — Can a medical-pretrained trunk earn its seat in an ensemble solely through corpus diversity?

2026-09-04

## Bottom line for Claude

- **No published paper isolates a corpus-only ensemble seat gain** with the exact control you ran: same architecture, same SSL objective, same recipe, only the pretraining corpus changed. The experiment you just finished is cleaner than anything in the literature.
- The closest evidence (Khan et al., ML4H 2023) ensembles natural vs medical pretraining domains, but the reported gains **mix architecture family and SSL objective**; the one sub-ensemble that is same-arch/same-SSL (ViT masked, MAE + Medical MAE) shows a combined AUC of 81.0, while the natural-only member is 79.3. That implies the medical member must be stronger, not a weaker diversity seat.
- The decorrelation story you need for an ensemble gain is **directly contradicted** by Juodelyte et al. (TMLR 2023): ImageNet and RadImageNet representations diverge after fine-tuning, yet their **predictions stay similar** — i.e., low error diversity.
- Recent replication work (Juodelyte; Frees et al.) and a modality-matching study (Krones 2026) all point to the same explanation for your solo cost: **RadImageNet's 2D radiology-slice corpus is the wrong modality for volumetric knee MRI**, not merely too small.
- The smallest corpus shown to beat a matched natural-image trunk is **~112k chest X-rays** for MAE on chest X-ray (Khan Table 2). That result does not transfer to DINO self-distillation on knee MRI; for that combination the threshold is unknown and, on current evidence, **above 1.35M images**.

## Q-EO #1 — Has anyone measured corpus-diversity as an ensemble lever?

### Khan et al. is the closest published case

Khan et al., "How Fair are Medical Imaging Foundation Models?" (ML4H / PMLR 225:217–231, 2023) [[PDF](https://proceedings.mlr.press/v225/khan23a/khan23a.pdf)] constructs ensembles stratified by pretraining domain and by architecture-SSL combination on CheXpert. Their Table 3 reports:

| Ensemble | AUC ↑ | Sex FG ↓ | Race FG ↓ |
|---|---:|---:|---:|
| ViT (Masked) — MAE + Medical MAE | 81.0 | 1.21 | 2.82 |
| ViT (Contrastive) — MoCov3 + Medical MoCov3 | 80.3 | 1.13 | 2.90 |
| ResNet (Contrastive) — BiT + REMEDIS | 81.6 | 1.15 | 2.95 |
| Natural Imaging — MAE + MoCov3 + BiT | 80.4 | **0.78** | 2.94 |
| Medical Imaging — Med MAE + Med MoCov3 + REMEDIS | **81.6** | 1.39 | 3.12 |
| All Foundation Models | 81.4 | 1.12 | 2.84 |

The "ViT (Masked)" and "ViT (Contrastive)" rows are the only published ensembles that hold architecture and SSL objective fixed while varying only the pretraining corpus (ImageNet vs. CheXpert/ChestXray14/MIMIC-CXR). That is the design you are asking about.

**But the paper does not report the seat gain.** It reports the ensemble AUC, not `ensemble_AUC − max(solo_AUC)`. We can only bound it:
- Table 6 gives MAE (natural, ViT-B, masked) fine-tuned AUC = **79.3**.
- Section 4.1.1 states medical pretraining yields a "greater boost" than natural pretraining on chest X-ray, so Medical MAE solo is presumably ≥79.3.
- Therefore the ViT (Masked) ensemble AUC of 81.0 is likely close to the **average or slightly above the stronger member**, not a rescue of a weaker member.

The "Natural Imaging" vs "Medical Imaging" ensembles (80.4 vs 81.6) mix architecture, SSL method, and corpus, so they cannot isolate the corpus effect.

### No paper reports the exact "seat gain" metric

A direct answer to your question would require a table like:

```
member_1 (natural,  arch X, obj Y)  solo AUC = a
member_2 (medical, arch X, obj Y)  solo AUC = b  (b < a)
ensemble of {1,2}                   AUC = c       (c > a)
```

I found no such table. Khan's paper gives `a` and `c` for one of the two members but not `b`, and the other rows confound architecture/SSL. The literature on ensemble diversity in medical imaging (Q-EL, Q-EM) reports architecture-family and seed/augmentation diversity, not corpus-only diversity.

### The decorrelation story is weak

Juodelyte et al., "Revisiting Hidden Representations in Transfer Learning for Medical Imaging" (TMLR, 2023) [[arXiv:2302.08272](https://arxiv.org/abs/2302.08272)] is the paper most relevant to whether corpus diversity buys prediction diversity. They fine-tune ImageNet- and RadImageNet-pretrained ResNet50 on the same seven medical targets and compare both representations (CCA) and predictions. Their findings:

1. ImageNet tends to outperform RadImageNet on 6/7 targets (Table 2). Only knee is slightly better with RadImageNet, and the two are comparable.
2. ImageNet and RadImageNet **representations diverge after fine-tuning** on the same target (Figure 3b).
3. Despite that divergence, **predictions remain similar** — "more similar than expected for independent predictions" (Section 5.3, Figure 5).

This is the exact combination that undercuts a corpus-diversity ensemble gain: the two trunks learn different features but make the same mistakes. If that holds for DINO self-distillation as well, adding `radiodino_corpus` to a DINOv2/DINOv3 ensemble would not produce a large decorrelation dividend.

## Q-EO #2 — Does domain-pretraining literature predict solo cost with ensemble gain?

### The DAPT/medical-pretraining literature predicts solo *gain*, not solo cost

The standard claim in domain-adaptive pretraining (Gururangan et al.; REMEDIS; Medical MAE; Radio DINO) is that in-domain pretraining improves **solo** downstream performance. Khan Section 4.1.1 explicitly says medical pretraining "yields a greater boost in Chest X-ray classification performance compared to natural imaging pre-training." Radio DINO (Zedda et al., 2025) reports gains on radiology benchmarks.

Your measured solo **cost** (−2.4 to −2.9 points) is therefore not predicted by the main medical-pretraining narrative. It is predicted by recent replication work that finds ImageNet often wins:

- **Juodelyte et al.** replicate four of Mei et al.'s RadImageNet experiments and find ImageNet wins on most targets.
- **Frees et al., "Towards Optimal Convolutional Transfer Learning Architectures for Breast Lesion Classification and ACL Tear Detection"** (2025) [[arXiv:2508.17567](https://arxiv.org/abs/2508.17567)] could not replicate Mei et al.'s RadImageNet superiority. Their best ACL and breast models were **statistically significantly better with ImageNet pretraining**.
- **Krones, "From pre-training to downstream performance: Does domain-specific pre-training make sense?"** (2026) [[arXiv:2605.08819](https://arxiv.org/abs/2605.08819)] finds that **only pretraining data closely matching the target modality improves downstream performance**; mismatched medical pretraining can actually hurt versus ImageNet.

### The likely explanation for your cost is modality mismatch, not capacity

Krones' Table 2 is the cleanest analogue. Fine-tuning on chest X-ray (NIH), they report:

| Exp. | Init | Pretrain strategy | Pretrain domain | NIH 14-class AUC |
|---|---|---|---|---:|
| 2.2 | ImageNet | — | — | **0.7780** |
| 3.5 | ImageNet | self-supervised | OCT (retina) | 0.7729 |
| 3.6 | ImageNet | self-supervised | NIH chest X-ray | **0.8020** |
| 3.7 | random | self-supervised | CT chest | 0.7565 |
| 3.8 | ImageNet | self-supervised | CT chest | 0.7605 |

Key observation: self-supervised pretraining on a **matched** modality (chest X-ray) helps (+0.024 over ImageNet), but pretraining on a **mismatched** medical modality (OCT or CT) **hurts** versus the ImageNet baseline (−0.005 to −0.017). Krones explicitly writes: "using a healthcare modality not closely aligned with the downstream task can potentially negatively impact performance, compared to just using a larger, domain-unspecific dataset such as ImageNet."

RadImageNet 1.35M is a mixture of CXR, CT, MRI, and ultrasound 2D slices. Knee MRI in your task is volumetric, T1/T2/PD-weighted, with pathology in specific meniscal/ligamentous/cartilage subregions. The corpus is medical but **not modality-matched**. Your measured cost is consistent with Krones' mismatch finding, not a contradiction of it.

The small-width result (`radiodino_s16` −0.0394 vs `native224` 0.8502) is especially telling: if the gap were a capacity/undertraining artefact, widening to base should close it. It did not (−0.0241 at base). That is the signature of a corpus-quality problem, not a corpus-size problem.

### Ensemble gain is not the rescue the literature promises

There is no published result saying "a medical corpus that hurts solo will help in ensemble." The only ensemble-domain paper (Khan) shows medical-pretrained models **already strong solo** being combined for fairness/performance balance. Juodelyte's prediction-similarity result suggests the error diversity between natural and medical trunks is low, so the ensemble gain from corpus diversity should be small.

## Q-EO #3 — What is the smallest in-domain corpus shown to beat a natural trunk?

### For MAE masked SSL on chest X-ray: ~112k images

Khan Table 2 ("Impact of increasing pre-training data") pretrains MAE on chest X-ray corpora of increasing size and evaluates on CheXpert:

| Pretraining data | Classif. AUC |
|---|---:|
| ChestXray14 (~112k images) | 79.97 |
| CheXpert (~224k images) | 80.98 |
| ChestXray14 + CheXpert (~336k images) | 81.38 |

The natural baseline in their setup is MAE on ImageNet-1K, fine-tuned AUC = **79.3** (Table 6). So **~112k chest X-rays** is enough for MAE to beat the natural trunk on a matched-modality chest-X-ray task.

### For DINO self-distillation on knee MRI: threshold is unknown and >1.35M

Your `radiodino_corpus` result gives one point on this curve: 1.35M medical 2D slices under DINO self-distillation loses to LVD-142M/LVD-1689M natural images by 2.4–2.9 points on knee MRI.

No published study gives the other points for this combination. The relevant scaling work is:

- **Lee et al., "Knowledge Transfer Scaling Laws for 3D Medical Imaging"** (2026) [[arXiv:2605.06859](https://arxiv.org/abs/2605.06859)] studies MAE pretraining on six 3D medical domains. They find cross-domain transfer is **strongly asymmetric** and that some domains are "hubs" while others are "islands" requiring direct investment. At 100k pretraining volumes, transfer-aware allocation beats uniform/data-proportional by up to 58% on reconstruction loss.

Lee's framework is the right conceptual model: knee MRI may be an "island" domain that does not benefit much from mixed-modality 2D radiology pretraining and needs direct volumetric knee-MRI investment. But the paper does not report a threshold at which in-domain knee-MRI DINO beats natural LVD.

### Therefore

- **If the question is "can any in-domain corpus beat ImageNet?"** the answer is yes, and the smallest published example is ~112k matched-modality images for MAE.
- **If the question is "can a 1.35M mixed-modality medical corpus beat LVD-142M/1689M under DINO self-distillation on knee MRI?"** the answer is no, and the threshold is above that.
- **If the question is "how big does a knee-MRI-native corpus need to be?"** the answer is not in the literature; it would require running the scaling experiment.

## Implications for `radiodino_corpus` as a seat

1. **There is no published precedent for a corpus-only weak member adding ensemble value.** The closest paper (Khan) combines two corpus-diverse members that are each strong solo.
2. **The error-diversity argument is weak.** Juodelyte shows ImageNet and RadImageNet make similar mistakes despite different features. Your DINOv2/DINOv3 vs RadioDino members may behave the same way.
3. **The solo cost is explained by modality mismatch, not by corpus size alone.** Krones shows mismatched medical pretraining hurts; the small-width leg of your experiment shows the gap does not close with width.
4. **A card-free test is possible and mandatory.** Before mounting `radiodino_corpus` in `honest12`, run a marginal-contribution gate on the 58-study gold set or a held-out block: measure `AUC(ensemble + radiodino_corpus) − AUC(ensemble)` and require it to beat a pre-registered floor (e.g. >+0.0005) *and* show per-label error correlation <0.95 with the existing ensemble.
5. **MRI-CORE should be scoped as modality-matched volumetric data, not more 2D slices.** If you want to test the corpus-size threshold, the corpus needs to be knee-MRI volumes (or at least matched 3D musculoskeletal MRI), not more RadImageNet-style 2D slices. Otherwise you are testing Krones' mismatch hypothesis, not a scaling threshold.

## Sources

- Khan et al., "How Fair are Medical Imaging Foundation Models?", ML4H / PMLR 225:217–231, 2023 [[PDF](https://proceedings.mlr.press/v225/khan23a/khan23a.pdf)]
- Juodelyte et al., "Revisiting Hidden Representations in Transfer Learning for Medical Imaging", TMLR 2023 [[arXiv:2302.08272](https://arxiv.org/abs/2302.08272)]
- Frees et al., "Towards Optimal Convolutional Transfer Learning Architectures for Breast Lesion Classification and ACL Tear Detection", 2025 [[arXiv:2508.17567](https://arxiv.org/abs/2508.17567)]
- Krones, "From pre-training to downstream performance: Does domain-specific pre-training make sense?", arXiv:2605.08819, 2026 [[arXiv:2605.08819](https://arxiv.org/abs/2605.08819)]
- Lee et al., "Knowledge Transfer Scaling Laws for 3D Medical Imaging", arXiv:2605.06859, 2026 [[arXiv:2605.06859](https://arxiv.org/abs/2605.06859)]
- Mei et al., "RadImageNet: A Large-scale Radiologic Dataset for Training Deep Learning Models", Radiology 2022 (PMC9530758)
