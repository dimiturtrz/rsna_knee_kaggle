# 2026-09-15 — Knee MRI intensity normalisation, augmentation and sampling: literature mapped onto our recipe

Scope: web research only, no card. The question was what the published record says about
three input and training axes, and which untried ideas are worth card hours on OUR recipe:
DINOv2-small at 336 px, 2.5D slot tokens, labelwise pooling, slot dropout 0.40, 24 epochs
OneCycle, 12 labels, macro AUC. Our own code was deliberately NOT audited here; a parallel
audit covers that. KneeCoT is excluded.

Tags: [CLF] classifier endpoint, [SEG] segmentation endpoint, [RAD] radiomics (not deep
learning). Every number carries its source URL. Anything without one is marked **unverified**.

## Short answer

1. **Intensity normalisation is a domain-shift lever, not an in-distribution one.** The best
   deep-learning evidence is a segmentation study where standardisation helps mostly
   out-of-distribution. The one large classification number is radiomics, where Nyul
   changes texture features that a pretrained ViT does not compute. The knee test set is
   presumably drawn from the same distribution as train (our inference, not a source), so
   the expected gain is small.
2. **Knee MRI classification augmentation in the literature is almost entirely affine:**
   flip, rotate, translate. No paper found measures MRI-artefact augmentation (motion,
   ghosting, bias field, Gibbs) on a classifier AUC. TorchIO's evidence is [SEG]
   robustness only.
3. **Class-balanced sampling targets threshold metrics (F1, minority recall), not AUC.**
   AUC is invariant to class prior, so balanced sampling mainly changes gradient allocation.
   No source found measures it on multilabel macro AUC. Domain-balanced batching has an
   out-of-distribution AUC result on chest X-ray.
4. **The biggest published input-side number is still ROI localisation and cropping.** That
   axis is already in `research/2026-09-01_input_side_knee_mri_gains.md` and is not
   repeated here.

## 1. Intensity normalisation

| Source | Setting | Number | Mechanism |
|---|---|---|---|
| Sci Rep 2020, "Impact of preprocessing on radiomics" ([nature.com/articles/s41598-020-69298-z](https://www.nature.com/articles/s41598-020-69298-z), PMC7378556) [RAD] | Brain MRI glioma grade, first-order and texture radiomics | Balanced accuracy 0.67 with no normalisation, 0.82 Nyul, 0.79 WhiteStripe, 0.82 z-score | Handcrafted features read absolute intensity bins, so any standardisation removes scanner offset. Nyul also reshapes texture. |
| arXiv [2307.03827](https://arxiv.org/abs/2307.03827) [SEG] | Multi-centre FLAIR white-matter lesion segmentation. Nyul, z-score, WhiteStripe and IAMLAB compared. | In-distribution Dice: IAMLAB 0.78, ensemble 0.80. Out-of-distribution: large lesions 0.77 vs 0.71 original, medium 0.66 vs 0.61, small 0.53 vs 0.52. | The gain concentrates out of distribution: standardisation removes the inter-site intensity shift the network would otherwise have to generalise over. |
| ELNet, arXiv [2005.02706](https://arxiv.org/abs/2005.02706) [CLF] | MRNet knee, trained from scratch | ACL +0.004 AUC and abnormal +0.005 AUC over MRNet | Multi-slice normalisation layer plus BlurPool. The delta sits at our noise floor and is an architecture change, not a preprocessing one. |
| intensity-normalization package ([intensity-normalization.readthedocs.io](https://intensity-normalization.readthedocs.io)) | Tooling: Nyul, WhiteStripe, z-score, fuzzy C-means, KDE | No numbers | Reference implementation. Licence **unverified**; check the repo before use. |

**Not found:** a head-to-head of z-score vs percentile clip vs ImageNet mean/std for DINOv2 or
ViT fine-tuning on MRI with a classification AUC. Common practice (per-volume percentile clip
to [0, 1], then ImageNet stats) is folklore here: **unverified** as a measured optimum.

**N4 bias correction:** no classifier AUC delta found. Its benefit in the literature is [SEG]
and [RAD]. **Unverified** for classification.

## 2. Augmentation

| Source | Setting | Number | Mechanism |
|---|---|---|---|
| Bien et al. 2018, MRNet (PLOS Med, PMC6258509) [CLF] | Knee, 3 planes, AlexNet per plane | Abnormality AUC 0.937 (from our 2026-09-01 note). Augmentation: random rotation, shift, flip. | Affine invariance on 1,130 training exams. No augmentation ablation reported. |
| Azcona et al., arXiv [2010.01947](https://arxiv.org/abs/2010.01947), code [github.com/dazcona/mrnet](https://github.com/dazcona/mrnet) [CLF] | MRNet, ResNet18 transfer | 93.4% validation AUC. Transfer learning and tuned augmentation named as the crucial factors. | Same affine family. The individual augmentation delta is not isolated in the abstract (**unverified** per-op). |
| Frontiers Bioeng 2025, knee injury detection ([doi 10.3389/fbioe.2025.1590962](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full)) [CLF] | Knee MRI | Horizontal flip p=0.5 used. No ablation found. | Mirror invariance. See the laterality note below. |
| arXiv [2006.01693](https://arxiv.org/abs/2006.01693) [CLF] | Prostate DWI, 217/102/95 patients, 2D slice classifier | Rotation was the best single augmentation, slice-level AUC 0.85. Flip, crop and translate numbers are not in the abstract. | Rotation covers real patient-positioning variance. Flip adds anatomically implausible views. |
| Steiner et al., "How to train your ViT", arXiv [2106.10270](https://arxiv.org/abs/2106.10270) [CLF, natural images] | ViT fine-tuning sweep: RandAugment m∈{0,10,15,20}, mixup α∈{0,0.2,0.5,0.8} | On small datasets the right AugReg combination matches roughly a 10x data increase. | ViTs have a weaker inductive bias, so they lean more on augmentation and regularisation. Our 0.40 slot dropout is one such regulariser. |
| TorchIO, Pérez-García et al. ([PMC8542803](https://pmc.ncbi.nlm.nih.gov/articles/PMC8542803/)) [SEG] | RandomMotion, RandomGhosting, RandomBiasField, RandomSpike, RandomAnisotropy | No classifier AUC. Reported as improving segmentation generalisability. | Simulates k-space and scanner artefacts, so robustness shows up under acquisition shift. |
| Brain MRI multi-scale framework ([PMC13298221](https://pmc.ncbi.nlm.nih.gov/articles/PMC13298221/)) [CLF] | 4-class brain MRI, MixUp/CutMix plus focal loss plus label smoothing | Macro AUC 0.9993 with an ablation. The MixUp/CutMix share was not extracted (**unverified**). | A ceiling-saturated task, so it does not transfer to a ~0.94 multilabel problem. |
| Flow-Mixup, arXiv [2102.08148](https://arxiv.org/abs/2102.08148) [CLF] | ChestX-ray14 multilabel with corrupted labels | Reported AUC gain over mixup. Exact delta not extracted (**unverified**). | Mixup in feature space regularises against label noise. Label noise is our documented bottleneck (knee-z4yfe, knee-wko44). |
| Balanced-MixUp (MICCAI 2021, [ResearchGate 354792016](https://www.researchgate.net/publication/354792016_Balanced-MixUp_for_Highly_Imbalanced_Medical_Image_Classification)) [CLF] | Imbalanced retinal and GI images | Improved minority metrics. Delta **unverified** (abstract only). | Mixes class-balanced and instance samples, so it is both a sampler and a regulariser. |
| RSNA 2024 Lumbar 2nd place, B. Artley ([github.com/brendanartley/RSNA-2024-Competition](https://github.com/brendanartley/RSNA-2024-Competition)) [CLF] | Sagittal-only two-stage pipeline, 2.5D | README: "data augmentation tricks, pseudo labeling, TTA". The per-trick delta is not in the README (**unverified**). Licence **unverified**. | Staged crop then classify, the same localisation lever as CoPAS. |
| RSNA 2024 Lumbar 1st place writeup (Kaggle) | — | Not retrievable: the page is JS-rendered and WebFetch returned nothing. **Unverified.** | — |

**Laterality and flip (mechanism argument, no source measured it).** A left-right flip of a
right knee produces a plausible left knee, so it is label-preserving for the whole-knee label
set. It is NOT label-preserving if any of our 12 labels encode medial vs lateral (for example
medial vs lateral meniscus or MCL vs LCL) while the flip is applied across the plane that
distinguishes them. Sagittal slice ORDER reversal is the actual medial-lateral axis. An
in-plane horizontal flip of a sagittal image mirrors anterior-posterior, which is
anatomically implausible but label-safe. Which flips are safe depends on plane and label
map, which is for the code audit to check.

**Not found with a measured classifier delta on MRI:** elastic deformation, gamma/contrast
jitter, Gibbs ringing, slice-order jitter, slice dropout (our slot dropout 0.40 is an
in-house instance of it), resolution/anisotropy jitter. All **unverified** as gains.

## 3. Sampling

| Source | Setting | Number | Mechanism |
|---|---|---|---|
| arXiv [2112.13734](https://arxiv.org/abs/2112.13734) [CLF] | Chest X-ray, multi-domain training | Domain-balanced batching gave better out-of-distribution AUC than random batching. Delta not extracted (**unverified** magnitude). | Each batch sees every source, so the network is not optimised toward the majority site. |
| arXiv [2605.02328](https://arxiv.org/abs/2605.02328) [CLF] | Chest X-ray CBAM DenseNet121 | Mean AUC 0.868. Cites class-balanced sampling and CutMix improving minority F1 in histopathology (secondary claim). | Balancing moves the operating point and minority recall. AUC is prior-invariant, so the AUC effect runs only through representation learning. |

**Mechanism note (our argument, not a source).** In multilabel training, a sample drawn to
balance one label unbalances the other eleven, so "class-balanced" is ill-defined for 12
co-occurring labels. Per-label loss weighting (pos_weight) is the clean analogue.
Curriculum learning and hard-negative mining: no MRI classifier AUC evidence found
(**unverified**).

## 4. Mapped onto our recipe

| Axis | Ours | Literature state | Verdict |
|---|---|---|---|
| Intensity normalisation | Audited separately | Gain is mostly out of distribution. [RAD] numbers do not transfer. | Low expected in-distribution gain. Worth it only if the test set spans scanners the train set lacks. |
| Affine augmentation | Audited separately | The standard knee-MRI kit. Rotation is the best single op in prostate DWI. | Check which flips are label-safe for our 12 labels and planes. |
| Artefact augmentation (TorchIO) | Not in recipe (as far as this note knows) | [SEG] robustness only | Instrument-grade: in distribution, likely ≤ noise. |
| Slice dropout | Slot dropout 0.40 | In-house version. No published classifier number. | Already spent. Sweeping upward is filed as knee-1zaem. |
| Mixup / feature mixup | Not in recipe (assumed) | ViT AugReg evidence plus label-noise robustness | Mechanism matches our label-noise bottleneck. |
| Balanced sampling | Uniform (assumed) | Targets F1, not AUC. Ill-defined for multilabel. | Low expected AUC gain. |
| Domain-balanced batching | Unknown | Out-of-distribution AUC on chest X-ray | Only if scanner or site metadata exist and the board spans them. |

## 5. Top 5 untried ideas, ranked by gain per card hour (fork's judgment; numbers are priors, not measurements)

A 24-epoch arm is priced at ~2-4 h (**unverified** for this exact recipe; check the ledger).
The P values are this fork's subjective priors against the 0.015 bar.

1. **Feature-space mixup on slot tokens (Flow-Mixup / manifold mixup), α ~0.2.**
   PRICE ~3 h. P(clears 0.015) ~0.10.
   Mechanism: mixup interpolates labels as well as inputs, which softens confidently wrong
   parser labels. It also fits Steiner's finding that ViTs lean on regularisation.
   GPU-free prep: yes (code plus unit test).
2. **Label-safe flip and rotation audit.** Remove any flip that swaps a medial/lateral label,
   and add ±10-15° in-plane rotation if absent.
   PRICE ~3 h. P ~0.08; higher if the audit finds a label-corrupting flip, in which case it is a bug fix.
   GPU-free prep: yes (read the label map and aug code).
3. **Per-label pos_weight in place of any sampler change.** This is the multilabel-correct
   form of balancing.
   PRICE ~3 h, warm-start possible because the input is unchanged. P ~0.05.
   GPU-free prep: yes (compute prevalence off the label table).
4. **TorchIO bias-field plus mild motion/ghosting at p ~0.3 each** (2.5D: apply per volume
   before slicing).
   PRICE ~3-4 h, plus CPU decode cost. P ~0.04 in distribution. Its value is as an
   INSTRUMENT: it tests whether acquisition-shift robustness moves the board where the
   holdout cannot see it.
   GPU-free prep: yes (transform plus smoke on CPU).
5. **Nyul histogram standardisation per sequence** (landmarks fitted on train only) vs the
   current normalisation.
   PRICE ~4 h, since it needs a re-decode on CPU and then an arm. P ~0.03. This is also an
   instrument, for the scanner axis already noted in interpretations
   (2026-09-05_the-scanner-axis-exists-in-the-data-and-not-in-the-ruler).
   GPU-free prep: yes (fit landmarks, write the cache).

None of the five has a published knee-classifier number that clears 0.015. The largest
documented input-side lever remains ROI localisation (see the 2026-09-01 note).

## Licences

- TorchIO: Apache-2.0. Stated from memory of the repo, so **verify** at github.com/fepegar/torchio before depending on it.
- intensity-normalization (jcreinhold): **unverified**.
- dazcona/mrnet, brendanartley/RSNA-2024-Competition: **unverified**. Treat as reference-only until the LICENSE file is read.
- No weights from any of these are proposed. All five ideas are methods reimplemented in our code.
- Nothing here is KneeCoT-, fastMRI-, SKM-TEA- or Rijeka-derived.

## Unverified list

- The per-op deltas for MRNet, Azcona, Artley and Flow-Mixup. None of these abstracts or
  READMEs isolates them.
- The RSNA 2024 lumbar 1st-place writeup: not retrievable.
- The PMC13298221 ablation share for MixUp/CutMix.
- The 2112.13734 AUC magnitude.
- All licences except the TorchIO recollection.
- The 24-epoch arm price and all P(clears) values: fork priors.
