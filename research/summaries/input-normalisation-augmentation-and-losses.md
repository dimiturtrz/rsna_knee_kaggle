# Input normalisation, augmentation and losses — topic summary

**Current verdict (2026-09-25).** Slot dropout at 0.40 is the only regulariser that carries transfer. Every other
lever in this class is closed on our own measurement or has an in-distribution prior far under 1.5%. Closed:
augmentation, mixup, focal, AUC-margin, per-label weighting, ordinal loss, crop/FOV/padding, resolution
[AXES:123,138-139,107-112][axes]. Intensity normalisation and acquisition-physics augmentation are
*domain-shift* levers. The holdout cannot see them by construction, so they are priced on `decode_cost` and the
board, never on holdout AUC [AXES:126][axes] [norm] [toll]. Ghosting is the one open point: +0.0057, directional.

Tags: **[M]** measured by us · **[L]** literature · **[U]** unverified / asserted / figure-locked / derivation.
Rulers are quoted where named. An unmarked number of ours is midpoint (knee-f9aao).

---

## 1. Intensity normalisation

- In the literature, normalisation's gain concentrates out of distribution [L] [norm]:
  - FLAIR WML segmentation, OOD Dice: large lesions 0.77 vs 0.71, medium 0.66 vs 0.61, small 0.53 vs 0.52.
  - In-distribution: IAMLAB 0.78, ensemble 0.80.
- The only large classification number is radiomics [L] [norm]:
  - Balanced accuracy 0.67 without normalisation, vs 0.82 Nyul / 0.79 WhiteStripe / 0.82 z-score.
  - The mechanism is hand-crafted intensity bins, which a pretrained ViT does not compute. It does not transfer.
- ELNet (knee, MRNet): ACL +0.004, abnormal +0.005 [L] [norm]. The delta sits at our noise floor and comes from
  an architecture change (multi-slice norm + BlurPool).
- No head-to-head exists of z-score vs percentile clip vs ImageNet stats for ViT fine-tuning on MRI
  classification [L-null] [norm]. "Percentile clip then ImageNet stats" is folklore [U].
- N4 bias correction has no classifier AUC evidence; its benefit is SEG/RAD only [L-null] [norm] [toll].
- Albert 2023, rectal MRI, 6 centres [L] [toll]:
  - z-score was significantly worse than percentile+HM.
  - The normaliser choice buys ~0.01–0.03 AUC.
  - No normaliser repaired the cross-centre toll: Dworak collapses to ~0.52–0.58.
  - This matches our flat toll: −0.0336, sd 0.0038, n=11, 909-study holdout, midpoint [M] [toll] [crop].
- Large published deltas buy back normalisation we already do [L] [toll]:
  - DISARM++ +0.32 has a raw baseline. The verifiable figure is 0.6284→0.9459; the "0.857→0.899" quote was a
    misquote and is withdrawn.
  - Saponaro +0.09 is ComBat on FreeSurfer features, not pixels.
- OrthoFoundation-L pretrains with ImageNet mean/std, (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225) [M-code]
  [ortho]:
  - no per-volume normalisation
  - 224 px, patch 16
  - 4 register tokens
  - the released weights are the student
  - Feeding it 336 px interpolates its position embeddings.
- A BN trunk's "frozen" statistics drift in `train()` mode: `requires_grad=False` does not freeze running stats
  [L] [bn].
  - RadImageNet 0.6716 ran with `num_batches_tracked = 226711`, i.e. knee-MRI statistics. That result is void as
    a transfer measurement [M] [bn].
  - Fixes: `.eval()` the backbone, `norm_eval=True`, or AdaBN [bn].
- Proposed Nyul per sequence: PRICE ~4 h, P ~0.03, instrument-grade (for the scanner axis) [U-prior] [norm].
  No run is recorded in these sources.

## 2. Crop extent, FOV, padding, resolution, patch size

- CoPAS ROI crop: +0.056 mean AUC (0.725→0.781) on a single MRNet model [L] [ig] [ceil].
  - This is the largest published input-side number.
  - **It does not transfer:** it was bought against a no-crop baseline, and we crop at a physical 130 mm
    [M] [ig] [AXES:86][axes].
  - Only 0.43% of series are narrower than that crop [M] [ig].
- ROI crop/upsample: CLOSED, "the crop costs, and the upsample was exactly half of why" [M] [AXES:87][axes].
- Crop centre: CLOSED [AXES:85][axes].
- FOV/band width: CLOSED, MCL is why [AXES:84][axes].
- FOV jitter: NULL [AXES:88][axes].
- Padding mode: refuted 3/3; EXTENT withdrawn for want of a matched control [AXES:89][axes].
- Every published spacing/extent toll is pre-physical-normalisation or a SEG endpoint. Comparability is NO in both
  directions [L] [crop]:
  - Xie 2025 crops 256 px pixel-grid.
  - Consispace/PVS are SEG.
  - CARVE −0.125 is CT depth shift.
- Correction: D266's "mm-already-paid" framing is retracted. The live axis is extent WIDTH inside a 130 mm
  physical decode [crop].
- Extent direction:
  - Chang 2019: a narrower intercondylar FOV won in-domain (AUC 0.97) [L] [crop].
  - Our n=1 wider-field reading: `crop160_span` −0.0265 vs pool −0.0336 (+1.87 sd, midpoint) [M] [crop].
- Pad-as-site-cue: every component is measured separately, but the composition was never priced on a classifier
  [L-null] [crop]. Components: Weng border shortcut; Alsallakh pad type; Islam zero-pad position.
- Resolution:
  - No clean knee curve exists [L-null] [ig].
  - MRNet 256 px ACL ~0.97; CoPAS 224 [L] [ig].
  - Competitor 192 vs 288 px is confounded with crop [ig].
  - Ours: 224 costs 0.0049 solo and buys 3.2× throughput; 336→392 CLOSED; 420 UNREAD [M] [AXES:74-76][axes].
- Patch size: Dehghan's +12.78% (2D) and +23.78% (3D) are on 28 px MedMNIST. Their patch 2/1 equals our 16/8 on
  224 px only in token count [L] [pat].
  - Their schedule is identical across patch sizes: 80 epochs, AdamW 1e-4.
  - So our `radiodino_s8` −0.0137 is not a schedule artefact. The likely cause is resolution/granularity mismatch
    plus overfitting [M] [pat].
- Slice span at 12 slices: 0.20–0.80 0.8525 / **0.10–0.90 0.8565** / 0.00–1.00 0.8427 [M] [ig].
  - Competitor: 9 adjacent 0.7815 vs 9 equal-spaced 0.7635 = +0.018 [L] [ig].
  - Slice count: see the aggregation summary. 12→24 flat, 24→36 −0.0224 [AXES:90-91][axes].

## 3. Augmentation

- Knee classifier augmentation in the literature is almost all affine (flip / rotate / translate) [L] [norm]:
  - MRNet, Azcona 93.4% vs MRNet 91.6% [U: trunk+aug confounded] [ceil].
  - Prostate DWI: rotation was the best single op, AUC 0.85.
- No MRI-artefact augmentation has a classifier AUC in the literature. TorchIO has SEG evidence only [L-null]
  [norm].
- The same is true of elastic, gamma, Gibbs, slice-order jitter and anisotropy jitter [U] [norm].
- ViTs lean more on AugReg. On small datasets the right AugReg is worth roughly 10× data (Steiner) [L] [cnn] [norm].
- ConvNets are more annotation-efficient than ViTs on medical fine-tunes (Taher 2025) [L] [cnn].
  - Hence the prior that a CNN wants *less* augmentation than DINOv2 [U] [cnn].
- Laterality: a sagittal slice-ORDER reversal flips medial/lateral. An in-plane horizontal flip of a sagittal
  image mirrors A-P, which is label-safe [U-derivation] [norm]. This was handed to a code audit; it is not
  resolved in these sources.
- **Ours:**
  - General augmentation: CLOSED, "the regulariser fired, the overfitting did not care" [M] [AXES:139][axes].
  - Slot dropout: KEEP at 0.40, dose-response saturating, and "the *sole* transfer carrier — augmentation is not"
    [M] [AXES:123][axes].
  - Ghosting / k-space motion: OPEN at one point [M] [AXES:126][axes].
    - `ghosting_s2026` decode −0.0304 vs control −0.0361 = +0.0057 DIRECTIONAL.
    - Seat −0.0002.
    - `--ghosting 0.040` is the measured KMAR-50K median residual, not a tuned knob.
  - External data as labelled augmentation: CLOSED (`noisy_student` 0.8548) [M] [AXES:157][axes].
- **Supersedes** [norm] ideas #2 and #4 (flip/rotation audit, TorchIO artefacts). The ghosting row is the
  measured instance of #4; #2 is still unrun per these sources.
- Train-time shift-robustness for a classifier toll: no paper prices a consistency / shift-invariance loss
  [L-null] [toll]. BigAug failed to generalise on skin-lesion CLF (LDDG) [L] [toll].

## 4. Mixup and regularisers

- Flow-Mixup (feature-space mixup under label noise, CXR14): reported gain, delta not extracted [U] [norm].
- Balanced-MixUp: delta unverified [U] [norm].
- [norm] ranked feature mixup #1 (PRICE ~3 h, P ~0.10) as a fork prior [U-prior].
- **Superseded / contradicted by our ledger:** mixup CLOSED, "the regulariser works and the ceiling does not move"
  [M] [AXES:138][axes].
- SAM:
  - rho=0.05: −0.0421 [M] [AXES:143][axes].
  - rho=0.01: 0.8981 vs 0.8949 ledger75 = +0.0032, DIRECTIONAL ONLY. Composition onto shipping is P ~0.05 and
    unarmed [M] [AXES:238][axes].
- Stochastic depth, EMA and epoch budget: CLOSED [M] [axes].
- WiSE-FT is monotone: 0.9061/0.9038/0.8895 [M] [AXES:81][axes].

## 5. Losses and targets

- AUC-margin (LibAUC):
  - `margin_seat` −0.0053 solo.
  - Two-stage `auc_ft_lw24` −0.0096 ledger75 on the labelwise line [M] [AXES:107][axes].
  - "The literature's single best loss offer for imbalanced medical AUC, and it is spent."
- Focal:
  - `focal224` −0.0024.
  - `focal_labelwise224` 0.8796 vs 0.8832 = −0.0036.
  - No interaction [M] [AXES:112][axes].
- Per-label weighting / pos_weight: refuted twice [M] [AXES:109][axes]. This supersedes [norm] idea #3 (pos_weight,
  P ~0.05).
- Severity weighting: CLOSED, it moved the graded findings the wrong way [M] [AXES:110][axes].
- Ordinal `grade_ladder`: buys nothing [M] [AXES:111][axes].
  - Literature: OAI KL +0.3–1.6pp accuracy, measured in accuracy not AUC [L] [ceil].
- Twelve aux weights: −0.0019 [M] [AXES:113][axes].
- Contrastive: CLOSED [M] [AXES:115][axes].
- MedianLogit: loses [M] [AXES:116][axes].
- ASL: micro mAP 84.8→86.3 [L]; the CXR14 figure is [U] [ceil].
- Uncertainty heads: no AUROC gain over BCE across a 13-method benchmark [L] [ceil].
- Class-balanced sampling targets threshold metrics. AUC is prior-invariant, and "balanced" is ill-defined for 12
  co-occurring labels [L + U-derivation] [norm].
- Hedge soft targets are ALREADY the label: 0.08/0.28/0.68/0.82/0.94 (`report_reader.py:109`). Rep-GLS softening
  is not a new lever [M] [AXES:125][axes] [ceil].
  - Rep-GLS: BCE 79.63→84.14 [L] [ceil].
  - The gold-58 read contraindicates the sharpening half [M] [ceil].
- `--calibrated-targets` × `--hold-annotations`: KEEP; the effect is entirely an interaction [M] [AXES:124][axes].

## 6. Loss correction with a measured noise matrix

- Natarajan 2013 consumes known asymmetric ρ₊₁, ρ₋₁ [L] [nm]. Patrini 2017 (forward/backward) is single-label [L]
  [nm].
- Per-column forward correction for multi-label BCE is unbiased by linearity. That result is **unpublished and
  untested** [U-derivation] [nm]. The same holds for soft targets [U] [nm].
- Menon 2015 [L] [nm]:
  - Under FN-dominant noise, AUC is optimisable with **no** noise parameters, because ranking is preserved.
  - A T matrix's residual value is calibration, not ranking.
- Mis-specified T: an estimated T̂ loses a median ~10 accuracy points (0 on binary IMDB, 27 on MNIST) [L] [nm].
- BoMD NIHxPDC, 13-label CXR, mean AUC [L] [nm]:
  - BCE 85.65, NVUM 87.89, BoMD 89.76 at inherent noise.
  - The +1.4 to +4.0 gains start from 0.84–0.86 baselines. Nothing is measured above a 0.90 baseline.
- **Warning** [L] [nm]:
  - Naive per-label F-correction scored 80.8 vs BCE 87.0 (NIH-Google pneumothorax).
  - DivideMix/FINE hurt OOD.
- Label-noise correction at knee scale is a silent cell. κ is measured (0.55–0.72, Astuto); correction gains are
  measured only at 10⁵ [L-null] [ceil].
- Pseudo-labels: Zhan +8.1pp (0.631→0.712) on 641 nodules [L] [ceil].
  - At our ~1.2:1 unlabeled:labeled ratio, the cell is empty [L-null] [ceil].
  - Ours: CLOSED as external labelled aug [AXES:157][axes].

## 7. Test-time adaptation and decode-toll shrinkers

- CARVE: +0.036 internal / +0.034 external [L] [toll].
  - It is the only published result in our cell (model fixed, multi-label, post-normalisation).
  - The external +0.034 is from a near-chance base, 0.499→0.533.
  - Recipe: V=8, ρ=0.25, lr 1e-5, 2 steps, norm-affine only.
  - No V=1 row is quotable (figure-locked) [U].
- Our single-view CARVE pass: −0.0307 [M] [toll].
  - It sits in an unpublished cell on both the view-count and the LN-vs-BN axes.
  - Two suspects remain, so it is not a CARVE refutation.
- The medical classifier TTA tier is 100% BatchNorm. Our trunks are LayerNorm [L] [toll].
- CoWA: the absolutes are image-locked and its repo holds no results, so it is UNPRICED [U] [toll].
- LUMINA: figure-only [U] [toll].
- Kim SHOT-FT: +0.011 [L] [toll].
- "I Can't Believe TTA Is Not Better": 11/12 degrade, up to −31.6 pts [L] [toll].
- Dutt (LoRA crossover ~3.8k images) is F1-only, with no AUC hits in the paper. Our `ortho_lora` −0.0103 contradicts
  its transfer [M] [toll].
  - LoRA: CLOSED [AXES:132][axes].
- Overlap TTA: consumed. Band TTA loses [M] [AXES:199-200][axes].

---

## Open questions

- Ghosting dose scan and a board read: one point, +0.0057 directional on `decode_cost` [AXES:126][axes].
- Label-safe flip audit (slice-order reversal vs medial/lateral labels): proposed, unrun here [norm].
- Nyul / per-sequence standardisation as a scanner-axis instrument: needs a re-decode; unrun [norm].
- Resolution 420 under labelwise: UNREAD [AXES:76][axes].
- CARVE at V=8 on an LN trunk, pre-registered with both suspects: unrun [toll].
- Per-column forward correction with a measured T: unpublished theory with one adverse precedent. The Menon result
  says ranking needs no T [nm].
- Wide-fixed-field → pad → site-cue composition on a transfer endpoint: never priced [crop].

## Sources

- [ig]: [2026-09-01_input_side_knee_mri_gains.md](../deep_dives/2026-09-01_input_side_knee_mri_gains.md)
- [cnn]: [2026-09-01_cnn_vit_augmentation_sample_efficiency.md](../deep_dives/2026-09-01_cnn_vit_augmentation_sample_efficiency.md)
- [ortho]: [2026-09-02_orthofoundation_preprocessing_qaf_answer.md](../deep_dives/2026-09-02_orthofoundation_preprocessing_qaf_answer.md)
- [bn]: [2026-09-04_frozen-batchnorm-drift-in-transfer-learning.md](../deep_dives/2026-09-04_frozen-batchnorm-drift-in-transfer-learning.md)
- [pat]: [2026-09-04_qei-dehghan-patch-size-replication-check.md](../deep_dives/2026-09-04_qei-dehghan-patch-size-replication-check.md)
- [crop]: [2026-09-10_crop-extent-and-physical-normalisation-literature.md](../deep_dives/2026-09-10_crop-extent-and-physical-normalisation-literature.md)
- [toll]: [2026-09-10_cross-decode-toll-shrinkers-ranked-by-measured-effect.md](../deep_dives/2026-09-10_cross-decode-toll-shrinkers-ranked-by-measured-effect.md)
- [ceil]: [2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md](../deep_dives/2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md)
- [nm]: [2026-09-11_loss-correction-with-measured-noise-matrix.md](../deep_dives/2026-09-11_loss-correction-with-measured-noise-matrix.md)
- [norm]: [2026-09-15_knee-mri-normalization-augmentation.md](../deep_dives/2026-09-15_knee-mri-normalization-augmentation.md)
- Axis index: [axes] = [docs/AXES.md](../../docs/AXES.md). Row numbers are file lines as of 2026-09-25. §B holds
  rows 74-98, §C 100-117, §D 123-144, §E 157, §H 199-204, §Open 238.

[ig]: ../deep_dives/2026-09-01_input_side_knee_mri_gains.md
[cnn]: ../deep_dives/2026-09-01_cnn_vit_augmentation_sample_efficiency.md
[ortho]: ../deep_dives/2026-09-02_orthofoundation_preprocessing_qaf_answer.md
[bn]: ../deep_dives/2026-09-04_frozen-batchnorm-drift-in-transfer-learning.md
[pat]: ../deep_dives/2026-09-04_qei-dehghan-patch-size-replication-check.md
[crop]: ../deep_dives/2026-09-10_crop-extent-and-physical-normalisation-literature.md
[toll]: ../deep_dives/2026-09-10_cross-decode-toll-shrinkers-ranked-by-measured-effect.md
[ceil]: ../deep_dives/2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md
[nm]: ../deep_dives/2026-09-11_loss-correction-with-measured-noise-matrix.md
[norm]: ../deep_dives/2026-09-15_knee-mri-normalization-augmentation.md
[axes]: ../../docs/AXES.md

## 2026-09-28: SCE and DINOv2 registers priced, not filed

- SCE: P ~0.03. It is conditional on ELR (knee-bpuei) reading non-null.
- DINOv2-reg: P ~0.03. It is an instrument first: read the attention maps of the shipped solo for background artifact tokens.

See [forum and noisy-loss sweep](../deep_dives/2026-09-28_forum-and-noisy-loss-sweep.md).
