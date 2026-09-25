# All-lanes web sweep — model, architecture, loss, data, optimisation — 2026-09-24

Four parallel literature sweeps (arch / loss / data / optim), each told to read `docs/AXES.md` first and
return only what the ledger has NOT already run. Probabilities are priors on clearing +0.0150 ledger75
over the shipping labelwise line (`labelwise_depth24_drop40_long_s2026` 0.9053), not measurements.

## Verdict

No lane offers a single lever at ≥1.5% with decent odds. Aggregation is saturated (every pooling within
0.0019, README 16948-16963), so what is left targets the per-slice FEATURE, the TARGET, or the
holdout→board SHIFT (a flat −0.034 across eleven members).

## Ranked by card-hours per expected gain

| rank | candidate | lane | card | P(+0.015) | status |
|---|---|---|---|---|---|
| 1 | two-stage LibAUC: AUC-margin fine-tune warm from the CE fit | loss | 0.5 h | ~0.06 | **running** `auc_ft_lw24` (knee-rt48x) |
| 2 | post-hoc pull-to-pretrained (WiSE-FT α, LiNeS, Model Stock on basinsoup children) | optim | minutes | ~0.03-0.05 | needs a script around `weight_soup.py`; aims at the board shift |
| 3 | Rep-GLS reliability-gated target sharpening (hedged report cells soft, explicit sharp; arXiv 2508.02495) | loss | ~0.5 h warm | ~0.10 | needs hedge-flag per cell from the report parser |
| 4 | multi-block fusion, DINOv2 blocks 9-12 (fused, not replaced) | arch | ~1.6 h | ~0.05 | small code change |
| 5 | in-encoder inter-slice context (zero-init adapter / depth shift) | arch | ~1.6 h | ~0.08 | code |
| 6 | `multiscale` under labelwise | arch/input | ~1.6 h + ~143 GB decode | ~0.08-0.10 | UNREAD in AXES; cache build is the cost |
| 7 | NVUM logit-memory regulariser | loss | ~0.5 h warm | ~0.07 | code |
| 8 | series-choice augmentation (random eligible series per slot) | data | ~1.6 h | ~0.05 | code |
| 9 | SPD (decay toward pretrained during training) | optim | ~2 h | ~0.05 | code |
| 10 | attention-guided zoom (AOLM), cross-series attention, series embeddings, registers | arch | ~1.6 h each | 0.02-0.07 | lower |

Barred by the data ruling and not listed: anything trained on Rijeka / MRNet / fastMRI / SKM-TEA.

---

# Architecture-lane web research — kneemri (2026-09-24)

Scope: model/architecture levers only. The trunk stays DINOv2 (currently small, last 6 blocks, 336px, labelwise slice attention plus slot head). No external-data training, solo checkpoint only, no fold or seed sweeps.
Cross-checked against docs/AXES.md, experiments/README.md (the aggregation saturation note at l.16940-16963, the block-9 readout at l.1205-1224, ROI at l.17116-17134), kneemri/models/slice_pooling.py and kneemri/models/slot_head.py.
The repo research docs that were already present were not duplicated: 2026-09-04_qen, 2026-09-04_qdt, 2026-08-11 slice-aggregation, 2026-09-11 views/MIL, 2026-09-10 unseen-candidates, and 2026-08-24 knee SOTA (CoPAS).

## Honest headline

No source I read offers a single mechanism with a credible path to +0.015 on its own.
- The repo's own record says every aggregation combiner lands within 0.0019. The limit sits upstream, in one slice's feature vector (README:16948-16963; interpretation 2026-08-21).
- The ideas that could still move that vector are:
  - in-trunk cross-slice context;
  - multi-layer feature fusion;
  - a within-slice self-localised zoom.
- All three are untried here, and they compose.
- Price each one at +0.003-0.01 and treat them as a chain that has to sum past the bar.

## Sources read (primary)

| Source | What it actually says (read) |
|---|---|
| AnyMC3D https://arxiv.org/html/2512.12887v3 | Frozen 2D FMs (DINOv2/v3-L, MedImageInsight, MedGemma) + LoRA r8 per task. Query-based attention pooling beat avg/median/LSTM/Transformer slice fusion (Table 6). Queries are view-specific, followed by patient-level fusion. There is an optional aux seg head on patch tokens. DINOv3 0.894 avg AUC vs MST full-FT 0.869. |
| MST https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/ | DINOv2-small CLS per slice, fine-tuned, then a 1-layer transformer over slices; best with no positional embedding. Knee meniscus (sag T2FS, 1,199 pts): transformer 0.85, averaging 0.82, linear 0.78. Frozen DINOv2 gives 0.62-0.66. Small matched base. |
| RSNA 2024 spine 2nd https://github.com/brendanartley/RSNA-2024-Competition | Stage 1 regresses coordinates (8 mid frames, attention head, 99.1% within 5px). Stage 2 crops 24 frames and runs encoder, LSTM, attention pooling. Loss 0.5 label + 0.5 pseudo. Seq-flip aug, manifold mixup, 9-rot TTA. The axial pipeline gave +0.01-0.02 CV but nothing in the ensemble. |
| RSNA 2023 abdominal 1st https://github.com/Nischaydnk/RSNA-2023-1st-place-solution | 2.5D (3 adjacent slices as channels), 32 slices, CNN + GRU, organ crops from 3D seg. Aux dice loss (w 0.125) gave "+0.01 to +0.03". 384px. |
| DINOv2 linear eval https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py | Grid over n_last_blocks in {1,4} and avgpool in {F,T}. Features = concat CLS of the last n blocks (+ mean patch of the last block). No numbers in the code. |
| Registers https://arxiv.org/html/2309.16588v2 | DINOv2 linear IN1k 84.3 to 84.8. Dense tasks gain more (ADE20k 46.6 to 47.9). |
| MM-DINOv2 https://arxiv.org/html/2509.06617v1 | Multi-sequence tokens concatenated into one ViT-B trunk (early fusion). External MCC: 0.29 baseline, 0.36 with per-image pos emb, 0.49 with modality emb, 0.57 with full-modality masking. Glioma. |
| CoPAS https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/ | Cross-plane attention (Q main plane, KV other planes), cross-sequence channel attention, label-correlation module. Supp. Table 10 (from memory of the earlier read): no ROI 0.725, ROI 0.781, +CPA 0.774, +CSA 0.799, +CRM 0.812. |
| Knee nine-abnormality (eClinicalMedicine) https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/ | ResNet per plane, concatenated. 3D U-Net meniscus crop. AOLM coarse-to-fine attention localisation; its ablation moves AUC range 0.730-0.966 to 0.687-0.953 without it. |

## Unverified (title, snippet or abstract only)

- RSNA 2024 spine 1st (Avengers): localize-then-classify on multiview 2.5D crops (search snippet only).
- RSNA 2024 3rd: not read.
- RSNA 2022 cervical 1st: 3D seg, then per-vertebra 15-slice 2.5D with mask channels, then CNN + LSTM (snippet).
- SAM+D https://arxiv.org/abs/2607.29033: zero-param Depth Shift Module (TSM-style channel swap between adjacent slices) inside frozen ViT blocks, plus depth-routed LoRA. Segmentation only; no numbers read.
- TomoGraphView 2511.09605, M-SCAN 2503.01634 (spine AUROC 0.971, no ablation read), DinoAtten3D 2509.12512, 2508.07165 (multi-seq pretraining, 12-label private knee): abstracts only or unread.
- "last-4 CLS significantly beats last-1": search snippet, source not verified.

## Ranked candidates (gain / cost)

1. **Multi-layer feature fusion (blocks 9-12 CLS+meanpatch, concat or learned softmax-weighted sum) into the token pooling.**
   - Source: DINOv2 linear.py.
   - Mechanism: intermediate blocks keep the local, low-level texture that focal findings need, while block 12 is the most semantic.
   - AXES: untried. The "readout block 9" closure (-0.0227) REPLACED block 12 with block 9; it did not fuse them, so it is not the same test.
   - Cost: one small module plus a 24-ep arm.
   - Gain: +0.002-0.008. P(>=0.015) ~0.05.
2. **In-trunk cross-slice context: a zero-param depth shift (TSM/DSM) in the unfrozen blocks, or a zero-init inter-slice attention adapter after blocks 9/12.**
   - Sources: SAM+D (unverified numbers); MST shows slice context helps knee (0.85 vs 0.82), though only post-trunk.
   - Mechanism: a slice's token features see their neighbours before they are reduced to one vector. This attacks the diagnosed "limit is upstream in the per-slice vector".
   - AXES: untried in-trunk. Related closures are through-plane 2.5D channels (+0.0005) and DepthAttention post-pooling (-0.0015); both mix AFTER or AT the input, not inside the blocks.
   - Gain: +0.003-0.01. P ~0.08.
3. **Additive multi-scale slots (`multiscale`, 6 wide plus 6 tight) under labelwise.**
   - Sources: spine and abdominal two-stage crops; CoPAS ROI 0.725 to 0.781.
   - Mechanism: add a zoomed view of the focal compartment without removing context.
   - AXES: listed UNREAD/OPEN. It is the "cheapest unrun input test".
   - Gain: +0.003-0.01. P ~0.08-0.10.
4. **Self-localised second-pass zoom (AOLM-style).**
   - Source: PMC12508577.
   - Mechanism: threshold our own labelwise/patch attention, crop, and re-encode at higher effective resolution. Only our own model's attention is used, so no foreign labels.
   - AXES: ROI crop CLOSED. That closure was a detected box that REPLACED the slots plus an upsample; an additive self-guided zoom was never isolated (the 2026-09-10 doc lists it as "partially priced").
   - Cost: about 2x the forward passes on the zoomed slots, which is a VRAM concern at 336.
   - Gain: +0.003-0.01. P ~0.07.
5. **Cross-slot (series/plane) self-attention before the per-diagnosis slot queries.**
   - Source: CoPAS (CSA +0.025 and CPA on a ResNet3D baseline).
   - Mechanism: a meniscus finding on sagittal gets contrast and plane context from coronal/axial. SlotHead currently has no slot-slot interaction.
   - AXES: untried as a layer. "Learned combiner +0.0015-0.0021" and the labelwise seat results bound it, and the CoPAS gain was measured on a much weaker baseline.
   - Cost: tiny.
   - Gain: +0.002-0.008. P ~0.06.
6. **Series/sequence embeddings added at the trunk input (MM-DINOv2).**
   - Mechanism: tell the trunk which contrast/plane it is reading, so blocks 7-12 can specialise features per sequence.
   - AXES: untried at the trunk; slot_emb exists only in the head.
   - Cost: trivial.
   - Gain: +0.001-0.005. P ~0.03.
7. **Aux dense head on patch tokens with self-derived targets.**
   - Source: the abdominal 1st-place aux seg loss (+0.01-0.03, with real masks).
   - Mechanism: dense supervision regularises the patch features.
   - AXES: aux objectives CLOSED. Real masks are barred, and foreign-derived masks are barred. Only reopen it with a target we own, e.g. a heatmap from the per-label attention.
   - P ~0.03.
8. **DINOv2-with-registers small as the trunk.**
   - Source: Registers paper.
   - Mechanism: removes high-norm artefact tokens and gives cleaner patch maps. Classification gain is +0.5 on IN1k.
   - AXES: untried (registers are only handled for DINOv3).
   - Cost: checkpoint swap plus pooling handling.
   - Gain: 0-0.005. P ~0.02. It matters more if 2 and 4 read the patch maps.
9. **1-layer slice transformer (MST).**
   - Mechanism: slice self-attention replaces the attention pool.
   - AXES: effectively covered. The aggregation stage is "saturated", and AnyMC3D itself found query pooling at or above a transformer.
   - P ~0.02.
10. **Coordinate-regression stage 1 plus crop (spine 2nd).**
    - AXES: ROI/CoPAS registration CLOSED, and knee has no coordinate labels. Skip.
11. **LSTM over slices, manifold mixup, 0.5/0.5 pseudolabel (spine 2nd).**
    - AXES: order-aware DepthAttention closed, mixup closed, ensemble-teacher distillation closed (0.002 headroom). Not findings.
12. **View-specific LoRA (AnyMC3D).**
    - AXES: LoRA/PEFT CLOSED, and full fine-tuning of 6 blocks already beats it. Not a finding.

## Suggested sequencing (cheapest signal first)
- Run 3 (already built) and 1 (small code change) first; they are independent.
- Then run 2 at a zero-init start, so it equals the champion at step 0.
- Run 4 only if 3 moves.
- 5 and 6 are cheap add-ons for whichever arm wins.
- Read every result on ledger75, and check whether the gain concentrates on the focal labels (menisci, cartilage) as README:16963 predicts for any real mechanism.


---

# Loss / labels / targets / metric-aligned training — research lane (2026-09-24)

Scope: 12-label knee MRI, report labels (~18-20% cell disagreement vs image GT, FN-dominant: the reports
over-call and "on the fence" findings are graded negative), metric mean AUC. Solo DINOv2-small, labelwise
MIL, slot pooling. The shipping objective is BCE on graded targets plus a pairwise softplus rank term
(`--rank-weight 1.0`, experiments/recipe.sh:23).
Bar: +0.015 (1.5%). Noise: paired floor 0.0053, seed spread 0.0062.

## Repo state read first (docs/AXES.md in full, ledger greps)
Loss-side rows CLOSED or refuted:
- AUC-margin (margin_seat −0.0053)
- focal (both rulers)
- severity weighting
- per-label weighting
- grade_ladder
- 12 auxiliary weights (−0.0019)
- contrastive lane (contrast_dense −0.0087)
- MedianLogit
- label-correlation head
- EMA
- DML (−0.0015)
- label correction / masking / table / reader (errors_relabel flat; the detector's recall 0.129 is what binds)
- report_supervision cosine aux (−0.0053 / −0.0063)
- born-again gen2 (+0.0121 was RETRACTED as a leak; the honest distillation term is +0.0056)

KEEP / directional:
- calibrated-targets + hold-annotations (+0.0061, interaction-only)
- SAM rho=0.01 (+0.0032 ledger75, directional)

kneemri/training/objective.py has Objective, RankedObjective, DistilledObjective, DecorrelatedObjective,
MarginObjective and ContrastiveObjective. It has NO ELR/NVUM memory, ASL, GLS sharpening, PU treatment,
or own-history self-distillation.

MarginObjective (objective.py:530-569) is a closed-form AUC-M that REPLACES BCE. It runs from scratch
under AdamW one-cycle (no PESG) at batch 12. Its own docstring admits that rare labels (Fracture ~7%)
skip the term in about half of steps.

## Read (verified at fetch)
| Source | What it says | Numbers read |
|---|---|---|
| NVUM, arXiv 2103.04053 (PDF pp4-9) | BCE + log(1−σ(tᵀp)) regulariser, where t = non-volatile running average of logits minus log π (β=0.9) | NIH→OPI mean AUC: BCE 82.91; +π in BCE 85.80; +π in REG 85.24; both 86.68; full NVUM 88.17±0.48. β=0.1 collapses; 0.5→0.9 is monotone up. **Table 3 (1,962 NIH images, ≥3 radiologists per image): BCE 87.0/84.3, ELR 87.1/83.2, NVUM 88.9/85.5, F-correction 80.8/84.8** |
| BoMD Table 4 (via research/deep_dives/2026-09-11_loss-correction-with-measured-noise-matrix.md) | NIH×PDC, inherent NLP noise | BCE 85.65 / NVUM 87.89 / BoMD 89.76 |
| Rep-GLS, arxiv.org/html/2508.02495 | (1−r)·CE + r·uniform with r∈(−1,1); negative r = SHARPENING of reliable report labels | MIMIC-CXR 14-label: BCE 79.63 → e2e 80.27 → no-sharpen 81.56 → full 84.14 |
| ELR, arXiv 2007.00151 (abstract) | temporal-ensemble targets + log(1−⟨p,t⟩) | on CXR it is ~null (NVUM Table 3 above) |
| LibAUC CheXpert example + docs | AUCM_MultiLabel + PESG, lr 0.1, margin 1.0, gamma 500, DenseNet121, 2 epochs; docs say "pretrain with regular loss, then switch to AUCMLoss for finetuning with a smaller lr" | 1st place on CheXpert 2020 (no ablation number read) |
| Menon 2015 (via the repo doc) | under FN-dominant noise, AUC-optimal ranking needs no noise parameters | theory |
| Competition repo github.com/raahimnawaz/rsna-knee-abnormality-detection | GT = 2 MSK radiologists + adjudicator, severity-thresholded; report agreement ~82% | — |

**Key caveat from NVUM's own ablation:** of its +5.3 OPI gain, +2.9 comes from putting the log-prior
into the BCE logits (logit-adjusted training). Only the remainder comes from the memory term. Two things
follow:
- Most of the gain is imbalance handling on a 0.83 baseline.
- ELR alone is null on the clean-relabelled NIH benchmark.

## Seen but not read deeply (UNVERIFIED numbers)
- MIDAM (multi-instance AUC maximisation), arXiv 2305.08040
- AUC-mixup, arXiv 2310.11693
- long-tailed noisy multi-label CXR, arXiv 2311.17334
- PU-DPO (unmentioned findings = unlabeled), arxiv.org/html/2608.05341
- PS-KD / self-KD medical: nature.com/articles/s41598-025-10717-4, arXiv 2508.20461
- SSL pretraining under noisy labels, arXiv 2308.04551
- LRC-CXR (label-wise reliability + kNN smoothing), sciencedirect S0957417426003519
- RSNA 2024 lumbar 1st-place writeup (kaggle.com/.../avengers-1st-place-solution): NOT read. There is no
  verified loss-side lesson from Kaggle RSNA winners.

## Ranked (plausible gain ÷ cost)
1. **Reliability-gated sharpening on the calibrated targets (Rep-GLS-style).** Sharpen (r<0) the cells
   whose report is explicit or confident. Smooth (r>0) the cells that are hedged or on the fence.
   - Touches: the calibrated-targets KEEP row. Untried.
   - Cost: one target transform, warm-start short arm.
   - Published: +2.6 from sharpening alone, on a 0.80 baseline.
   - P(≥+0.015): ~10%.
2. **NVUM memory regulariser (non-volatile logit memory, β=0.9).** Untried here: EMA was over weights,
   and distillation came from OOF twins. It is not the model's own logit history.
   - Published: memory-only share ≈ +1.4-2.4 on 0.83-0.86 baselines. ELR's sibling is null on
     clean-relabelled CXR.
   - Drop the log-π part: our rank term plus per-label AUC already neutralises prior shifts.
   - P(≥+0.015): ~7%.
3. **LibAUC two-stage: CE checkpoint, then short AUC-M/PESG fine-tune at low lr.** This re-opens
   margin_seat on MECHANISM: that arm replaced BCE from scratch at batch 12 under AdamW, which is not
   the published recipe.
   - Rare-label skip needs a fix: a per-label positive memory or larger effective batch.
   - Cost: cheap (warm start).
   - P(≥+0.015): ~5-8%.
4. **In-domain image-report contrastive continued pretraining of DINOv2 (~4.4k train reports).** This is
   the only literature-backed open text mechanism (repo audit 2026-09-01).
   - Neighbours: report_supervision and contrast_dense are negative, but they are mechanism (b), not (a).
   - Cost: expensive.
   - P(≥+0.015): ~5%.
5. **PU treatment of unmentioned negatives (Menon / PU-DPO).** Mostly overlaps calibrated-targets, and
   AUC is already parameter-free under FN noise. Low.
6. **kNN label reliability (SimiFeat / LRC-CXR) as the label-axis detector.** The ledger names this as
   the next detector after recall 0.129. The label axis is closed at the CURRENT detector only. Low-moderate.
7. **Own-history self-distillation (PS-KD).** Neighbours: DML closed, honest distillation +0.0056. Low.
8. **MIDAM stochastic-pooling AUC.** Unread numbers. Low.
9. **ASL.** Dominated by the focal closure. Very low.
10. **Label-correlation / ML-Decoder heads.** Already refuted. Skip.
11. **Forward (F-)correction with the measured T.** Loses to BCE on the only multi-label medical
    head-to-head (80.8 vs 87.0). Skip.
12. **Kaggle RSNA winners' loss side.** Nothing read. Unverified.

## Honest bottom line
The loss lane is largely mined here. No single item has better than ~10% odds of +0.015 solo.

Every published gain sits on a 0.80-0.86 baseline. Nothing is measured above 0.90.

Items 1-3 are cheap warm-start, single-objective changes whose mechanism targets FN-dominant report
noise or the metric directly. They are candidates for the "sub-bar increments that compose" chain.
They are not a lever the size of the gap on their own.


---

# Data-pipeline lane: research sweep (2026-09-24)

Scope: this covers preprocessing, normalisation, slice sampling, series selection, orientation, cropping, MRI augmentation, sampling and balancing, and missing-sequence handling at test time. It is read-only and does not include external training data, corpus expansion, or seed/fold sweeps. It builds past `research/deep_dives/2026-09-15_knee-mri-normalization-augmentation.md`, `research/deep_dives/2026-09-10_cross-decode-toll-shrinkers-ranked-by-measured-effect.md` and `research/deep_dives/2026-09-23_forum_sweep.md` and does not repeat them.

## What the record already closes (this is why the lane is thin)

- **Normalisation:** z-score is at a flat optimum (README:12396). The reader/normalisation axis moved +0.0015 (README:10249).
- **Intensity augmentation:** refuted twice. `intensity_augment` read -0.0064 and `aug_long` read -0.0076.
- **Pixel mixup:** α 0.2 read -0.0006 and α 0.4 read +0.0014, so the axis is CLOSED (README:3981).
- **Slices, resolution and crop:** slice count 12→24 flat, 24→36 -0.0224. Resolution 224/336/392 closed. FOV, band, crop centre, ROI, jitter and padding closed. Registration closed.
- **Curriculum and sampling:** `disagreed_ramp` read +0.0067 on midpoint but **+0.0008 on ledger75** (README:12924-12934). `disagreed_full` read -0.0036. So the disagreement sampler has run and is null on the good ruler.
  - Loss-based mining is rejected because about 87% of the residual is aleatoric.
  - Per-label weighting is refuted twice, and oversampling the corpus does not lift the mean.
- **Labels:** label correction, label masking and the noisy student are all closed.
- **Laterality:** already handled (no flip augmentation; the coronal mirror axis is closed at 0.70%). Slices are ordered by IPP·normal (`core/data/corpus/acquisition.py:340-372`).
- **The one tiny untested corner:** `README:1396-1398` says "Nothing has tested a geometric or slice-space perturbation".

## Sources read

- **Artley, RSNA 2024 lumbar 2nd place:** https://github.com/brendanartley/RSNA-2024-Competition
  - Uses sequence-order flipping and combines left/right sides from different sequences.
  - Applies manifold mixup on the embeddings.
  - Uses variable T2 frame counts and sizes for diversity, and nine-rotation TTA.
  - Pseudo-labels at 0.5/0.5 with confident-learning pruning.
  - **No per-trick deltas are published.**
- **RSNA 2024 lumbar 1st place:** the writeup page is JS-rendered and could not be read, so it is UNVERIFIED.
- **Kaggle competition page:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection. Only 58 of 4,407 studies have radiologist labels; the rest are report-derived.
- **TranBaDat2607:** https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection. LLM-graded soft labels for all studies; 0.930 holdout and 0.939 LB with a four-model ensemble.
- **raahimnawaz:** https://github.com/raahimnawaz/rsna-knee-abnormality-detection
  - Says the frontier uses 64 slices over 6-94% of the stack.
  - Their NIfTI path cost -0.085 against DICOM.
  - Positional label binding misaligned 86 labels.
- **Forum sweep 2026-09-23** (already in the repo):
  - 741916: the host's fix for corrupted train DICOMs "may have failed".
  - 742122: the hidden rerun fails on series with mixed slice sizes.
  - 737696: slice DENSITY 44→80 took the Raptor CoAtNet from 0.917 to 0.932. This is architecture-bound; ours reads 24→36 at -0.0224.

## Candidates, ranked by gain/cost (P = honest odds of >= +0.015 on ledger75)

1. **Series-choice augmentation (untried).** At train time, draw uniformly among all eligible series per slot instead of always the densest. At test time, average over candidates.
   - Mechanism: this is real acquisition diversity from within the study, i.e. new views with correct labels. That is what pixel augmentation could not supply.
   - Needs:
     - a census of how many studies have more than one candidate per slot (`slots.py contested()` counts only ties);
     - a cache holding all candidates.
   - P ~0.05.
2. **Slice-subset / slice-position jitter (untried; README:1398).** Draw a random 24 from a denser cache with jittered positions, and read evenly spaced slices at eval.
   - Mechanism: it perturbs the only axis never perturbed. It is aimed at memorisation, since `band_dense_long` peaks at epoch 6.
   - Needs a 36-slice cache (knee-jb0ci: check it is still on disk).
   - P ~0.04.
3. **Ghosting dose scan (AXES OPEN, one point).** Sweep k-space motion/ghost strength.
   - Mechanism: close the acquisition toll rather than the in-domain holdout.
   - Price it on decode_cost or the board; holdout cannot see it.
   - P ~0.03.
4. **Slot-level CutMix across studies (untried).** This is Artley's "sides from different sequences".
   - Swap one slot between two studies. Mix the labels using per-label slot attention or the per-label prior of which slot carries each label.
   - Unlike pixel mixup, each input stays a real image.
   - Pixel mixup is closed, so P ~0.03.
5. **Manifold mixup on pooled embeddings (Artley).** Different space, same class as closed pixel mixup. P ~0.02.
6. **Slot-subset TTA over present slots (untried; grep found nothing).** Average over leave-one-slot-out passes. It is cheap (inference only) but TTA is mostly closed. P ~0.02.
7. **Rotation TTA, rot3 (AXES untested, knee-c9q4; Artley used nine).** Inference only. The affine sweep is closed (knee-2sqyb), so P ~0.02.
8. **k-space anisotropy / resolution downsampling augmentation (untried).** Simulates thicker slices and lower matrix sizes. Same toll logic as ghosting. P ~0.02.
9. **Bias-field augmentation (TorchIO RandomBiasField).** It belongs to the refuted intensity class, so it is instrument-only. P ~0.01.
10. **Robustness audit (not a gain).**
    - Check for mixed-slice-size series (742122).
    - Check for corrupted train DICOMs the host fix may have missed (741916), and count how many are silently decoded wrong.
    - A data-integrity check; P of a gain ~0.02, and only if many studies are corrupt.
11. **Extended slot scheme on the labelwise head.** 9.9% of series are unrouted. extended_slots is UNREAD on max; verify what ships. P ~0.02.
12. **Wider band (0.06-0.94) with 64 slices (public frontier).** The band and count axes are closed on our architecture; mention only. P ~0.01.

## Honest summary

No published source gives a per-trick delta near +0.015 for any item in this lane. The ledger already closes the big ones. The two untried mechanisms that add within-study diversity with correct labels are items 1 and 2, and they are the best bets. Each is well under one in ten.


---

# Research — optimisation / fine-tuning recipe / regularisation / evaluation lane (2026-09-24)

Scope: web research only, read against docs/AXES.md. Nothing here was run. "Read" = I fetched the page;
numbers marked (abstract/summary) came via a summariser and were not checked against the paper's tables.

## Headline

The literature's live offer in this lane is almost entirely **post-hoc weight-space editing toward the
pretrained anchor** (WiSE-FT, Model Stock, LiNeS, MonoSoup, LN-shift rescaling). All of them:
- are "averaging as the LAST operation", so they respect §F's surviving rule;
- run card-free-ish (a forward pass over holdout909 per setting) on checkpoints already on disk
  (basinsoup, its children, the shipping solo);
- are one axis — *how far the unfrozen 6 blocks sit from DINOv2 init, per layer* — which AXES has NOT
  measured post-hoc (warm start, LP-FT, soft fork are training-time; soups average children, not child vs init).

Honest prior: on a WELL-TUNED in-domain checkpoint every one of these buys +0.1-0.5 pt ID in the papers
(CLIP/ImageNet). Their big numbers are OOD. Our board is OOD to holdout (flat toll −0.034, AXES §I), which is
the only reason to price them above noise — and it also means the holdout cannot pick the knob; a board slot
is the instrument. P(>= +0.015 AUC) is low for all of them (~0.03-0.08).

## Ranked by gain/cost

| # | idea | source | mechanism (1 line) | AXES row | cost | P(>=+0.015) |
|---|---|---|---|---|---|---|
| 1 | **Model Stock on the basinsoup children** | https://arxiv.org/abs/2403.19522 (read, html) | pull the k-child average toward W_pretrained by per-layer t = k·cosθ/(1+(k−1)·cosθ), θ = angle between child deltas | §F basin soup — untried variant (current soup = uniform, t=1) | card-free, <1 h eval | ~0.05 |
| 2 | **WiSE-FT α-sweep on the shipped solo** (trunk only; head kept) | https://arxiv.org/abs/2109.01903 | interpolate unfrozen blocks between init and fine-tune, α∈{0.5..1} | untried (LP-FT/warm-start closed are training-time) | card-free | ~0.04 |
| 3 | **LiNeS** — depth-scaled shrink of the update | https://arxiv.org/abs/2410.17146 (read) | λ(ℓ)=α+β(ℓ−1)/(L−1): shallow unfrozen blocks pulled back to init, deep kept; single-task ID held at 99.8% | untried | card-free | ~0.03 |
| 4 | **MonoSoup** — SVD of one checkpoint's delta | https://arxiv.org/abs/2602.09689 (read, html) | per-layer SVD of W_ft−W_0; reweight high/low-energy parts, hyperparameter-free. ViT-B/32 best ckpt: +0.18 ID/+0.93 OOD; gains big only on badly-tuned ckpts (+2-3 ID) | untried; no-seed, one checkpoint → fits the solo rule | card-free | ~0.03 |
| 5 | **LayerNorm-shift rescaling** | https://arxiv.org/abs/2508.07577 (read, abstract) | scale the fine-tune's LN γ/β shifts by λ; paper finds scarce/pathology data favour CONSERVATIVE LN updates | untried | card-free | ~0.02 (unverified numbers) |
| 6 | **SPD — selective projection decay** (decay toward init, per layer, only when grad agrees) | https://arxiv.org/abs/2411.01713 (read, html) | replace wd→0 with conditional decay to θ0; CLIP ViT-B ImageNet ID 83.66 → 84.21, L2-SP 83.44; λ~1.0-1.4 | untried (L2-SP is also unlisted in AXES) | one arm ~2 h | ~0.05 |
| 7 | **SAM on LN params only (SAM-ON) / SAM for the last few epochs** | https://arxiv.org/abs/2306.04226, https://arxiv.org/abs/2410.10373 (read, abstracts/html) | SAM benefit carried by ~0.1% norm params; late-SAM ≈ full SAM (from-scratch CIFAR only, not fine-tuning) | §Open SAM rho=0.01 (+0.0032 ledger75, directional) — this is a cheaper form, not a new axis | 1 arm; ~half SAM cost | ~0.03 |
| 8 | Schedule-Free AdamW | https://arxiv.org/abs/2405.15682 (search summary only) | removes schedule via implicit averaging; wins AlgoPerf (from-scratch) | §D OneCycle KEEP | 1 arm | ~0.01 |
| 9 | TWA / LAWA / SWA | https://arxiv.org/abs/2502.06761 (read) | with LR fully annealed to zero, WA "converges closely to the annealed model" — speed, not generalisation | CONFIRMS §D EMA closed / §F SWA not worth implementing | — | ~0 |
| 10 | Test-time entropy adaptation of LN (SAR/TENT) | https://arxiv.org/abs/2302.12400 (search summary) | entropy-min on LN affine at test; SAR stabilises LN-ViT | untried; transduction measured +0.0008 (knee-nchjt) | kernel runtime + multi-label entropy ill-posed | ~0.01 |
| 11 | Non-smooth components / plasticity (tune attn+FFN, not LN) | https://arxiv.org/abs/2602.06883 (read, abstract) | high-plasticity modules drive fine-tune gains | overlaps §D depth closed at 6 | 1 arm | ~0.01 |
| 12 | Clean-subset / confident-learning validation | https://www.nature.com/articles/s41467-022-28818-3 (search summary) | noisy test labels mis-rank models | already covered by ledger75 + ρ=0.88 transfer (§I) | — | n/a (ruler, not lever) |

## Notes per item

- **1-5 are one experiment.** Build a single `weight_edit` read: for each of {basinsoup, shipping solo} and each
  editor (Model Stock t, WiSE α grid, LiNeS γ, MonoSoup, LN λ), score holdout909 on ledger75. Pre-register:
  in-domain flat/negative is EXPECTED (the papers' ID deltas are 0-0.5 pt); the ONLY branch that ships is a
  setting that is flat in-domain (≥ −0.002) — then spend one board slot as an INSTRUMENT for the OOD toll.
  Model Stock needs the children's deltas (on disk for basinsoup k7? verify). Caveat: our "pretrained" is
  DINOv2 for the 6 unfrozen blocks, but the head/labelwise attention has no pretrained anchor — edit trunk only.
- **Model Stock numbers (read):** CLIP ViT-B/32 ImageNet: single FT 79.72, uniform soup 79.76, Model Stock
  79.89 (2 models); ViT-B/16 vanilla FT 82.8 → Model Stock 84.1. i.e. +0.17 pt at B/32 — tiny ID gain; the
  B/16 gap mixes in a better FT recipe. Paper also merges at every epoch end (periodic) — that is a
  resumed-gradient-after-average form, which §F says forfeits the averaged point; prefer post-hoc.
- **SPD:** our wd 0.02 decays toward ZERO, i.e. away from DINOv2 init for the unfrozen blocks. Decay-to-init
  (L2-SP) is not in AXES at all; SPD is its best-published form. Only CLIP-B evaluated, no medical.
- **Late-SAM:** only from-scratch CIFAR evidence; transfer to fine-tuning is unverified.
- **Evaluation lane:** nothing found that beats what the repo has (ledger75, paired floor 0.0053, ρ 0.88).
  The flat −0.034 toll across 11 members is consistent with a label-extraction/site shift rather than
  model fragility — which is exactly the case where anchor-pulling editors (1-6) have their published wins.
- **Not found / not verified:** Lion/Adan/Sophia fine-tuning results on ViT medical tasks — no source with
  numbers read; FLYP is CLIP-contrastive only (not applicable to DINOv2). Surgical fine-tuning (Lee 2022) is
  subsumed by AXES depth=6 closure.

Sources read: arxiv 2403.19522 (html), 2410.17146 (html), 2411.01713 (html), 2602.09689 (html),
2502.06761 (html), 2410.10373 (html), 2306.04226 (abs), 2508.07577 (abs), 2602.06883 (abs), 2512.12887 (abs).


---

