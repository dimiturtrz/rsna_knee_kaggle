# AXES — every lever this project has tried, its verdict, and where it is written down

## Why this file exists

The record is now **465 ledger sections** (`experiments/README.md`, 17,881 lines), **240 interpretations**,
**152 research docs**, **434 arms** and **505 closed beads**. Every one of those is correct and none of them
is an index. Answering "have we tried X?" costs a grep cascade, and the failure mode is not that the answer
is missing — it is that the answer is found too slowly to change what the card runs next, or not found at
all and the axis is paid for twice.

**This file is an INDEX, not a second home for numbers.** A row carries the VERDICT and the POINTER. The
number stays where it was measured, because a number copied into two files is a number that will disagree
with itself — this repo has already spent a board constant, a transfer slope and a pair count on exactly
that. Where a row quotes a figure it is the *key* you grep for, not the citation.

Two companion files, neither of which this one replaces:
- `interpretations/kneemri/2026-10-09_levers_before_final_picks.md` — the levers still OPEN before the final picks (supersedes `2026-09-18_state_of_the_levers.md`).
- `docs/PLAN.md` — the goal, the metric, and what the metric removes from the design space.

## Status vocabulary

| status | means |
|---|---|
| **CLOSED** | measured on a matched control, bounded below the bar. Re-opened only by answering the mechanism, never by re-quoting a number. |
| **KEEP** | measured positive and in the shipped recipe. |
| **BARRED** | not refuted — unavailable. Licence, disk, or rules. A different kind of dead end. |
| **OPEN** | live, priced, unspent. |
| **UNREAD** | an artefact exists and nobody has taken its reading. Outranks bookkeeping. |

---

## A. Trunk / architecture

**The class headline:** *the stack is DINOv2-SPECIFIC — it carries +0.054 on a ViT and 0.000 on a CNN*
(README:17307). Every row below is downstream of that one.

| axis | verdict | where |
|---|---|---|
| DINOv2-small | the trunk everything ships on | shipped recipe |
| capacity ladder — base / large over small | **RE-OPENED AND ANSWERED: capacity PAYS under a readout that can spend it.** The CLOSED verdict (−0.0010, five isolating pairs, median 0.0000) was measured on `rank24` members — a MAX head, pre-whole-stack, pre-labelwise — which is §B's disowned-ruler shape exactly. Re-read under `--slice-pooling labelwise`, one variable: `base_labelwise224` **0.8899** @ e8 vs `labelwise224` **0.8832** @ e9, both 909 studies, matched ruler = **+0.0067**, above the 0.0062 paired floor, converged over 12 full epochs. So the three rank24 rows (small 0.8272, base 0.8283, large_deep_rank24 0.8304) are **DISOWNED, not suspect**, and `adaptation.py:121` ("capacity is the only axis that has ever cleared our noise floor") is the sentence that survives. **It composes; it is not a result alone** — far short of the 1.5% bar. **And the shipping corner is barred by the MACHINE, not by an argument — SEVEN OOMs, and the cause is ALLOCATOR FRAGMENTATION:** at every crash PyTorch holds 23.21 GiB allocated while the card reports 3.38-6.17 GiB free of 31.84, i.e. ~28.5 GiB reserved against 23.21 in use, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a confirmed no-op on this Windows machine. Four candidate causes are ruled out **by measurement, not argument**: NOT missing gradient checkpointing (`recipe.sh:31` defines `THROUGHPUT` as exactly `--recompute-activations` and every arm passes it, so it was ON for all seven — the 2026-09-22 commit `da8af9c8` claims the opposite and **is wrong**), NOT the slice count (12 slices died at a LARGER allocated figure than 20, so the footprint is not monotone in slices), NOT `torch.compile` (`--eager-encoder` reproduced the request to the byte), NOT the `--vram-gb` cap (the 27 GB attempt died at 25.59 GB of demand, under its own cap). Depth is closed there too: `--unfreeze-last` 12 and 9 both OOMed at 336/20sl with the flag confirmed in the stamped recipe, so only the fitted 6 fits. Batch may not be cut because `--rank-weight` is pairwise within a step (66 pairs → 6). The 1.89× base/small footprint ratio measured at 224/12sl understates as tokens grow — do not reuse it at a higher resolution. **The capacity axis therefore moves to 224/12sl**, where it has paid twice (knee-jzho6). **CLOSED THERE TOO, 2026-09-23, FLAT: the RUNG ABOVE BASE DOES NOT PAY.** `large_labelwise224_half` (DINOv2-large, `--unfreeze-last 12` = half of 24 blocks, the fraction the base control gets from its inherited 6 of 12) reads **0.8936 ledger75 @ e8** against `base_labelwise224_dense` **0.8950** = **-0.0014**, inside the 0.0062 paired floor, for **3.1x the parameters** (304M vs 86.6M) and **3.1x the wall-clock** (534 vs 170 s/epoch). **Read it on ledger75 or get the opposite answer** — midpoint says large WON (e8 0.8607 vs 0.8589). The first attempt (`large_labelwise224_dense`, **0.8888 @ e6**) was MY OWN confound: `--unfreeze-last 6` is a constant fitted on a 12-block trunk and adapts a QUARTER of a 24-block one, moving width and adapted fraction together. The fix is proven by train loss at e5 — matched arm **0.6856** against the control's **0.6824**, where the quarter-adapted arm read **0.7015**. So width is not what is missing at any corner this card can reach, and every larger-trunk proposal is retired unless it arrives with a mechanism this arm did not test | knee-ss5rd, knee-mm6v0; `experiments/arms/base_labelwise224.sh`; superseded: knee-5dx47, README:1755 `base_dense`, :3758 base224 |
| **OrthoDiffusion as a volumetric trunk** (3D denoiser U-Nets read as a frozen encoder) | **CLOSED at Stage II, 2026-09-25: `orthodiff_ft_s2026` 0.8054 ledger75 vs control 0.9006 (−0.0952), plateau by e4 of 6, loss monotone — the budget did not bind. Fine-tuning lifts the probe +0.037 midpoint and stays ~0.10 under the trunk; LSE/3-timestep fusion would need a 6× recovery. Earlier status, kept for the reasoning:** **UNREAD, and the ledger calls it closed.** `orthodiff@mean` probes 0.7435, seats +0.0001 — but `ortho_diffusion.py:15-18` PRE-REGISTERED that their 256-wide middle block against dinov2-large's 1024 makes a LOSS uninterpretable and only a WIN clean, and the ledger then read the loss as "the clearest decline". Our read is the paper's Stage I at its weakest: one block not three, one timestep not three, GAP not LSE, a ridge not their learned fusion, frozen not LP-FT. Paper reports 0.908 macro AUROC from Stage II. **Still DISTINCT (js 0.0418 vs the roster's 0.0150)** — but the 2026-09-22 per-label read shows that distinctness carries NO complementary column: the probe is behind on **all twelve**, by 0.0699 (Synovitis) to 0.1790 (Fracture), mean 0.7407 against 0.8724, and a rank-mix with the champion base reads +0.0017 at w=0.15, under the 0.0053 paired floor. A weak-but-complementary trunk shows at least one column ahead; this shows none, which is evidence about the REPRESENTATION rather than about the fusion, and it raises the price of Stage II without closing the axis. **And Stage II is no longer cheap** — `features_orthodiff_336px_12sl_r256_gap256_volume.npy` was deleted in the approved 355 GB cleanup, so a re-read means re-encoding 4,407 studies through three 3D U-Nets; the cache key bakes `gap256` in, so LSE was never recoverable from it either | README:7367-7398; knee-mxz08; code removed 2026-09-25, last present at `b693f32b` (`kneemri/models/ortho_diffusion.py`, `research/scripts/ortho_finetune.py`); `research/deep_dives/2026-09-02_diffusion_feature_extraction_for_knee_mri_classification.md`; knee-vntv |
| LoRA adaptation of a frozen trunk | **CLOSED on all three capacities we own** — small, base, large, one direction. `ortho_lora` 0.8349 vs `ortho_dense` 0.8452 = −0.0103, converged. The base pair's quoted −0.0244 was CONFOUNDED; the matched cell is −0.0103, so the confound carried ~0.0141. DUTT (arXiv:2305.08252) predicted the largest GAIN exactly here | README:11920-11960 |
| DINOv3 | **CLOSED.** "DINOv3 is weak" was an ADAPTATION finding wearing an encoder's name; best honest v3 member still trails the seats | README:1924, :3036; `2026-09-17_member_candidates.md` |
| ConvNeXt | **CLOSED** — and the inherited rate cost the whole trunk before the own-recipe re-run | README:1841 `cnx_dense`, :3326 `cnx_own_recipe`, :6004/:6659 `cnx_mid_rate` |
| ResNet | **CLOSED** — peaked epoch 12 of 48, so the budget was never the constraint; the ViT recipe carries less than half onto it | README:12872, :17087 `resnet_224_ext` |
| the CNN axis as a class | **CLOSED on its own named re-opener, card-free** | README:7796; `2026-09-05_the-cnn-axis-is-maximally-distinct-and-still-worthless.md` |
| **does the CNN fill the ViT's per-label BLIND SPOTS** (the complement question the mean cannot ask) | **CLOSED, card-free, on the per-label ruler.** `resnet_224_long_s2026` against `basinsoup_dmlb_k7_s2026`, @holdout909@cut0.5, paired bootstrap over studies: behind on nine of twelve, eight of those excluding zero; ahead NOWHERE significantly — ACL +0.0047, Lateral OA +0.0109, Synovitis +0.0051 all span zero. Worse on **MCL (−0.0344, CI excludes zero)**, which is one of the two columns the record names weakest, so the complement fails exactly where it was wanted. A 50/50 probability average reads **−0.0025 [−0.0067, +0.0019]** against DINO alone — not even a free seat. **The one piece of apparent support is an instrument artefact**: `competence_axis --on MCL "Lateral OA"` ranks it 7th of 172 at +0.0162, but that is a RESIDUAL against the member's own level and the axis reads `r = −0.60 CONFOUNDED`, the state the module forbids describing ends by recipe in. **Re-open only by naming an architecture with a route to within ~0.01 of the champion** — an off-axis member must be COMPETITIVE first | knee-b39kg; `kneemri/evaluation/competence_axis.py` |
| RadImageNet DenseNet | **CLOSED** — closes the convolutional trunk lane empty | README:6685 |
| RAD-DINO (medical-report backbone) | **CLOSED** −0.037 against its own exact control — 0.8141 e7 vs `base_dense_s2026` 0.8512, six times the paired floor. The cleanest single-variable encoder read this project owns: RAD-DINO is finetuned FROM `facebook/dinov2-base`, so the arms differ in **initialisation weights and nothing else**. Normalisation and truncation both cleared before the verdict; a rate refit is owner-blocked | README:2790, :4291 |
| continued multi-crop SSL pretraining on a medical corpus | **CLOSED AT THIS CORNER, not as a class.** RAD-DINO *is* that mechanism, built by Microsoft on chest radiographs, and it loses 0.037. Chest-radiograph→knee-MRI is not knee-MRI→knee-MRI, so SSL continuation on OUR corpus survives — but it costs days of card and the nearest datapoint is negative. **Do not rank it above §B's labelwise re-reads.** DINO's multi-crop objective (2 global 224 ⧺ 8 local 96, student-local matched to teacher-global through an EMA teacher) is a PRETRAINING mechanism, discarded at inference; it is not what `multiscale` tried | README:4295; `research/deep_dives/2026-09-02_orthofoundation_preprocessing_qaf_answer.md:23` |
| Swin (supervised) | **CLOSED**; block discovery had found ONE stage on a staged trunk | README:6130, :6374; knee-94ec |
| ViT-in21k | **CLOSED** on a trajectory instrument | README:6196 |
| EfficientNetV2-L | **NULL, not a loss** — the 44-slot corpus takes it | README:9106 |
| V-JEPA (volume pretraining) | **CLOSED** — a TIE on frozen features; the finding is the DISTANCE, not the score | README:2542, :2590; `2026-08-22_vjepa_frozen_probe.md` |
| MRI-core in-domain trunk | **CLOSED** as a queue item, 0.8390 | README:8107 |
| architecture monoculture as a seat lever | **CLOSED, negative six of six** | README:10988; `2026-08-28_architecture-monoculture.md` |
| trunk axis inside the foreign recipe | **CLOSED**, and a fit predicted it out of sample | README:9236 (D36) |

## B. Input geometry

**THE RULER UNDER FOUR OF THESE ROWS IS DISOWNED.** `extended_slots`, `resolution_420`, `multiscale` and
`knee-ezz` were each measured on a **max-pooling head**, which demonstrably suppresses a real input change,
so a flat read from it is not evidence the input change is inert. Those four are **UNREAD, not REFUTED —
and not re-opened as promising**: the 2×2 that found this gives one interaction on a single seed, and
+0.0010 is itself inside the macro noise floor. What clears the floor is the per-column pair, which is
mechanism, not score. **Re-open one by re-running it under `--slice-pooling labelwise`, one axis at a time,
priced against its own labelwise control — never against a max-pooling number.** The ledger names
`multiscale` as the input change most likely to be dilution-limited and as the cheapest such test, unrun;
its cache is not on disk, and the decode that must run alone is what prices the arm, not the training.
(README ~14510.)

| axis | verdict | where |
|---|---|---|
| resolution 224 / 336 | **CLOSED.** 224 costs 0.0049 solo and buys 3.2× throughput | README:8327, :7915 |
| resolution 336 → 392 | **CLOSED** — does not lift the meniscus columns | README:3729 |
| resolution 420 | **UNREAD** — max-pooling ruler, see the caveat above | README ~14510 |
| `multiscale` (12 slots: six wide ⧺ the same six tight) | **CLOSED UNDER LABELWISE, 2026-09-24: −0.0046.** `labelwise_ms` **0.8960 ledger75 @ e12** vs its own control `labelwise_ms_ctrl` **0.9006 @ e20**, same 336px/12-slice cache, recipe and seed; led by +0.0136 midpoint at e3 and the lead decayed to below zero by e9. Labelwise pooling does not rescue the dilution. Previously UNREAD: Gated out 2026-08-16 at ≈ **+0.0016 on the mean for a 71.6 GB cache and a training slot** — per-label at matched e7 it landed the diffuse half (Baker's −0.028, Synovitis −0.019, Effusion −0.015) and missed the half the build depended on (Medial Meniscus −0.012, ACL/MCL at floor, Fracture +0.019 on the worst-agreement label). `multiscale.sh` is `retired` in `lib.sh` terms. **But that gate was a max-pooling read**, and this is now the cheapest unrun input test | README:676-682, ~14519; `interpretations/kneemri/2026-08-16_crop80_resolution_gate.md` |
| `slice_shift` (gated neighbour-slice tokens inside the last blocks) | **CLOSED on the shipping line, 2026-09-25.** +0.0035 on the 12-slice line (`shift4_lw12` 0.9041 vs 0.9006), but `shift4_depth24` reads **0.9037 vs 0.9053** ledger75 (−0.0016, gates fired |g| 0.043). Does not transfer to 24 slices; FF twin not armed (knee-6u1tq) | README (2026-09-25 SliceShift); `kneemri/models/slice_shift.py`; knee-ypu31 |
| `fuse_blocks` (gate earlier encoder blocks into the last) | **CLOSED 2026-09-25: +0.0008.** `fuse3_lw12` **0.9014 ledger75 @ e18** vs `labelwise_ms_ctrl` **0.9006 @ e20**, same cache/recipe/seed | README "Gated block fusion"; knee-vivli |
| `extended_slots` (nine slots) | **CLOSED under labelwise too, 2026-09-25: −0.0003.** `labelwise_ext_drop40_long` **0.9045 ledger75 @ e23** vs `labelwise_dense_drop40_long` **0.9048 @ e23** (20sl, 0.1-0.9, drop40, 24 epochs; ext at batch 4×3). Max-pooling read +0.0002. Composing with 24 slices needs a ~107 GB decode against 50 GB free — not worth it on a flat axis | README (2026-09-25 extended under labelwise) |
| WiSE-FT — pull the fine-tuned trunk back toward pretrained DINOv2 | **CLOSED, 2026-09-24, monotone.** Linear interpolation of the moved backbone tensors (blocks 6-11 of `labelwise_depth24_drop40_long`, 0.9053 ledger75) at α = 0.85 / 0.70 / 0.50 reads **0.9061 / 0.9038 / 0.8895**: +0.0008, −0.0015, −0.0158. No interior optimum; the fine-tuned endpoint is where the gain is. Head unchanged, so the head/trunk mismatch at low α is part of the loss — a re-open needs a head refit, not a different α | README 2026-09-24T11:20Z; `scratchpad` script, not committed |
| cached slices 20 vs 12 (`knee-ezz`) | **UNREAD** at −0.0001 — max-pooling ruler | README ~14510 |
| **the depth step, and what it costs to read it** | **UNREAD under labelwise, and blocked by the cache rather than by VRAM.** Its max-head calibration is now measured: `native224` (12sl, 0.2-0.8) 0.8800 vs `band224` (20sl, 0.1-0.9) **0.8801 = +0.0001**, flat — so anything a labelwise arm reads above noise there is the INTERACTION, which is what would make the two-variable step readable. **No labelwise control exists on the 224/20sl array**: `band224`, `band224_hold` (0.8824 @ e11) and `band_deep224` (0.8718 @ e11) are all `--slice-pooling max`. **`base_labelwise224_20sl` died on `FileNotFoundError`** — the on-disk 20-slice array is `images_224px_20sl_0.1-0.9band_extended.npy` (≈37 GB), built under `--scheme extended`, which changes which SERIES fill the slots and so would move a THIRD variable (`arm_check.py:548`: `--crop-mm`, `--scheme` and `--tie` together pick a cache file). **CORRECTED 2026-09-23: there is no decode.** `images_224px_20sl_0.1-0.9band_extended.npy` is ON DISK at 37.1 GB, built 2026-09-19; the first attempt died only because its stamp read `"scheme": "canonical"` and asked for a file that was never built. `--scheme extended` is the whole fix and costs ~1.0 h of card. The three-variable objection STANDS and is answered by reframing rather than by a decode: extended slots + 0.1-0.9 band + 20 slices are together the DATA CORNER the shipping family runs at 336, so the arm asks whether that corner beats the dense 224 recipe — a corner verdict, not a depth verdict. Re-armed 2026-09-23T04:32Z, **and MACHINE-CLOSED the same hour on two OOMs whose numbers name two DIFFERENT binders.** At `--vram-gb 24` the CAP bound: 19.31 GiB allocated, 3.00 reserved-unallocated, a 2.18 GiB request refused with **6.92 GiB still FREE on the device**. Raised to 28 — the largest cap this machine runs clean (31.84 total minus the 2.34 GB LibreOffice holds by arrangement) — and the CARD bound: 23.92 allocated + 3.75 reserved-unallocated = 27.67 of 28.00 with only **1.59 GiB free** when a 1.34 GiB request failed. There is no cap above 28, so the `--scheme extended` fix was correct (the stamp read `extended` and the FileNotFoundError never recurred) and irrelevant: what closes this arm is VRAM. Escapes barred by prior measurement — batch may not be cut (pairwise `--rank-weight`, 66 pairs to a handful), slices ARE the thesis, 224 is already the small side. **Asking it at the SHIPPING corner needs a decode nobody has paid:** the 336 arrays on disk are 20sl-extended (83.4 GB, what ships) and 24sl-canonical (66.7 GB); `images_336px_24sl_0.1-0.9band_extended.npy` does not exist, so ~80 GB on a disk at 196 GB free plus ~3-4 h of decode before a ~2 h arm — the `multiscale` ratio, declined the same morning. **A `--smoke` cannot catch this** — it reads the `_first60` array it writes itself and never touches the corpus | knee-mm6v0; `experiments/arms/base_labelwise224_20sl.sh` |
| field of view / band width | **CLOSED** — the FOV axis closes and MCL is why | README:2032 `band140`; `2026-09-09` FOV census README:12178 |
| crop CENTRE | **CLOSED** — moving it at the control's extent does nothing, which is the point | README:517 |
| **slot-to-slot REGISTRATION (the CoPAS mechanism)** | **CLOSED, both halves, on two card-free instruments.** Input read: slots disagree p50 5.7 mm ≈ 15 px against a 1-2 mm meniscal tear, so misregistration is **3-6x the finding size** — and it predicted the weak columns (MCL 0.7893, Lat OA 0.8246 vs Baker's 0.9086). But two thirds of that is COMMON MODE, and `tissue224` tested the common-mode fix at **−0.0011 FLAT** (converged, flags verified, train loss lower at every epoch — the retained tissue is real and does not convert). `alignment_damage` then tested the residual at a pre-registered differential of **+0.0103 [−0.0305, +0.0421]**, inside its own ≤0.020 kill band, against the ~0.051 the 1.5% bar needed. **CoPAS's +0.056 does not transfer** — theirs was bought against a full-FOV 576 no-crop baseline; we already crop. Unpredicted sign: the poorly-agreeing half scored BETTER, so the Otsu centroid is not tracking study quality and the input read is attenuated too. **Survives:** a model TRAINED aligned from the start (a different experiment, needs the 99 GB rebuild this evidence declined to license) | README:2911-2998, :335 `tissue224`; `core/data/corpus/crop_alignment.py` |
| ROI crop / upsample | **CLOSED** — the crop costs, and the upsample was exactly half of why | README:17112 |
| FOV jitter | **NULL** — buys nothing the wide field had not already bought | README:11854 |
| padding mode | **refuted three of three**; EXTENT withdrawn for want of a matched control | README:12011, :12216 |
| slice COUNT 12 → 24 | **CLOSED, flat** on the shipped intervention | README:17396 |
| slice COUNT 24 → 36 | **CLOSED** −0.0224, behind at every one of 7 matched epochs, at 50% more wall-clock | `2026-09-18_coverage_is_closed_on_a_trajectory_not_on_a_score.md`; knee-jb0ci |
| slice POOLING × count | **INTERACT** — max wins short stacks, loses long ones | README:8011 |
| through-plane neighbours | **density gradient, not a flat** — the named cause is SPAN LOSS, which shrinks with density | README:2189, :4635 |
| readout block depth | **CLOSED** −0.0227 at block 9 of 12. Spelled `encoder_layer`, **not** `--layer` | README:1200 |
| planes (coronal / axial) | **already consumed** — six slots span all three | `core/data/corpus/slots.py:58` |
| slice ORDER | **the pool is order-blind by construction** | README:11211 §3 |
| contiguity / adjacency | **CLOSED by arithmetic**; the docstring that opened it was wrong | README:8143, :4629 |
| region / depth pooling | patellofemoral loss was the band; the meniscus gain does not transfer | README:1168, :8199 |

## C. Objective / loss

**The class headline:** the objective axis is **not** a seat lever (`dissent_seat`), and the deliverable is
now a solo — so a loss that buys diversity and costs solo quality buys nothing at all.

| axis | verdict | where |
|---|---|---|
| AUC-margin surrogate (LibAUC / Deep AUC Maximization) | **CLOSED** — `margin_seat` −0.0053 solo. Two-stage (CE then AUC fine-tune, the paper's recipe) `auc_ft_lw24` −0.0096 ledger75 on the labelwise line, 2026-09-24 (knee-rt48x). This is the literature's single best loss offer for imbalanced medical AUC, and it is spent | README:374; knee-bdhb |
| rank / dissent objective as a seat lever | **CLOSED** — 38th of 66; the per-label table says why | README:463 |
| per-label loss weighting | **refuted twice**, once from the training side | README:567, :1145 |
| severity weighting | **CLOSED** — the knob fired and moved the graded findings the WRONG way | README:1476 |
| ordinal supervision (`grade_ladder`) | **lands, and buys nothing the roster can see** | README:2067, :2700 |
| focal | **CLOSED ON BOTH RULERS** — buys nothing on the mean under max (`focal224` −0.0024) and nothing under labelwise either (`focal_labelwise224` 0.8796 vs `labelwise224` 0.8832 = −0.0036). Both rows of the 2×2 negative, same sign, same size: no interaction, and `cls_mean_focal` closes. This is a **SELECTION** mechanism — re-weighting tokens the head already holds — so it constrains but does not refute a **SAMPLING** change (more pixels, more slices), which is why it lowers P on a resolution arm without closing one | README:1224; `2026-08-25_the-focal-deficit.md`; `experiments/arms/focal_labelwise224.sh` |
| the twelve `--*-weight` auxiliary objectives | **CLOSED** −0.0019 against a matched converged control | knee-evdfd |
| side / laterality contrast | **the loss cannot buy the side** | README:2761, :2818 |
| contrastive representation lane | **CLOSED on mechanism**, not on the batch-size alibi | README:5197, :6039 |
| robust vote (MedianLogit) | **LOSES** — the shape of the loss is why | README:4992 |
| label-correlation / residual head | **REFUTED for zero card** | README:2485 |

## D. Regimen

| axis | verdict | where |
|---|---|---|
| **slot dropout** | **KEEP.** Dose-response, three points, one direction; saturates, **0.40 is the operating point**. And it is the *sole* transfer carrier — augmentation is not | README:16180, :16307, :17351 |
| `--calibrated-targets` × `--hold-annotations` | **KEEP.** Calibration's entire effect is an INTERACTION; replicates at a second slice count | README:14188, :14252 |
| hedge-aware soft targets (Rep-GLS / uncertainty-graded report labels) | **ALREADY THE LABEL — not an open lever.** The report reader maps five grades onto measured agreement rates (absent 0.08, unclear 0.28, possible 0.68, present 0.82, severe 0.94); a 2026 web sweep surfaced Rep-GLS as new, and it is what the targets already are. Re-open only with a DIFFERENT grade→value mechanism, never by re-proposing hedge softening | `core/data/labels/report_reader.py:109`; `research/deep_dives/2026-09-24_all-lanes-web-sweep.md` |
| **acquisition-physics augmentation (k-space motion / ghosting)** | **CLOSED on the shipping line (2026-09-28): `decode_cost` −0.0249 vs control −0.0223 = −0.0026, NULL in the pre-registered band; both land on foreign 0.8457.** The max-pool point below stands, so the effect is recipe-dependent; the dose is untested and unpriced, since there is no slope to follow. History: `ghosting_s2026` reads decode −0.0304 against control −0.0361 = **+0.0057 DIRECTIONAL** on the cross-site decode axis; seat −0.0002, DISTINCT (js 0.0180). Severity is NOT a dialled knob — `--ghosting 0.040` is the MEASURED median per-pixel residual of real patient motion in KMAR-50K (109 paired volumes). **Our own ruler cannot see this axis by construction**: holdout909 is drawn from the same decode as training, so it must be priced on `decode_cost`, never holdout AUC. On the shipping line (`labelwise_depth24_ghost`, 2026-09-25) holdout reads 0.9049 vs 0.9053 ledger75 = −0.0004, the blind read this row predicts; its `decode_cost` is OWED (GPU) | README:16234-16345; `experiments/after_ghosting.sh`; `kneemri/training/ghosting.py`; knee-bwqi9 |
| OneCycle | **KEEP** — but delivers half the constant it replaced; a peak is not a rate | README:815 |
| epoch budget | **CLOSED** — seven "undertrained" runs were plateaus; the budget does not reopen | README:6162, :15550 |
| EMA | **CLOSED** — buys nothing where a holdout picks the epoch. The table's coefficient had the WRONG SIGN, confounded with fold | README:606, :17219 |
| warm start | **CLOSED, flat at two rates** — the ramp was not the cause, and a 4× lower rate moved less than the floor | README:5308, :6464, :6506; `2026-09-17_warm_starting_is_flat…md` |
| LP-FT | **CLOSED** — the ID gain is a convergence-SPEED effect and a long fine-tune erases it | README:7841 |
| LoRA / PEFT | **CLOSED** — loses on the largest trunk we own; the adaptation lever closes | README:11916; knee-crh |
| `backbone-lr` | **at its optimum.** −0.0022 at 8e-5; the sign reverses at 24 epochs with dropout | knee-e9f83 |
| adaptation DEPTH (blocks unfrozen) | **CLOSED at six.** +0.0129/block from 4→6, then −0.0012 at seven | knee-p724l; README:1258 |
| stochastic depth | **CLOSED** — the published rate was wrong, and the budget was the real product | README:15508, :15550 |
| batch / accumulation | **CLOSED** — three points inside one floor | README:1805, :1997 |
| gradient clipping | measured (`clip224_tight` — clipping, **not** a CLIP trunk) | README:11669 |
| mixup | **CLOSED** — the regulariser works and the ceiling does not move | README:3977, :4244, :5388 |
| general augmentation | **CLOSED** — the regulariser fired, the overfitting did not care | README:1340 |
| curriculum | **a schedule, not a reweight** | `2026-09-01_curriculum-is-a-schedule-not-a-reweight.md` |
| mutual supervision / DML (live peer) | **CLOSED** −0.0015 on the strongest trunk; the per-epoch deficit is FLAT, not converging | README:16754; knee-z81wd |
| masked adaptation / in-domain SSL continuation | **CLOSED** — every masked adaptation this project ran switched attention OFF; the gate that catches it is a RATIO. Five "collapses" were partly a harness defect (knee-c3qfe) | README:11389, :11441, :11683, :11726 |
| **SAM / sharpness-aware minimization** | **DIRECTIONAL — read in §Open below** (`sam_r01` +0.0032 ledger75; composition P ~0.05, unarmed). The original argument: rho=0.05 refuted at −0.0421 — *"what this refutes is rho=0.05 at this budget, not sharpness-aware minimization"*. The radius is sized for from-scratch normalised weights, not a pretrained trunk at 4e-5. Deficit uniform from epoch 0; ascent witness fired live, so the harness is cleared. The honest revisit is **rho=0.01 or 0.005, once** | README:16584; knee-zqouw |
| recipe fragments (leave-one-out over the champion family) | **CLOSED entirely** — sign unanimous across seven | README:16503, :16541; knee-rn9py |

## E. Data

| axis | verdict | where |
|---|---|---|
| data VOLUME | **the only axis that ever cleared the bar** — and the slope is CONCAVE. beta = 0.22 on our own three-point curve | README:2252, :2407, :14352, :17540 |
| corpus merge / extra sites | **saturates at two sites**; `mergedk` NULL — the gain is the SITE, not the supervision | README:16346, :16398; `2026-09-03_the-corpus-axis-saturates-at-two-sites.md` |
| `--corpus-oversample` | **does not lift the mean** | README:10748 |
| label TABLE choice | **CLOSED** — ours is the worst of five extractions, and the offline ruler cannot see it. v2 wins the axis | README:14520, :17444; `2026-09-17_labels_axis_is_measured_and_v2_wins_it.md` |
| label CORRECTION / relabelling | **CLOSED at 0.1%** — founded on the wrong table; relabelling is FLAT and pruning COSTS | README:8381, :7639, :7681, :7732 |
| label MASKING | **CLOSED** — both selectors cost score, and the cost is NON-MONOTONIC | README:14976, :15045 |
| the reader / extraction side | **CLOSED** — the lane is three findings, not twelve | `2026-09-08_the_label_lane_is_three_findings_not_twelve.md`; README:15857 |
| external data as LABELLED augmentation | **CLOSED on our own measurement** — `noisy_student` 0.8548, `pool_specialise` FLAT | README:12485, :15254 |
| **GENERATED images (sampling a generative model for training data)** | **NEVER RUN, and there is no code path to run it.** `diffusion_schedule.py` (retired, last at 398eea9b) was forward-only — `noised()` and nothing else; no reverse sampler exists anywhere in the tree, so we cannot produce an image at any budget. Mechanism is also circular: pathology MORPHOLOGY is not a forward model, so any generator must be fitted on our own 4,407 studies and can only resample what the classifier already sees. Compositional lesion-insertion (the audio copy-paste route) dies at the entrance on `core/data/corpus/anatomy.py:6` — "no localisation at all", no boxes or masks — and the corpora that DO carry boxes (Rijeka ACL, fastMRI+) are BARRED for training. MRI does not superpose the way audio does, so there is no free label either | `kneemri/models/diffusion_schedule.py` (retired, last at 398eea9b); `core/data/corpus/anatomy.py:6` |
| external data for SSL pretraining | **published wash at our labelled scale** (+0.073 at 80 subjects, −0.005 at 800; we hold ~4,400) | README:7436; Zhang et al. MedIA 2024 |
| foreign TEACHER / distillation | **bounded below the gap.** Teacher headroom 0.002; the honest term is +0.0056, under the 0.0062 floor; no teacher on disk is above its student. Distillation from a foreign teacher DESTROYS distance instead of importing it. Own-OOF distillation on the labelwise shipping line: −0.0010 ledger75 (knee-bgvmd) | README:8424, :14113, :16623, :18232 |
| foreign WEIGHTS as members | **CLOSED for the CC0 replicas; RE-OPENED and SHIPPED for nartaa's OAI-supervised CoAtNet.** A cross-program equal-weight join of the 2026-09 public weights is negative. The mechanism that answers it: nartaa's trunk carries 2,399 OAI knees of external supervision and a four-source label blend, so it is off-axis to ours rather than a weaker copy. A rank blend at 1/2 is the champion `mirrorseat_v1` (board 0.953) | README:9094, :8940; knee-dbj03; `comp/champion.json` |
| better-label tables from the field (JEV, `riadmohamed42`, CC0) | **CLOSED, 2026-10-09, gold58 only.** JEV_HYBRID macro 0.9020 vs our `blend_hybrid_fx_v4b` 0.8995 vs v2 0.8659. JEV was selected on the same 58 studies, and a gold58 gain has already failed to transfer (`v4solo_v1` 0.922 vs `holdsolo_v1` 0.928) | knee-dbj03 notes 19:00Z; README:17448 |
| **PSEUDO-LABEL / teacher-relabelled training for a single model** | **OPEN, and the offline ruler is blind to it by construction.** The holdout truth IS the v2 report table, so any lever that replaces report labels is scored against the thing it replaces. Field evidence, forum 735304, 2026-10-09: a single 2.5D ConvNeXt-T reads 0.953 and DINOv2 ViT-S 0.951 with pseudo labels and no external data. A 5-fold ConvNeXt-T reads 0.959. "model is not the key. it is your labels". Our best solo is 0.946. Our only board reading is `oofsolo_v1` 0.926 vs `holdsolo_v1` 0.928, which used a leaky teacher no stronger than its student on the 0.85-era recipe. That is not this lever. Must be read on the BOARD. The arm `lw24_targetblend` (0.3·calibrated report + 0.7·`teacher_clean`) is written and blocked only on the 24sl cache, the card and the owner's go | knee-p0s4f; `research/deep_dives/2026-09-30_single-model-thread-reread-and-biohub-transfer.md`; `interpretations/kneemri/2026-10-09_levers_before_final_picks.md` |
| **training on Rijeka / MRNet / fastMRI / SKM-TEA** | **BARRED, not refuted.** A competition winner must publish weights as a public Kaggle dataset (COMPETITION_RULES.txt:187-196); a foreign-fitted member's weights are a distribution all four licences forbid. **Measurement-only use is clean** and named in the rules (:144) | `docs/EXTERNAL_CORPORA.md`; README:15671 |
| KneeCoT | **BARRED** — never shipped, in any form | owner ruling 2026-09-14 |
| transduction (pseudo-labelled holdout in training) | **CLOSED at zero shift** — ridge head: 40 cells, none clears +0.0005 (knee-tta1). Member head: six seeds, mean 0.9075 vs parent 0.9078 = TIE; the +0.0012 over control is the control's own −0.0015 drift. Shift is unsimulatable, so the real-test-set version is untested, not refuted (knee-nchjt) | README §Member-head transduction (2026-09-27) |

## F. Weight space

**Surviving rule of the whole class: AVERAGING MUST BE THE LAST OPERATION.** A one-shot soup off a plain
fit is the only form that pays, because leaving a soup is a STEP, not a slope — any resumed gradient
forfeits the averaged point in full.

| axis | verdict | where |
|---|---|---|
| uniform soup over the top-K pool | **NEGATIVE, and degrades monotonically with K** | README:15423 |
| soup of tight siblings | lands ON the same K-law, so **heterogeneity is not the driver** | README:15473 |
| basin soup — the two-term model | soup PREMIUM minus fork DEGRADATION. r1 net **+0.0014**, r2 net −0.0001 | README:17759, :17798; knee-8pj5j |
| — the premium's dependence on child divergence | **NONE.** Child spread moved 34×, the premium moved 0.0006 — inside noise | README:17759 |
| — soft fork (¼ the weight movement) | **CLOSED, and it fails its own null diagnostic.** Degradation got WORSE, premium halved | knee-grkrv — *reading in §Unread below* |
| — K sweep | **CLOSED** — k4 → k7 lands in the pre-registered "weak or decaying, record and stop" branch | README:17847; knee-gxu5r |
| `--fold -1` refit | **CLOSED** — the +0.008 was an EMA misattribution; it is ~+0.004 with spread as wide as the mean, capped below the champion. All five best bases refit and shipped | README:17492, :14296 |
| SWA | **not implemented, and priced UNDER THE BAR (2026-09-28) — do not build it as a score lever.** No `--swa` exists in our tree. The only reading we have is from a foreign recipe: variants `_swa` / `_epens` score 0.8880 against the best epoch's 0.8875, i.e. +0.0005 (README:8901). The ledger75 sweep has `raptor_v4_swa` 0.9425 against `raptor_v4` 0.9399, i.e. +0.0026 (README:9142), but that family's solo read was voided as in-sample. On our own stack the same averaging is already paid for twice: EMA is CLOSED where a holdout picks the epoch (above), and the basin-soup premium is +0.0014. Both are temporal or weight averaging over the same basin. The ceiling is therefore ~+0.001 to +0.003 on a parent, and less once it goes through a soup, since the rule above says averaging must be the LAST operation. P(solo clears 0.947) ~0.03. Re-open only as SWA *replacing* the soup's final step, measured against the soup | README:8868-8901, :9142 |

## G. Ensemble / seats / combiners

This section was headed "OUT OF SCOPE — the deliverable is a single solo checkpoint" until 2026-10-09. That is no
longer true: every champion since `tripleseat_v2` has been an ensemble, and the current one blends our seven
with a foreign seat (`comp/champion.json`). The rows below were closed over OUR pool and still bind there.

| axis | verdict | where |
|---|---|---|
| seat COUNT | **plateaus from three**; 37 members, honest peak from six seats onward | README:4194, :5805, :13157 |
| learned combiner | **LOSES to the flat vote**, re-read after its own bug was fixed | README:7543, :11041 |
| diversity as a SELECTION rule | **refuted** | README:5489, :3274 |
| averaging as an operation | **a CONTRACTION — the pool cannot produce a new board point at any seating** | README:15590 |
| the ensemble axis as a whole | **worth +0.011 and spent** | `2026-09-11_the_ensemble_axis_is_worth_0p011_and_it_is_spent.md` |
| per-finding specialists in the pool | **0 of 12 columns clear** their own 95% paired interval over the shipping solo | `2026-09-18_the_pool_holds_no_per_finding_specialist.md` |
| pool exhaustion (held-out per-label oracle) | **CLOSED** — the +0.0124 in-sample oracle is **+0.0020 ± 0.0036** held out; member ranking is honest (split-half corr 0.989) | `2026-09-20_pool_exhaustion.md` |
| foreign-seat SHARE (ours vs nartaa 384, rank blend) | **FLAT at board resolution.** Ours at 1/3 (`triseat_v1`) and 1/2 (`trihalf_v1`) both read 0.952 with goodpjw2008's reader in the wave. Without the reader, 1/2 (`mirrorseat_v1`) reads 0.953, so the reader is redundant once ours is seated. Against nartaa alone, 1/3 (`mirrorthird_v1`) also reads 0.953: flat from 1/2 to 1/3. As a SCORE lever the share is ±0.001. It is spent for MEDAL SAFETY instead: at an equal public score a lower ours share has lower private variance against the 0.950 pile, so pick 1 = the lowest share holding the best score: mirrorthird. Ours at 1/4 (`mirrorquarter_v1`) reads 0.952, one tick down, so the floor is 1/3 and the axis is spent | knee-dbj03 |
| reliability-gated share (per scanner, field, series, disagreement stratum) | **CLOSED** — ours leads the raptor replicas in every readable stratum. The only structure is per finding (Fracture to the raptor family) | `2026-10-09_no-stratum-flips-the-blend-share.md` |
| label post-processing (cross-label stack, truth-correlation coupling, per-label member reweight, max/median/power aggregation) | **CLOSED** — every form reads −0.001 to −0.019 under 5×5-fold CV on 896 studies; the models already carry the label correlations. Only per-finding SHARE reads positive, which is the tilt kernels' question | `2026-10-10_postprocessing_and_pick_risk.md` |
| more public families in the wave | heliosli compact **DEAD** (weights retired 2026-10-09 13:50Z). lixin73 MaxViT fold0 is weaker than our trio and flat in a blend. nartaa's 224-crop seat `mirrorpair_v1` read 0.952 against mirrorseat's 0.953: a second checkpoint of the same family adds nothing (closed) | knee-dbj03 |

## H. Serving

| axis | verdict | where |
|---|---|---|
| overlap TTA | **the same mechanism as whole-stack** — already consumed | README:872 |
| band TTA | **loses to the single band it averages** | `2026-09-06_band-tta-loses-to-the-single-band-it-averages.md` |
| serve-time column mixing | **NULL** | `2026-09-04_serve-time-column-mixing-is-a-null.md` |
| calibration / thresholds | **worth zero by construction** — AUC reads order only (`docs/PLAN.md`) | knee-e8a |
| efficiency lane | **CLOSED.** 0.001 AUC = 70.9 s; full-runtime-to-zero gains ~16 ticks | `2026-09-21_the_efficiency_lane_priced_and_closed.md` |
| rot3 TTA | **CLOSED on the board at two views: 0.925 with views vs 0.925 without, identical members** (`kneemri-topviews` 55607795 vs `kneemri-solotop` 55607054). The earlier blank score was an uncatchable OOM in `SpatialViews._warp`, fixed by `WARP_CHUNK=512`, not a timeout. Three views were never re-fired; with two views at 0.000 that is not a lever (P(clears) ~0.02) | knee-c9q4 |

## I. Rulers and floors — not levers, but every row above is read through them

| quantity | value | where |
|---|---|---|
| paired holdout floor (Hanley-McNeil) | **0.0053** | `2026-08-15_the-paired-floor-is-2x-not-11x.md` |
| seed spread | **0.0062** | ledger |
| board resolution | **0.001** — the competition's own | `2026-09-11_board_quantization.md` |
| board blur on tied pairs | **0.0020** | README:9783 (D86) |
| top-band residual sd | **0.0013 ledger75** (0.0017 whole record; 0.0022 was the MIDPOINT fit) — so a board delta under ~0.003 between two of our own entries means nothing | `CLAUDE.md`; `board_check --live` |
| midpoint vs ledger75 | the midpoint ruler is **seven times blinder**; a number in this repo is midpoint unless it says otherwise | README:15942; knee-f9aao |
| board transfer slope | **read it, never quote it** — `board_check --live --ruler ledger75`. It has been quoted wrong five times | `CLAUDE.md` |
| generalisation toll, holdout → board | **flat −0.0336 ± 0.0038** across eleven members — nobody is robust and nobody is fragile | README:11766 |

**The consequence that sets the whole strategy:** the toll is flat and the offline ruler *does* transfer
(Spearman ρ = 0.8818) — but every axis in tables A-H was closed **in-domain**, against a holdout, while the
board is out-of-domain to it. That is the standing reason a generalisation-side lever is priced differently
from a knob.

**A PROBE's holdout record is not on the pool's ruler.** `feature_probe` writes `truth` already binarised
(int64 {0,1}); every trained member writes the graded 7-rung table (float32). `RecordMix` therefore REFUSES a
probe against the trained pool — `competence_axis` dropped `orthodiff_mean_probe` as *"not mixable with the
majority"*. The cut is not the difference (`feature_probe.py:283` uses `> MIDPOINT`, the ledger's own cut); the
LABELS TABLE has drifted, 32 of 10,908 cells (0.29%), 3,352 probe positives against 3,336 graded. Two costs: no
probe can enter any pooled instrument, and a binarised truth **cannot be recut**, so no probe number will ever
read at `ledger75`. Same seam as knee-f9aao, one layer earlier. knee-4c9w4.

---

## Open, and unread

| item | state |
|---|---|
| **SAM at rho=0.01** | **RAN 2026-09-17 and this row was stale for five days.** `sam_r01_s2026` reads **0.8655 / 0.8981 ledger75** against `calib_dense_hold_s2026` 0.8634 / 0.8949 — **+0.0021 mid, +0.0032 ledger75**, inside the arm's own pre-registered 0.8604–0.8684 directional band. So the radius WAS the cause of the rho=0.05 collapse (all four points recovered) and the flag is **DIRECTIONAL ONLY: keep it where free, never plan around it.** Composition onto the shipping recipe (`sam_ship`, knee-k3v5t) is priced at **P ~ 0.05 and unarmed** — the give-back proxy SAM would have to recover is 0.0036 at the dense corner and **0.0006** at the shipping corner, a tenth of the 0.0062 paired floor | README:16588; knee-fw1hc, knee-k3v5t |
| `basinsoup_dmlb_ff.pt` | **READ, bottom branch** — `dmlbffsoup_v1` board **0.940** (ref 56409058) against `basinsoupff_v1` 0.946; ff-of-soup loses to soup-of-ff, and the dmlb family's better fold-0 number does not transfer | README:17990; knee-17la8 |
| `headlr_ship_ffp` | **READ, middle branch** — `headlrffp_v1` board **0.939** (ref 56493655) vs last-epoch `headlrff_v1` 0.938: +0.001, inside scatter. Stop-at-peak refits buy nothing visible; last-epoch refits stand | knee-yatne |
| `basinsoup_dmlb_k7_s2026` (solo) | **READ, top branch by the letter** — `dmlbsoup_v1` board **0.944** (ref 56500295) vs `basinsoupfold0_v1` 0.943: dmlb family is not worse at fold 0, so `dmlbffsoup_v1`'s 0.940 was the ORDER. dmlb soup-of-ff licensed but priced ~10-14 h for ~+0.001 — not queued | SUBMISSIONS.md |
| ResNet-vs-DINO decorrelation read | **READ, closed — see the CNN blind-spot row above (knee-b39kg).** `resnet_224_long_s2026` vs `basinsoup_dmlb_k7_s2026`, per-label paired bootstrap: behind on nine of twelve, ahead nowhere significantly; 50/50 average −0.0025 [−0.0067, +0.0019]. The EfficientNet question inherits that row's re-opener: an off-axis member must be competitive (~0.01 of the champion) first |
| `multiscale` under `--slice-pooling labelwise` | **CLOSED, null, 2026-09-24.** `labelwise_ms` 0.8960 ledger75 vs `labelwise_ms_ctrl` 0.9006 = **−0.0046** (336/12sl, tissue-centred, 909 studies). Multiscale is null under both max and labelwise | knee-5jfix |
| focal / sided pooling under labelwise | **ANSWERED, negative — see §D.** `focal_labelwise224` 0.8796 vs `labelwise224` 0.8832 = −0.0036, the same sign and size as `focal224`'s −0.0024 under max. Both rows of the 2×2 are negative, so there is no interaction to find and the disowned-ruler caveat is answered for the token axis rather than quoted | knee-2sqyb; `experiments/arms/focal_labelwise224.sh` |
| label-specific in-plane selection (`labelwise_patch`) | **CLOSED, null, 2026-09-23.** Per-diagnosis attention over slice×patch tokens, zero-init residual on labelwise: 0.9064 ledger75 vs control 0.9053 = **+0.0011**, inside the 0.0062 floor; the projection trained off zero (norm 12.46), so the mechanism fired. With focal/sided this closes token selection in both label-agnostic and label-specific forms. Re-open only with a LOCATION target from clean data | knee-hbta3; README "Label-specific IN-PLANE selection"; `experiments/arms/labelwise_patch.sh` |
| map-driven erasing (`--patch-maps/--patch-erase`) | **CLOSED on both families, 2026-09-24.** `erase_own_cells` null on the 224 probe (knee-0e2gd); `erase_lw24` on the shipping labelwise line 0.9057 ledger75 vs 0.9053 = **+0.0004**. Re-open only with a map from a clean localiser | knee-xxr8l; README "Map-driven erasing is null on the labelwise family too" |
| `basinsoup_soft` (knee-grkrv) verdict | **decided, negative, and written down nowhere but here** — see §F |
| `basinsoup_wide` (knee-gxu5r) verdict | **decided, middle branch** — see §F |

---

## How to maintain this

- A closed axis adds **one row**, not a paragraph. The paragraph goes in the ledger.
- A row states the verdict and points. If a row starts growing a derivation, the derivation belongs in an
  interpretation and the row shrinks back to a link.
- A re-opened axis is re-opened by **answering the mechanism in its row**, never by re-quoting a number.
  Move it to §Open with the argument, don't delete the closure.
- Numbers that live in a live tool (`board_check`, `champion_check`, `ledger75`) are **named here, not
  copied here**. Every constant this repo has quoted wrong was quoted from prose.
