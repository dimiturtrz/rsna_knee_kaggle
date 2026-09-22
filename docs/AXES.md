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
- `interpretations/kneemri/2026-09-18_state_of_the_levers.md` — the levers still OPEN, priced against the board gap.
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
| capacity ladder — base / large over small | **CLOSED** −0.0010; five isolating pairs, median 0.0000 | knee-5dx47; README:1755 `base_dense`, :3758 base224 |
| DINOv3 | **CLOSED.** "DINOv3 is weak" was an ADAPTATION finding wearing an encoder's name; best honest v3 member still trails the seats | README:1924, :3036; `2026-09-17_member_candidates.md` |
| ConvNeXt | **CLOSED** — and the inherited rate cost the whole trunk before the own-recipe re-run | README:1841 `cnx_dense`, :3326 `cnx_own_recipe`, :6004/:6659 `cnx_mid_rate` |
| ResNet | **CLOSED** — peaked epoch 12 of 48, so the budget was never the constraint; the ViT recipe carries less than half onto it | README:12872, :17087 `resnet_224_ext` |
| the CNN axis as a class | **CLOSED on its own named re-opener, card-free** | README:7796; `2026-09-05_the-cnn-axis-is-maximally-distinct-and-still-worthless.md` |
| RadImageNet DenseNet | **CLOSED** — closes the convolutional trunk lane empty | README:6685 |
| RAD-DINO (medical-report backbone) | **CLOSED** −0.037 against its own exact control | README:2790, :4291 |
| Swin (supervised) | **CLOSED**; block discovery had found ONE stage on a staged trunk | README:6130, :6374; knee-94ec |
| ViT-in21k | **CLOSED** on a trajectory instrument | README:6196 |
| EfficientNetV2-L | **NULL, not a loss** — the 44-slot corpus takes it | README:9106 |
| V-JEPA (volume pretraining) | **CLOSED** — a TIE on frozen features; the finding is the DISTANCE, not the score | README:2542, :2590; `2026-08-22_vjepa_frozen_probe.md` |
| MRI-core in-domain trunk | **CLOSED** as a queue item, 0.8390 | README:8107 |
| architecture monoculture as a seat lever | **CLOSED, negative six of six** | README:10988; `2026-08-28_architecture-monoculture.md` |
| trunk axis inside the foreign recipe | **CLOSED**, and a fit predicted it out of sample | README:9236 (D36) |

## B. Input geometry

| axis | verdict | where |
|---|---|---|
| resolution 224 / 336 / 392 | **CLOSED.** 224 costs 0.0049 solo and buys 3.2× throughput; 336→392 does not lift the meniscus columns | README:8327, :7915, :3729 |
| field of view / band width | **CLOSED** — the FOV axis closes and MCL is why | README:2032 `band140`; `2026-09-09` FOV census README:12178 |
| crop CENTRE | **CLOSED** — moving it at the control's extent does nothing, which is the point | README:517 |
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
| AUC-margin surrogate (LibAUC / Deep AUC Maximization) | **CLOSED** — `margin_seat` −0.0053 solo. This is the literature's single best loss offer for imbalanced medical AUC, and it is spent | README:374; knee-bdhb |
| rank / dissent objective as a seat lever | **CLOSED** — 38th of 66; the per-label table says why | README:463 |
| per-label loss weighting | **refuted twice**, once from the training side | README:567, :1145 |
| severity weighting | **CLOSED** — the knob fired and moved the graded findings the WRONG way | README:1476 |
| ordinal supervision (`grade_ladder`) | **lands, and buys nothing the roster can see** | README:2067, :2700 |
| focal | **CLOSED** — buys nothing on the mean | README:1224; `2026-08-25_the-focal-deficit.md` |
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
| **SAM / sharpness-aware minimization** | **OPEN.** rho=0.05 refuted at −0.0421 — *"what this refutes is rho=0.05 at this budget, not sharpness-aware minimization"*. The radius is sized for from-scratch normalised weights, not a pretrained trunk at 4e-5. Deficit uniform from epoch 0; ascent witness fired live, so the harness is cleared. The honest revisit is **rho=0.01 or 0.005, once** | README:16584; knee-zqouw |
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
| external data for SSL pretraining | **published wash at our labelled scale** (+0.073 at 80 subjects, −0.005 at 800; we hold ~4,400) | README:7436; Zhang et al. MedIA 2024 |
| foreign TEACHER / distillation | **bounded below the gap.** Teacher headroom 0.002; the honest term is +0.0056, under the 0.0062 floor; no teacher on disk is above its student. Distillation from a foreign teacher DESTROYS distance instead of importing it | README:8424, :14113, :16623 |
| foreign WEIGHTS as members | **CLOSED** — released foreign weights are not a member lever; a cross-program equal-weight join is negative | README:9094, :8940 |
| **training on Rijeka / MRNet / fastMRI / SKM-TEA** | **BARRED, not refuted.** A competition winner must publish weights as a public Kaggle dataset (COMPETITION_RULES.txt:187-196); a foreign-fitted member's weights are a distribution all four licences forbid. **Measurement-only use is clean** and named in the rules (:144) | `docs/EXTERNAL_CORPORA.md`; README:15671 |
| KneeCoT | **BARRED** — never shipped, in any form | owner ruling 2026-09-14 |

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
| SWA | **not implemented** — no `--swa` anywhere in the tree | — |

## G. Ensemble / seats / combiners — OUT OF SCOPE

The deliverable is a **single solo checkpoint**. This class is kept for the mechanisms, which still bind.

| axis | verdict | where |
|---|---|---|
| seat COUNT | **plateaus from three**; 37 members, honest peak from six seats onward | README:4194, :5805, :13157 |
| learned combiner | **LOSES to the flat vote**, re-read after its own bug was fixed | README:7543, :11041 |
| diversity as a SELECTION rule | **refuted** | README:5489, :3274 |
| averaging as an operation | **a CONTRACTION — the pool cannot produce a new board point at any seating** | README:15590 |
| the ensemble axis as a whole | **worth +0.011 and spent** | `2026-09-11_the_ensemble_axis_is_worth_0p011_and_it_is_spent.md` |
| per-finding specialists in the pool | **0 of 12 columns clear** their own 95% paired interval over the shipping solo | `2026-09-18_the_pool_holds_no_per_finding_specialist.md` |

## H. Serving

| axis | verdict | where |
|---|---|---|
| overlap TTA | **the same mechanism as whole-stack** — already consumed | README:872 |
| band TTA | **loses to the single band it averages** | `2026-09-06_band-tta-loses-to-the-single-band-it-averages.md` |
| serve-time column mixing | **NULL** | `2026-09-04_serve-time-column-mixing-is-a-null.md` |
| calibration / thresholds | **worth zero by construction** — AUC reads order only (`docs/PLAN.md`) | knee-e8a |
| efficiency lane | **CLOSED.** 0.001 AUC = 70.9 s; full-runtime-to-zero gains ~16 ticks | `2026-09-21_the_efficiency_lane_priced_and_closed.md` |
| rot3 TTA | returned COMPLETE with a blank score — **untested, not measured** | knee-c9q4 |

## I. Rulers and floors — not levers, but every row above is read through them

| quantity | value | where |
|---|---|---|
| paired holdout floor (Hanley-McNeil) | **0.0053** | `2026-08-15_the-paired-floor-is-2x-not-11x.md` |
| seed spread | **0.0062** | ledger |
| board resolution | **0.001** — the competition's own | `2026-09-11_board_quantization.md` |
| board blur on tied pairs | **0.0020** | README:9783 (D86) |
| top-band ledger75 residual sd | **0.0022** — so a board delta under ~0.004 between two of our own entries means nothing | `CLAUDE.md`; `board_check --live` |
| midpoint vs ledger75 | the midpoint ruler is **seven times blinder**; a number in this repo is midpoint unless it says otherwise | README:15942; knee-f9aao |
| board transfer slope | **read it, never quote it** — `board_check --live --ruler ledger75`. It has been quoted wrong five times | `CLAUDE.md` |
| generalisation toll, holdout → board | **flat −0.0336 ± 0.0038** across eleven members — nobody is robust and nobody is fragile | README:11766 |

**The consequence that sets the whole strategy:** the toll is flat and the offline ruler *does* transfer
(Spearman ρ = 0.8818) — but every axis in tables A-H was closed **in-domain**, against a holdout, while the
board is out-of-domain to it. That is the standing reason a generalisation-side lever is priced differently
from a knob.

---

## Open, and unread

| item | state |
|---|---|
| **SAM at rho=0.01 or 0.005** | **OPEN** — the one live mechanism in table D; implemented, never tried at the right radius. Needs a bead carrying `PRICE:` and `P(clears …)` before `arm_check` passes it |
| `basinsoup_dmlb_ff.pt` | **UNREAD** — on disk since 2026-09-21 03:00. `--fold -1`, so readable only by a slot |
| `headlr_ship_ffp` | complete (`best e- of 18 run, 24 asked` is its design); awaiting a board slot as a declared INSTRUMENT |
| ResNet-vs-DINO decorrelation read | **free, unrun** — gates the owner's EfficientNet question |
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
