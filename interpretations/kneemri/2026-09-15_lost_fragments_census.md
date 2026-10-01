# Lost fragments census — what was switched off, and what an everything-on seat would carry

2026-09-15. Zero card, zero slots. Owner question: *"are there other lost and not found solution fragments
that were thrown away or permanently switched off?"*

## Scope and the classification rule

Base = the current seat recipe, `experiments/arms/labelwise_dense_drop40_long.sh` plus `$RECIPE`
(`experiments/recipe.sh:74`): labelwise pooling, 20 slices, band 0.1–0.9, calibrated targets, held
annotations, slot dropout 0.40, 24 epochs, one-cycle, weight floor, rank weight 1.0. Every flag
below defaults OFF in `kneemri/training/options.py` (lines quoted in the table) and is absent from all
six champion arm lines (read from the arm scripts, 2026-09-15).

Classification follows the owner steer of 2026-09-15: *"most of this stuff is obviously a good idea and
with a longer training schedule -0.003 old result is not a problem"*.
- **COMPOSABLE**: sound mechanism, delta ≥ about −0.005 on an older or shorter recipe.
- **CONFLICTS**: replaces or overlaps a mechanism already in the base.
- **DEAD**: a named mechanism refutes it, or it lost worse than −0.008 on a comparable recipe.
- **LICENCE**: the fragment cannot enter a pick.

Sources:
- Midpoint deltas are the ledger's own (`experiments/README.md`).
- The ledger75 column was measured for this note: `python -m kneemri.evaluation.ledger75 --holdouts
  D:/data/volumetric/mri/models/rsna_knee_abnormality --only …`, CPU, holdout909.

## The ledger75 re-read

| member | LEDGER | LEDGER75 | epoch |
|---|---|---|---|
| labelwise_dense_drop40_long_s2026 (base) | 0.8672 | 0.9048 | 23 |
| dropstack_light_s2026 | 0.8654 | 0.8974 | 10 |
| calib_dense_hold_s2026 | 0.8634 | 0.8949 | 6 |
| certainty_ramp_s2026 | 0.8584 | 0.8885 | 8 |
| band_dense_s2026 | 0.8578 | 0.8877 | 8 |
| band_dense_hold_s2026 | 0.8573 | 0.8877 | 7 |
| ema_dense_s2026 | 0.8588 | 0.8857 | 7 |
| mixup_dense_s2026 | 0.8552 | 0.8849 | 11 |
| ghosting_s2026 | 0.8576 | 0.8838 | 10 |
| disagreed_uniform_s2026 | 0.8507 | 0.8819 | 11 |
| relabel_hold_s2026 | 0.8515 | 0.8812 | 8 |
| aug_long_s2026 | 0.8511 | 0.8798 | 9 |

**Finding: the curriculum's +0.0077 was read against the weak control.**
- `certainty_ramp`'s +0.0077 (README:6695) is taken against `disagreed_uniform` 0.8507.
- Against `band_dense` (0.8578), the ledger's actual uniform draw (README:6677), the same member reads +0.0006 midpoint and **+0.0008 ledger75**.
- The +0.0077 is mostly the gap between two controls, not the curriculum. That matches `disagreed_ramp`'s own +0.0008 ledger75 (README:12867).
- The schedule mechanism stays COMPOSABLE: it is sound and costs nothing. It is not a measured lever, and knee-p28we must not be sold on +0.0077.

## Census

| fragment | flag / commit | measured delta (cut, era, source) | class | why it was off |
|---|---|---|---|---|
| stochastic depth, light | `--stochastic-depth 0.05` (options.py:229) | +0.0020 mid vs calib_dense_hold, 12 ep max corner (README:15412). **+0.0025 l75** (this note). At 24 ep, −0.0013 vs itself (README:15447) | COMPOSABLE | the budget-lever reading closed at max pooling. Never tried on labelwise or with a ramp |
| curriculum (easy→hard) | `--certainty-ramp 6` (options.py:196) | +0.0077 mid vs weak control (README:6695). **+0.0008 l75** vs band_dense (this note) | COMPOSABLE | used in 2 arms, never ported to a seat. Now knee-p28we on the shared schedule |
| weight EMA | `--ema` (options.py:131) | +0.0010 mid (README:4643). −0.0020 l75 (this note). +0.0013 on the last epoch where no holdout picks (README:592) | COMPOSABLE | flat where a holdout picks the epoch. Pays in last-epoch or fold −1 fits |
| motion ghosting | `--ghosting 0.04` (options.py:89) | −0.0002 native mid. **Cross-site decode toll +0.0057 shallower** (README:16128-16129). −0.0039 l75 (this note). Never dose-scanned (README:16236) | COMPOSABLE | the first dose did not earn a scan |
| intensity aug | `--intensity-aug` (options.py:132) | −0.0064 @392 (README:327). −0.0076 at 24 ep (README:328). −0.0002 as a seat ADD (README:11039) | COMPOSABLE (ramped, owner steer) | lost on the in-distribution ruler. Re-opened for the cross-site ruler and never asked (README:11083) |
| mixup | `--mixup 0.2..0.4` (options.py:80) | +0.0014 α0.4 @224 (README:3961). −0.0026 dense (README:5371). −0.0028 l75 (this note). −0.0088 long @224 (README:4233) | COMPOSABLE at low α, ramped | a regulariser that costs convergence at 12 ep |
| wide scale + reflect pad | `--scale-jitter 0.25 --padding-mode reflection` | decode +0.0042, flat band (README:15958). Padding rule UNANSWERED: fovjitter160_reflect killed 6/12 (README:12057) | COMPOSABLE | retired as a lever at that dose, not refuted as a mechanism |
| long warmup | `--warmup 0.25` | +0.0010 @224 (README:11610) | COMPOSABLE (knob) | noise-sized |
| anatomical prior | `--anatomical-prior` | −0.0021 vs band_dense (README:329). −0.0076 on another head pairing (README:280) | COMPOSABLE (weak) | added nothing over the learned head |
| unread-targets relabel | `--unread-targets` (options.py:143) | gold-58 +0.0104 [−0.0039, +0.0241] (README:8987). −0.0065 l75 holdout vs band_dense_hold (this note), a ruler scored on the labels it corrects | COMPOSABLE (label side) | one arm, inside its interval |
| through-plane decode | `--through-plane` | +0.0005 / +0.0065 / +0.0033 on three rulers (README:5077). In champion seat through_dense, not in labelwise seats | COMPOSABLE, untested with labelwise | never crossed with labelwise pooling |
| raptor distillation | `--distill-weight 0.3 --teacher-targets raptor_teacher.npz` | champion seat stack_increments. Solo delta UNVERIFIED here | COMPOSABLE, provenance caveat | teacher exposure (knee-wko44 notes). Must stay an out-of-fold teacher |
| layer decay | `--layer-decay 0.7` | −0.0004 (README:11041-11042) | COMPOSABLE (knob) | flat |
| weight decay / clip | `--weight-decay 0.1`, `--grad-clip 0.5` | −0.0015 / −0.0014 @224 (README:11612-11613) | COMPOSABLE (knob), low priority | flat |
| severity weight | `--severity-weight` | −0.0025 (README:4546) | COMPOSABLE (weak) | directional only |
| latent-MIM adaptation | `latent_mim` pretrain | +0.0014 on v3small (README:11699) | CONFLICTS | a different trunk stage from the DINOv2-small seat |
| region pooling | `--slice-regions 4` | +0.0016 (README:5076), −0.0029 (README:310). A SUBSTITUTE for span (README:315) | CONFLICTS | overlaps band 0.1–0.9 and labelwise pooling |
| attend / moments pooling | `--slice-pooling attend/moments_wide` | −0.0009 (README:319), −0.0034 (seated for diversity) | CONFLICTS | replaces labelwise pooling |
| grade ladder | `--grade-ladder` | −0.0013 (README:1528) | CONFLICTS | another target shape beside calibrated targets |
| AUC-margin loss | `--auc-margin 1.0 --rank-weight 0` | −0.0053 solo, seats (README:350) | CONFLICTS | replaces the rank term the recipe measured positive 3 of 3 |
| per-label loss norm | `--per-label-loss` | −0.0043, falsifiable clause WRONG SIGN (README:348) | DEAD | mechanism prediction refuted |
| SupCon contrast | `--contrast-weight 1.0` | −0.0087, batch suspect refuted (README:6018, 6053) | DEAD | loss > 0.008, harness suspect cleared |
| side ranking | `--side-weight 1.0` | −0.0641, term verified landed (README:2740) | DEAD | large loss |
| pair memory | `--pair-memory` | −0.0302 (README:345) | DEAD | large loss |
| confident-joint pruning | `--label-errors` | −0.0158 (README:353) | DEAD | large loss |
| 80 mm crop | crop80_span | −0.0099 (README:316) | DEAD | large loss |
| flips | augmentation.py:71 | laterality broken for 5 of 12 labels | DEAD | mechanism |
| KMAR-50K merge | mergedk_corpus | NULL (README:16152 section) | COMPOSABLE data, low value | pseudo-labelled, read null |
| SKM-TEA / fastMRI / Rijeka pretrain or locator | — | instrument only | LICENCE | owner ruling knee-vgz5b.7 |
| KneeCoT | — | — | LICENCE | never |

`comp/members.declines` was swept for weak reasons.
- 393 entries.
- The decline lines without a measured number are smoke or probe runs (declines:65-75) or stage-one trunks (declines:160,171).
- No lost member turned up there.

## What this licenses, and what it does not

- **No single fragment clears the 0.0062 floor on ledger75.** The largest is `dropstack_light` at +0.0025.
- **The deltas do not sum.** Several are regularisers that each spend the same convergence budget. Four measured at the max corner were individually flat.
- **The case for the everything-on seat is the owner's.** Longer budget plus a ramp that holds the heavy operators off early lets them stop fighting convergence, the failure mode `dropstack_light` exposed (README:15439-15456). No measured sum backs it.
- **Two of the fragments carry their signal on the CROSS-SITE ruler, which the own-holdout cannot see:** ghosting and intensity aug. That is the reason to read the arm on the decode toll too.
- **The +0.0077 curriculum number is retracted as a lever size.** Quote +0.0008 ledger75 vs band_dense.

Beads: everything-on seat knee-p0m58, now superseded by epic knee-feb62 (depends on knee-p28we, knee-amem9); decode-toll readout knee-amem9; padding rule knee-g5end.

## Closed-bead rescan — all 681 (knee-feb62.1)

The first pass read `bd list` at its default limit of 50. This one read
`bd list --status=closed --limit 0` (681 rows), and then the notes of every candidate. `comp/members.declines`
was already complete (393 of 393). New rows only; the classification rule is the same as above.

| fragment | source | measured delta (cut, source) | class | note |
|---|---|---|---|---|
| tissue-centre crop | knee-yrf, knee-2zbi | tissue kept +0.0294 [+0.0276, +0.0312] at 130 mm (knee-2zbi). `tissue224` −0.0011 mid vs native224 (README:317), −0.0004 as a seat add (README:11040) | COMPOSABLE (data geometry) | switching it needs the tissue-mask cache at 336 px, and nothing in the 336 px cache has one (UNVERIFIED) |
| 140 mm FOV | knee-n3yx, band140 | −0.0022 mid vs band_dense, MCL −0.016..−0.021 (README:2007-2020). −0.0003 as a seat add (README:11036) | COMPOSABLE (weak) | the periphery hypothesis read the wrong sign on its own label. Low priority |
| layer-wise LR decay + unfreeze 12 | knee-a2q7, `band_dense_llrd` | −0.0004 seat add (README:11042). ~580 s an epoch, 2x the base (README:2093) | COMPOSABLE (knob), costly | this is the `--layer-decay` row above. Unfreeze 12 is 2x wall |
| unfreeze depth 12 at the winning rate | knee-3om | −0.0007 vs depth 6 (bead notes) | COMPOSABLE (knob) | noise |
| one_cycle off | knee-fkw5, `nocycle224_long` | −0.0061 best, peak e9→e17, identical finish 0.8326 vs 0.8325 (README:4082-4095) | DEAD for the peak | keep one_cycle. The slide is overfitting, not the schedule, and that is the case for regularising a long run |
| resolution 392 | knee-dft, knee-wnkv | +0.0002 at a matched rate (bead notes). 28.4 GB VRAM | COMPOSABLE (knob), costly | not worth the VRAM against the 50-70% cap |
| cls_mean_focal pooling | knee-66c | −0.0025 vs whole_stack, seat −0.0012 (bead notes) | CONFLICTS | replaces labelwise pooling |
| order-aware depth aggregator | knee-tqn, `depth_true` | 0.8461, seat −0.0002 (bead notes) | CONFLICTS | replaces labelwise pooling |
| per-slice max + CNN | knee-wif | +0.0018 for 3x encoder cost (bead notes) | CONFLICTS | pooling and trunk change |
| loss divided by element count | knee-sbr | −0.0022 (bead notes) | COMPOSABLE (knob), no reason to | the weight floor already measured better |
| registered reformats | knee-vigl | never trained | not composable now | a data build, not a flag |
| overlap TTA | knee-36y | +0.0025 (bead notes) | already ON | inference, in both kernels |
| member count 20 | knee-uxg | best solo beats a 3-member ensemble (bead notes) | out of scope | folds and seeds are banned |

Net: the rescan adds no composable training flag beyond the table above. The two new composable ones,
tissue centre and 140 mm, are data geometry that needs a cache the 336 px recipe does not have. The one
mechanism it adds is the `nocycle` result. The late slide is overfitting, and one_cycle stays. That is
the input to the horizon section below.

## Horizon and curriculum shape — derived, not swept (knee-feb62.3)

### Evidence

| # | fact | source |
|---|---|---|
| E1 | Base `labelwise_dense_drop40_long` epoch trajectory: e6 0.8491, e9 0.8534, e12 0.8623, e17 0.8658, e18 0.8669, e23 0.8672. The last six epochs span 0.0010. Train loss still falls (0.5965 → 0.5587), so the plateau is holdout-side | `$MODELS/experiments/labelwise_dense_drop40_long_s2026.full.log` |
| E2 | 185 s an epoch at steady state (224 s for e0), so 24 epochs = 1.25 h | same log |
| E3 | drop40 at 12 → 24 epochs is +0.0059 l75, peak on the last epoch, CONVERGED | knee-cgt9m, knee-a21h9 |
| E4 | Without a regulariser the budget is not a lever. `dropstack_long` 24 ep 0.8641 vs 12 ep 0.8654. `labelwise_long24` retired unrun on that | README:15445-15447, 15684-15689 |
| E5 | Unregularised 224 at 24 epochs slides 14 epochs after e9 (0.8490 → 0.8325). Removing one_cycle only moves the peak (e17) and leaves the finish identical | README:4082-4095 |
| E6 | Regularisers move the peak later: aug_long e6 → e9; mixup_dense peaks at its last epoch; dropstack_light peaks at its last epoch while its control peaked at e6 | knee-gawj, knee-x580, README:15439-15456 |
| E7 | 8 → 24 epochs +0.0068 on the windowed recipe | knee-pxa |
| E8 | EMA +0.0010 mid / −0.0020 l75 where a holdout picks the epoch; +0.0013 on the last epoch | knee-xzr, knee-56w5, README:592 |
| E9 | The certainty ramp was measured over 6 of 12 epochs (share 0.5) | options.py:196, README:6695 |
| E10 | No SWA flag in `kneemri/training/options.py`; `ema` is the only weight average (options.py:131) | grep, 2026-09-15 |

### Derivation

**Epochs: 36.**
- E3 and E4 together say budget pays only when a regulariser holds off overfitting. The grand recipe stacks
  more regularisers than drop40 alone. E6 says each one moves the peak later.
- E1 says drop40 by itself reached its plateau at about e18, 75% of its budget.
- Under the ramp below, the first quarter runs at reduced dose and the last fifth tapers. That leaves
  about 20 epochs at full dose, which is roughly the 18 epochs drop40 needed to plateau at its own full dose.
  That is the argument for 1.5x.
- 48 is not chosen. E5 shows what a surplus budget does when regularisation falls short: 14 epochs of slide.
  The anneal does not rescue it, so the extra third would buy risk, not signal. It would also cost 2.5 h.
- 30 is not chosen: it would squeeze the full-dose plateau under the 18 epochs E1 measured.

**Augmentation schedule** (`--aug-start 0.25 --aug-mid 1.0 --aug-end 0.5 --aug-ramp 0.25 --aug-tail 0.2`).
- ramp 0.25 = 9 epochs. That covers the fast-fit phase, which E1 puts at e0–e9 (0.7305 → 0.8534). Holding
  heavy operators off there is the owner's mechanism, and dropstack_light's failure mode (README:15439-15456).
- start 0.25, not 0: a clean start and then a dose shock makes the ramp itself a distribution shift.
  A quarter dose keeps the operators live from step one. This is a judgment call with no measurement behind
  it, stated as such.
- tail 0.2 = 7 epochs, the length of drop40's flat top (E1, e17–e23). The low-LR anneal then consolidates
  on inputs closer to the test distribution.
- end 0.5, not 0: ghosting and intensity carry their signal on the cross-site ruler (the census above).
  Annealing them to zero would train that axis out in the final epochs.

**Hardness schedule** (`--hard-start 0 --hard-mid 1.0 --hard-end 1.0 --hard-ramp 0.25 --hard-tail 0`).
- start 0: easy, trusted cells first. That is the certainty-ramp direction, E9.
- ramp 0.25 = 9 epochs, at least the 6 epochs the ramp was measured over.
- The ramp is shared with aug. Hardness and augmentation both reach full at e9, so the model meets full
  dose and full labels in the same phase. Staggering them would need a measurement nobody has.
- end 1.0, no tail: hardness 1.0 is the ledger's label weighting, and the ruler scores every cell.
  Annealing back toward easy cells would fit the final epochs to a label distribution the holdout does not
  have. The asymmetry with aug is deliberate: aug is an input corruption, and toning it down moves toward
  test inputs; hardness is the target itself.
- The +0.0077 curriculum number is not the reason for any of this (retracted above, +0.0008 l75).

**EMA: ON.** SWA: no flag exists (E10), so it is not in the recipe, and adding one is code the preset does
not need. EMA is flat to −0.0020 when a holdout picks the epoch (E8), but in a 36-epoch run with a tapered tail
the pick should land in the last few epochs, where EMA read +0.0013. It is also the only protection a
fold −1 fit gets. The risk is the l75 −0.0020, which sits inside the floor.

**Stop rule, not a sweep.** Run `trajectory_check` live against the pre-registered target. Run
`convergence_check` after. UNDERTRAINED, meaning a peak at e35 with a still-rising tail, is the one outcome
that re-opens the horizon; everything else is read as-is.

### Card-hours for knee-feb62.4

36 x 185 s = 6660 s = **1.85 h** at the base's measured step (E2). The heavier operators add unmeasured
per-step cost (mixup, ghosting, intensity, stochastic depth), so budget **2.0–2.5 h**; the overhead is
UNVERIFIED until the knee-feb62.2 smoke reports s/epoch. The ~280 s an epoch in knee-ffku5 and knee-pfqtw
belongs to other recipes, not this one.
