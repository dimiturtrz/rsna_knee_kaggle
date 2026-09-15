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
  D:/models/rsna_knee_abnormality --only …`, CPU, holdout909.

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

Beads: everything-on seat knee-p0m58 (depends on knee-p28we, knee-amem9); decode-toll readout knee-amem9; padding rule knee-g5end. Closed-bead scan used bd list default limit 50, so it is partial.
