# The three weak columns are not a missing sequence, and not a combination we have not tried

**Date:** 2026-09-05 · **Card spent:** none · **Slots:** none · **Beads:** adjacent to knee-wko44

`2026-09-05_the-metric-is-a-plain-mean-and-three-columns-hold-two-points.md` priced the per-label axis
at **+0.0196 of board** if MCL, Fracture and Lateral OA are brought to 0.90, and closed post-hoc as a
class. This is the next two hypotheses on that axis, both killed card-free, before an arm was queued.

## Hypothesis 1: the weak columns are the findings whose SEQUENCE we do not hold

The mechanism is physical and it is the first one to check, because it would be a data defect rather
than a modelling one: MCL is read coronally, a fracture is marrow oedema on a fluid-sensitive
fat-suppressed sequence, and osteoarthritis is morphology that reads best on T1. If a study's cache is
missing that slot, the finding is not weakly visible — it is absent, and no trunk recovers it.

The corpus is not evenly covered, so the hypothesis had somewhere to live. Presence over all 4,407
studies in `mask_336px_20sl_0.1-0.9band.npy`:

| slot | present |
|---|---:|
| AX_FLUID_FS | 0.985 |
| COR_FLUID_FS | 0.955 |
| SAG_FLUID_FS | 0.935 |
| COR_T1 | 0.641 |
| SAG_FLUID_NOFS | 0.626 |
| SAG_T1 | 0.425 |

The three fluid-suppressed planes are near-universal; the T1 slots are held by four studies in ten to
six in ten. Mean 4.57 of six slots per study, and only 38 studies of 4,407 hold fewer than three.

**Conditioned per-column AUC says the effect is not there.** `band_dense`'s 909-study holdout record,
each column scored separately on the studies that HOLD a slot and on the studies that do not:

- **MCL** (0.7931 overall): SAG_FLUID_FS 0.796 present / 0.787 absent · SAG_FLUID_NOFS 0.802 / 0.784 ·
  COR_T1 0.803 / 0.781 · SAG_T1 0.777 / 0.805. Every split is inside a hundredth, and the coronal T1
  split goes the *wrong* way by less than the noise.
- **Lateral OA** (0.8156): COR_T1 0.799 / 0.810 — again the wrong sign, again tiny. The two fluid
  splits read 0.82 / 0.75 but on 37 and 65 absent studies, which resolves nothing.
- **Fracture** (0.8088): SAG_T1 0.759 present / 0.831 absent on 526 absent studies — the largest split
  in the table and it says a study that HOLDS the sparse T1 slot is read *worse*. That is a selection
  effect on which studies get a long protocol, not a sequence effect.

No column's weakness tracks the availability of the plane or the weighting it is read on. **The
missing-sequence hypothesis is dead**, and with it per-finding sequence routing as a fix for these
three columns — there is nothing to route around. It stays alive only as an architecture question
(does a finding-conditioned query help when every slot IS present), which is a different claim and a
much smaller one.

## Hypothesis 2: some member already reads these columns well and the pool averages it away

**This was already refuted and I re-derived it.** `experiments/README.md:6871` — *"And the specialist
route was already refuted, at maximum tilt"* — and `:6094`, which records the per-column specialist
increment at **+0.0043** and explicitly does not re-arm it. `record_mix --routing` is the existing
instrument, with a split-half honest estimator; what follows is an in-sample upper bound computed
without it. Keep it as a CONFIRMATION of the recorded null and nothing more. The rule that would have
saved the hour is in `CLAUDE.md`: read the ledger and the interpretations before filing a finding.

The board is a plain mean of twelve per-column AUCs, so a per-column combination is legal, and nothing
in the closed post-hoc result forbids it: `kneemri-monotone` closed *within-column monotone transforms
of one table*, and `column_mixing` closed *cross-column* mixing. Choosing a different member per column
is neither — it is the specialist route, arriving by a different road at the same number.

108 of the 125 roster members carry a holdout record on the same 909-study ruler. Rank-average of all
108 against the best single member in each column, **selected on the test set itself**:

| finding | pool of 108 | best single member | gain |
|---|---:|---:|---:|
| MCL | 0.8103 | 0.8253 `merged_blend` | +0.0150 |
| Fracture | 0.8279 | 0.8440 `merged_blend_cf` | +0.0161 |
| Medial OA | 0.8862 | 0.8968 `labelwise_dense` | +0.0106 |
| Baker's | 0.9186 | 0.9279 `merged_dense_cf` | +0.0093 |
| PF OA | 0.8651 | 0.8733 `stackfit` | +0.0082 |
| Lateral OA | 0.8329 | 0.8386 `stack_increments_base` | +0.0057 |
| Medial Meniscus | 0.9084 | 0.9135 | +0.0052 |
| Synovitis | 0.8773 | 0.8823 | +0.0050 |
| Lateral Meniscus | 0.8654 | 0.8700 | +0.0046 |
| Contusion | 0.8876 | 0.8867 | −0.0010 |
| Effusion | 0.8683 | 0.8640 | −0.0043 |
| ACL | 0.9032 | 0.8976 | −0.0056 |

Summed over the positive columns that is +0.0798 of column AUC, i.e. **+0.0066 of board — and that is
an upper bound obtained by picking each winner on the very studies it is then scored on**, over 108
candidates, with no nesting and no penalty. An honest nested selection is a fraction of it, and the
ledger's honest number for the same route is **+0.0043**, which sits inside this bound exactly as it
should. It is under half the +0.0150 a single result has to clear and about the size of the pure-refit
term that was already called dead.

**So the pool is saturated per column, not just in the mean.** The best member of 108 beats the average
of 108 by a hundredth or two in the weak columns and loses in three others. There is no member hiding
in the roster that reads MCL or Fracture well; every member reads them badly.

## What that leaves

Both hypotheses were the cheap ones, and both are gone without touching the card. What survives on the
per-label axis is training-side and shared: the weak columns are weak for **every member we own**,
which points at the supervision or the task rather than the trunk, the plane, or the mix. The two
readings that would separate those are (a) whether MCL's 0.81 is a label ceiling — it is the
second-worst-agreeing column against the radiologist at 50.0% of 18 calls, and it is the worst model
column — and (b) whether a member trained with the weak three upweighted moves them without costing
the other nine. Neither is priced yet, and (a) is card-free.

**What must NOT be concluded from this:** it does not say the per-label axis is closed. It says two of
its mechanisms are, and it removes the two that were free. The +0.0196 arithmetic in the earlier
interpretation stands unchanged — it is a statement about what those three columns are worth, not a
claim that any particular operation reaches it.
