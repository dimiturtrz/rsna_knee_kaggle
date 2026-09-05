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

## And one of the three is not a target, because it cannot be READ

`label_power` prices the instrument rather than the model: how precisely a per-column AUC can be
measured at all, given that column's positive count. On the 881-study fold at AUC 0.85:

| finding | positives | SE | 95% half-width | our AUC | gap to 0.90 | gap in SE |
|---|---:|---:|---:|---:|---:|---:|
| MCL | 153 | 0.0202 | ±0.040 | 0.7931 | 0.107 | **5.3** |
| Lateral OA | 217 | 0.0172 | ±0.034 | 0.8156 | 0.084 | **4.9** |
| Fracture | 62 | 0.0311 | ±0.061 | 0.8088 | 0.091 | 2.9 |

MCL and Lateral OA are real, resolvable weaknesses — five standard errors is not a sampling story, and
an arm that moved either of them would be readable on the holdout we already own. **Fracture is not a
target.** Its own ±0.061 interval is two thirds of the gap being chased, so a full +0.09 lift would land
inside the width of the ruler that has to confirm it; and the ledger already recorded the consequence
before I priced it — `label_power`'s own docstring notes Fracture reading 0.784 / 0.903 / 0.864 across
three folds under an IDENTICAL recipe. That is the same column disagreeing with itself by more than the
lift an arm would be spending hours to buy.

**So the axis re-prices.** MCL and Lateral OA to 0.90 is (0.107 + 0.084) / 12 = **+0.0159 of board**,
which still clears the result bar on its own. Fracture's +0.0076 stays in the arithmetic only as
something that may come for free with a shared mechanism — never as a thing an arm is aimed at, and
never as a number quoted from a single holdout.

(The per-column AUCs are `band_dense`'s on the 909-study holdout; the SEs are the 881-study fold. The
two denominators differ by 3%, which moves an SE in the fourth decimal and none of this reading.)

## Hypothesis 3: MCL is a LABEL ceiling — and it is the exact opposite

MCL is the second-worst-agreeing column against the radiologist (50.0% of 18 calls) and the worst model
column, so "the target is noise and no model can reach 0.90" was the obvious next hypothesis. It is
wrong, and `kneemri.evaluation.supervision_gap` — which already exists, and whose docstring already
warns that *"'CEILING' IS THE WRONG WORD"* because a model fitted on noisy labels can beat them by
looking at the image — is what says so. On the 58 annotated studies, each finding scored against the
RADIOLOGIST twice, once using the report label as the predictor and once using the model:

| finding | report label | model | model − label |
|---|---:|---:|---:|
| ACL | 0.9865 | 0.8640 | **−0.1225** |
| MCL | **0.9762** | 0.8798 | **−0.0964** |
| Lateral Meniscus | 0.8422 | 0.6497 | **−0.1925** |
| Lateral OA | 0.7892 | 0.7099 | −0.0793 |
| Effusion | 0.8149 | 0.9453 | +0.1304 |

**MCL's supervision is the second-best in the corpus.** The report label predicts the radiologist at
0.9762 — a near-perfect target — and the model reaches 0.8798 from the pixels. So the 50%
positive-call agreement is not a ceiling: the reader over-calls MCL and the over-calls are *orderable*,
which is why the label still separates the radiologist's cases almost perfectly. **MCL is a model
extraction failure with excellent supervision**, which is the one diagnosis on this axis that licenses
an arm rather than closing one. Its band is one of the four of twelve that exclude zero.

Lateral OA sits on the same side (model 0.0793 below its own label) with a weaker band. So both of the
two READABLE weak columns are extraction failures, not label failures — and the label lane, which is
where I was about to spend the next reading, is the wrong lane for both.

(Denominator: 58 studies, 696 cells. The aggregate label−model gap is +0.0266 with an interval spanning
zero, so this is a per-finding shape backed by four bands, not an aggregate claim.)

## What that leaves

Three hypotheses, all killed without touching the card, and the third one inverted into a direction.
What survives on the per-label axis is **showing the model more or better image for MCL and Lateral
OA** — the supervision is already there and is not what is missing. The ledger's own geometry arms
point the same way and disagree about how far: `crop80_span` cropped IN and lost MCL hardest
(−0.0265), `crop160_span` widened OUT and gained it (+0.0086), both inside per-label noise, so the
direction is consistent across two arms and the size is unresolved. What is NOT licensed is another
loss-side tilt — `focus_mcl` put the whole encoder behind MCL and regressed its own target by −0.0576.

**And the crop route must be priced before it is read as a lead, because it is one already-closed axis.**
`README:317` closes crop/centre on the mean at both extents (80 mm −0.0099, 130 mm −0.0011) and
`crop160_span` came in FLAT — 0.8564 against `band_span`'s 0.8565. Its MCL +0.0086 is a per-label cell
inside a ±0.040 ruler, and even taken at face value a wider crop that moved only MCL is worth
0.0086/12 = **+0.0007 of board**, a fifth of the resolvable tick. So "wider helps MCL" is a direction
with two consistent signs and no size — an argument for what a real mechanism would have to look like,
NOT an arm. A crop arm queued off this paragraph would be spending hours to re-measure a closed axis.

## Hypothesis 4: fastMRI+ is the coronal localisation signal MCL needs — four positives say no

The ledger's verdict on `aprior_dense` is that "the localisation argument has to buy a real localisation
signal", and fastMRI+ looked like exactly that purchase: it is CORONAL (the plane MCL is read on), it
carries real per-finding bounding boxes, `core/data/corpus/fastmri_corpus.py` states that *"Only ACL and
MCL get supervision on BOTH sides"*, the raw volumes have been on disk since the corpus module landed,
and no fastMRI arm has ever run. Building the label table is minutes and no card:

```
build_fastmri --root .../raw/fastmri --scheme fastmri --labels-only
1173 studies read, 974 carrying a box · 1172 of 1172 exams routed, 0 on an unknown acquisition
ACL:      {0.28: 1071, 0.82: 101}
MCL:      {0.28: 1168, 0.82:   4}
Effusion: {0.28: 1030, 0.82: 142}
```

**MCL has four positives in 1,172 studies.** The both-sides claim in the docstring is true and worth
nothing: the positive side is four exams. Every other finding is absence-only. So the coronal
foreign-supervision route **cannot touch either readable weak column** — not MCL at n=4, and not Lateral
OA, which fastMRI+ does not annotate at all. Killed at the gate, before an epoch.

What the corpus CAN supervise is ACL (101 positive, 1,071 assessed negative) and Effusion (142). Neither
is a weak column on our ruler — ACL 0.9032, Effusion 0.8683, both above the pool mean. ACL is however the
largest extraction gap in the `supervision_gap` table above (model 0.1225 below its own label against the
radiologist), so a fastMRI ACL arm is not empty — it is just a different question from this one, and it
belongs to whoever prices the gap between our ruler and the board's truth. Filed, not queued.

The open question, and the one now with Kimi as Q-COL: which plane and sequence MCL is actually read on
by a radiologist, and whether any published knee-MRI system routes a finding to its plane rather than
weighting its loss. A DECODE tilt (an MCL head that attends the coronal slots) is a different mechanism
from the LOSS tilt that failed, and `core/data/corpus/anatomy.py` already carries the per-finding
plane × fluid-sensitivity prior it would ride on.

**What must NOT be concluded from this:** it does not say the per-label axis is closed. It says two of
its mechanisms are, and it removes the two that were free. The +0.0196 arithmetic in the earlier
interpretation stands unchanged — it is a statement about what those three columns are worth, not a
claim that any particular operation reaches it.
