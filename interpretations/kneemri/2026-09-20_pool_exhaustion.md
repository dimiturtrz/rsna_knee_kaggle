# The member pool is exhausted, and the holdout says so honestly

2026-09-20, card-free, 389 members of population `eb3ba199` (909 studies, one truth vector verified
element-wise). Reproduce with `kneemri.evaluation.ledger75 --holdouts D:/data/volumetric/mri/models/rsna_knee_abnormality`
and the split-half script recorded in `knee-f8nwd`.

## The bar

`comp/champion.json` pins a 7-member ensemble at board 0.946; solos ship 0.936-0.938. At the `ledger75`
board slope of +1.09, a solo beating the champion needs **+0.0073 ledger75**. Every number below is
priced against that, not against the member it modifies.

## 1. The ladder is saturated

Best solo `dml_b_s2026` 0.9078 ledger75 / 0.8703 midpoint. Members within the best:

| band | meaning | count |
|---|---|---|
| 0.0022 | top-band board residual sd | 3 |
| 0.0053 | paired holdout floor | 14 |
| 0.0062 | seed spread | 15 |
| 0.0100 | — | 29 |

Fifteen distinct recipes are statistically indistinguishable from the best, and the top ten span 0.0046
— inside one seed spread. **The bar is wider than the band holding our fifteen best recipes.** No recipe
variation in this project's history has produced a member more than one seed spread above its own pack.
This is the quantitative form of the `ca31ae94` plateau finding.

## 2. The member ranking is honest — selection does not inflate it

Split-half over the 909 studies, 20 trials, choose on A and score on B:

- `corr(macro on A, macro on B)` across 389 members = **0.989**
- winner's optimism (its A score minus its own B score) = **-0.0025 +/- 0.0083**
- the A-winner's shortfall against B's own best = **+0.0041 +/- 0.0027**

So picking one member by macro is not an in-sample trick, and 0.8703 is a real level. This is the
opposite of the `raptor_solo` defect and worth stating plainly: the ranking can be trusted.

## 3. Per-label complementarity is 84% selection noise

The per-label oracle (best member per label, chosen on the holdout) reads macro 0.8827 against the best
member's 0.8703 — **+0.0124, which clears the bar**. It does not survive being held out. Choosing the
per-label argmax on half A and scoring it on half B:

**+0.0020 +/- 0.0036** (min -0.0047, max +0.0094)

Inside the 0.0053 paired floor, and a quarter of what is needed. The oracle's +0.0124 is 84% an artefact
of making twelve independent choices on one holdout. Selection is honest for ONE choice over a stable
ranking (§2) and dishonest for twelve over noisy per-label rankings; the count of choices is the whole
difference. **Per-label reseating, per-label distillation targets and per-label seat search are all
priced by this number and none of them can carry the bar.**

## 4. Two mechanisms refuted on the way

**The geometry hypothesis.** The three weakest labels — Lateral OA 0.8226, MCL 0.8233, Fracture 0.8241 —
sit ~0.05 below the three strongest and are peripheral or cortical, so the band window was a candidate.
Across 315 members carrying a band stamp, the peripheral-minus-central gap is flat: -0.0801 at band
0.2-0.8, -0.0827 at 0.1-0.9, -0.0884 at 0.0-1.0, `corr(width, delta) = -0.081`. The narrowest window has
the SMALLEST gap — the sign is opposite to the hypothesis. Geometry does not make those labels hard.

**`knee-wweo`'s premise is arithmetically wrong.** The bead motivates a remembered-margin objective by
"Fracture skipped ~half the steps at batch 12". Fracture's prevalence on the holdout is 0.245, so the
chance a batch of 12 lacks a positive is `0.755^12 = 0.034`. The objective drops it from **3.4% of
steps, not half**, and no label in the set is rare (lowest prevalence 0.229). The lever's ceiling is
3.4% of steps on 1 label of 12. The code ships and is tested; the motivation does not survive contact
with the prevalence table.

## Consequence

The best the existing pool can honestly yield is +0.0041, and that is the distance from the member we
would pick to the half-best — not a gain that survives shipping. Against a +0.0073 bar, **there is no
offline lever left in the member pool.** Only new training can pay: the trunk axis (`knee-4zw4n`, 7.7 h,
P 0.08, expected +0.0012 — the worst ratio on the board) or the data / cross-site axis, which is the
only family holding a measured non-knob precedent (slot dropout 0.40 buys +0.0132 on `foreign_readout`
against its control, saturated at 0.40 per `knee-yotj8`, so banked but proof the class is live).

A corollary for slot pacing: with nothing in the pool able to beat the champion, a submission mark
spends on an INSTRUMENT, labelled as one, and never on a reseating.

## 5. CORRECTION: the bar above was wrong by 8x, and the board says the same thing anyway

Sections 1-4 priced everything against "+0.0073, because solos ship 0.936-0.938 and the champion is
0.946". **That was a recency sample of the last four submissions, not the record.** Read from
`comp/board.jsonl` (107 priceable rows, 22 of them solo):

**The best solo this project owns is `extffsolo_v1` at board 0.945 — ONE TICK under the champion**
(`tripleseat_v2`, 0.946, submission 56232529). The solo bar is +0.001, not +0.008, and the ensemble
premium over the best solo is +0.001 at a board resolution of 0.001. `knee-4zw4n`'s "<=0.002 premium"
was right; the "+0.008" framing was mine and is retracted.

### Why the top solos have no offline number

`entry_offline --ruler ledger75 --write`: **0 fillable, 48 declined.** Every top solo is an `_ff` —
refit at `--fold -1`, so it has seen every holdout study and can carry no honest offline reading BY
CONSTRUCTION. That is not a defect; it is what a final fit is. The one honest solo row is `l75solo_v1`,
fold-0 `labelwise_depth24_drop40_long`, offline75 0.9053 -> board 0.941, which landed on its own point
prediction.

### The whole-corpus refit term is real, spent, and too noisy to steer by

Measured on three matched arrangements of one base against its own refit:

| base | fold-0 board | refit board | term |
|---|---|---|---|
| labelwise_ext_drop40_long | 0.937 | 0.945 | **+0.008** |
| labelwise_depth24_drop40_long | 0.941 | 0.944 | **+0.003** |
| headlr_ship | 0.937 | 0.938 | **+0.001** |

Mean ~+0.004 with a spread as wide as the mean. **All five of our best bases have now been refit and
shipped** — extffsolo_v1 0.945, ffsolo_v1 0.944, dmlbff_v1 0.943, headlrff_v1 0.938, dmlbfflong36_v1
0.938. The axis is closed by exhaustion, not by a verdict.

### The board confirms §2-§3 independently

Offline does NOT order the refit outcomes: offline75 0.9045 refit to 0.945, while the HIGHER 0.9059
refit to 0.938. The ~0.004 refit scatter swamps the ~0.0015 offline spread separating our top members.
This is exactly §3's result arriving from the other direction — offline differences among the top of a
saturated pool are real (§2, ranking corr 0.989) and too small to survive a transfer whose own noise is
larger.

### What this closes

**No submission built from the current member pool can be expected to beat 0.946.** Every base is
refit and shipped, the pool's honest internal spread is +0.0041, and the refit term is exhausted. A
submission mark therefore buys an INSTRUMENT, labelled as one, and the only route to 0.947+ is a member
genuinely better than anything on disk — which must clear ~+0.004 offline to be legible above the
transfer scatter at all. That is the same 1.5%-class bar as §1, reached by a different road.
