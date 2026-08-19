# Density inside the band trades the centre for the edge

`band_dense`, 2026-08-19. The widened band 0.10–0.90 at 20 slices instead of 16 — sixteen slices
inside the old 0.20–0.80 span against twelve for both the control and `band_wide`, with the periphery
unchanged. Everything else the control's recipe: 336 px, whole-stack, max pooling, backbone rate 4e-5,
12 epochs, fold 0, seed 2026.

**0.8578**, peak at epoch 8 of 12. +0.0053 over the control `rate_wholestack` 0.8525 and +0.0025 over
`band_wide` 0.8553. Strongest single member the project has produced.

## The prediction, scored

Written into `experiments/arms/band_dense.sh` before the arm started:

> If dilution is the mechanism: Medial OA recovers toward zero against `rate_wholestack` while
> MCL/ACL/Fracture/Contusion keep their `band_wide` gains, and the mean lands at or above 0.8553.

**Partial confirmation — the OA clause hit, the collateral clause missed.**

| label | vs control | vs `band_wide` | | label | vs control | vs `band_wide` |
|---|---|---|---|---|---|---|
| Synovitis | **+0.0218\*** | +0.0135 | | Medial Meniscus | +0.0064 | +0.0057 |
| PF OA | **+0.0192\*** | **+0.0194\*** | | Effusion | +0.0053 | +0.0039 |
| Medial OA | +0.0034 | **+0.0122\*** | | Lateral Meniscus | +0.0004 | −0.0007 |
| ACL | +0.0034 | −0.0037 | | Baker's | −0.0011 | +0.0016 |
| MCL | +0.0022 | −0.0111 | | Lateral OA | −0.0022 | −0.0009 |
| Contusion | +0.0020 | −0.0052 | | Fracture | +0.0022 | −0.0052 |

909 shared holdout studies; `*` marks a row whose interval excludes zero. **These are the first
per-label rows in this family that resolve on their own** — every row of the `band_wide` table
straddled zero and had to be read as a pattern.

The OA half is confirmed and then some. Medial OA goes −0.0088 → +0.0034 against the control, and
+0.0122 against `band_wide` with the interval clearing zero; PF OA goes −0.0002 → +0.0192, resolving
against both. Synovitis +0.0218 resolves too. Restoring central density recovers exactly the finding
class `band_wide` lost, and recovers it in the measurement that isolates density — the one against
`band_wide`, where the periphery is held fixed.

The collateral half does not survive. MCL falls +0.0133 → +0.0022 and ACL +0.0071 → +0.0034 against
the control; against `band_wide` directly they are −0.0111 and −0.0037, with Contusion and Fracture
−0.0052 each. The peripheral slices are still there — the same four, at the same positions — and the
signal they bought is largely gone.

## What that means

Dilution is confirmed, and it is **symmetric**. Max pooling gives each slice's evidence a share of the
pooled representation, and the share is what moves the label. `band_wide` added peripheral slices and
bought the collaterals at the centre's expense; `band_dense` adds central slices back and buys the
centre at the periphery's expense. Neither arm added or removed anatomy from the second half of that
trade — only its share.

That reframes what `band_coverage` and `band_wide` were measuring. The question was never "is the
outer band empty" (it is not) but "how is a finite pooled budget spent across the span", and both arms
answer the same question from opposite ends.

**It also puts a caveat on knee-ezz.** That arm priced density inside the band at −0.0001 on the mean
and the mean is what was recorded. This one says a flat mean can hide a per-label trade of this size:
+0.0122 and +0.0194 in one direction, −0.0111 in the other, summing to nearly nothing. knee-ezz is not
wrong about the mean; it is silent about the composition, and it was read as though it were not.

## What it is worth as a member

`band_wide` + `band_dense` alone reads **0.8633** over the 909 shared studies, +0.0055 over the better
of the two. The two are complementary rather than one superseding the other, which is what the
per-label table predicts: each holds what the other gave up. Both belong in the roster.

Read those as signs, not scores — a training-time record is the trainer's reading of the holdout, and
the holdout prices diversity ~2.4× too cheap.

## What it does not say

- **Not "20 slices everywhere."** One arm, one fold, one seed. The mean beats the noise floor
  (+0.0053 on 0.0062 — barely, and only against the control) but the per-label rows carry the finding.
- **Not that the periphery should be dropped again.** The collaterals are worse here than in
  `band_wide` but no worse than in the control; the peripheral slices are paying for themselves, just
  less than they did at a smaller central share.
- **Not a slice-count result.** Coverage in millimetres is identical between this arm and `band_wide`.
  What changed is sampling density inside a fixed span, and the effect is on the pooling, not on what
  the series contains.
- **Not a licence to re-open the refuted knee-jk3 items.** Resolution, slot count and cache density on
  the mean stay where they were.

## SUPERSEDED ON THE MECHANISM, 2026-08-19, by the fourth corner and then the third

The title claim — that density inside a band TRADES the centre for the edge — was written from two
corners of a square and it does not survive the other two.

`band_density` (0.20–0.80 at 20 slices) and `band_span` (0.10–0.90 at 12 slices) complete it:

| | 12 slices | 20 slices | density effect |
|---|---|---|---|
| **0.20–0.80** | `lr4e-5_ws` 0.8525 | `band_density` 0.8540 | +0.0014 |
| **0.10–0.90** | `band_span` 0.8565 | `band_dense` 0.8578 | +0.0013 |
| **span effect** | +0.0039 | +0.0038 | interaction ≈0.0001 |

Each main effect reproduces at both levels of the other, so **on the mean the two levers are ADDITIVE
and independent** — there is no fixed representational budget being re-spent, which is what "trades"
asserted. `band_dense`'s +0.0053 is simply span +0.0039 plus density +0.0014.

**What survives is the signature, and it is the better half of the finding.** The trade this document
saw is real inside the LABELS, just not on the mean: widening the span buys REGIONAL findings
(+0.0079 against focal +0.0011 — ACL, Contusion, Baker's, Effusion), and adding slices inside a span
buys FOCAL ones (+0.0036 against regional −0.0019 — Synovitis, PF OA). Two levers that move different
label populations and add on the mean are the diversity argument, not the budget argument.

The remaining conclusions above stand: not a slice-count result, and not a licence to re-open the
refuted knee-jk3 items. `band_full` (0.00–1.00 at 12 slices) tests where the span lever stops.
