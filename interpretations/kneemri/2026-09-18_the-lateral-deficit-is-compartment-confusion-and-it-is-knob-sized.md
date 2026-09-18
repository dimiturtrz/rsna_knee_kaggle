# The lateral deficit is compartment confusion, it is real on three members, and it is worth +0.002

**What was measured.** Per-label ROC AUC on the fold-0 909-study holdout (truth cut 0.75), computed
**conditionally** on whether the PARTNER compartment is diseased. No card, no submission slot — every
number came off holdout `.npz` already on disk. 2026-09-18.

The question: `2026-09-18_the-column-profile-is-architecture-invariant.md` closed six routes to the weak
columns and left one open — a genuinely different INPUT rather than a different encoder over the same
input. Three of the four weak columns are side-defined (Lateral OA 0.8755, Lateral Meniscus 0.8949, MCL
0.8950), so the input lever most worth pricing is spatial. This prices it.

## The lateral columns are not hard. They are hard only next to a diseased medial compartment.

Splitting each lateral column's studies by the state of its medial partner:

| member | lateral column | medial-NEGATIVE studies | medial-POSITIVE studies |
| --- | --- | ---: | ---: |
| `dml_b_s2026` | Lateral Meniscus | **0.9212** | **0.8392** |
| `labelwise_dense_drop40_long36` | Lateral Meniscus | **0.9247** | **0.8256** |
| `band_dense_s2026` | Lateral Meniscus | 0.8922 | 0.7863 |
| `dml_b_s2026` | Lateral OA | 0.8707 | **0.7890** |
| `labelwise_dense_drop40_long36` | Lateral OA | 0.8639 | **0.7948** |

On a knee whose medial compartment is clean, the lateral meniscus reads **0.92** — within 0.02 of the
medial meniscus's own 0.944. The column only collapses when both compartments are in play.

Paired bootstrap over studies, 2000 resamples, confusable minus clean:

| member | Lateral Meniscus | Lateral OA |
| --- | --- | --- |
| `dml_b_s2026` | **−0.0808** [−0.1423, −0.0171] | −0.0819 [−0.1624, −0.0033] |
| `labelwise_dense_drop40_long36` | **−0.0979** [−0.1632, −0.0293] | −0.0694 [−0.1499, +0.0116] |
| `band_dense_s2026` | **−0.1060** [−0.1796, −0.0314] | −0.0274 [−0.1135, +0.0630] |

Lateral Meniscus excludes zero on all three. Lateral OA holds the sign on all three and clears on one —
directional there, confirmed here.

It is symmetric, so it is confusion and not lateral weakness: Medial OA falls 0.9233 on lateral-negative
studies to **0.8582** on lateral-positive ones, and Medial Meniscus 0.9432 to 0.9329.

## The co-occurrence asymmetry IS the medial/lateral gap

**80 of 140 lateral-meniscus positives (57%) sit in the confusable regime, against 80 of 363 medial
positives (22%).** The weaker column is the one whose positives mostly live where the model cannot tell
the sides apart — not the one whose finding is intrinsically harder to see. The same holds for OA: 79 of
128 lateral positives (62%) against 79 of 233 medial (34%).

This explains two results already on the record that the invariance finding could not:

- **`side_dense` regressed −0.0641 with its mechanism confirmed firing** (`experiments/README.md:2750`).
  A loss term cannot recover a distinction that pooling has already discarded from the representation.
- **The profile is architecture-invariant.** Every family tried reads the same pooled volume, so every
  family inherits the same confusion.

## Re-reading `sided_pool` on the ruler that can see it

`sided_pool` (`experiments/README.md:2807`) gave the head an explicit medial/lateral axis and read
**−0.0064 [−0.0215, +0.0085] on Lateral Meniscus** — flat, inside the floor. Its row names an uncleared
suspect: `'centre': 'image'`, `crop_mm 130`, so the head splits the patch grid at the image centre
column, which is the anatomical midline only if the crop is anatomically centred.

The marginal column is the wrong ruler for that arm: 43% of lateral positives are already at 0.92 and
have nothing to gain, so averaging them in dilutes the targeted effect by more than half. Against
`band_dense_s2026`:

| | marginal Δ | **confusable Δ** | clean Δ |
| --- | ---: | ---: | ---: |
| Lateral Meniscus | −0.0035 | **−0.0213** [−0.0515, +0.0077] | +0.0092 |
| Lateral OA | −0.0162 | **−0.0336** [−0.0683, +0.0029] | −0.0015 |

Neither confusable interval excludes zero (P(Δ<0) = 0.92 and 0.96), so this is **directional, not a
result**. What it is, is the suspect's own prediction coming true on two independent pairs while both
clean regimes sit at zero: a midline drawn at the wrong column hurts specifically and only where
placement matters. That moves `'centre': 'image'` from an untested excuse to a supported explanation.

## And the price, which is why none of this is worth a card

The clean regime is the ceiling — it is what this representation achieves when confusion is absent.

- Lateral Meniscus: **0.8964 marginal → 0.9212 ceiling, +0.0248 on one column = +0.0021 macro** at
  `d/12`.
- Lateral OA: **no headroom at all** — clean 0.8707 against a marginal 0.8745.

**So the entire mechanism, solved perfectly, is worth ≤ +0.002 macro.** That is knob-sized, an order
below the 1.5% bar, and it is the answer to the spatial input lever this document set out to price.

**The follow-up `sided_pool` left standing is therefore retired on arithmetic, not on a spent card.** Its
row calls an anatomically registered split "cheap to test"; at ~1.5 h of card for a best case of +0.002,
the ratio is wrong even if the split works exactly as intended. The same disposes of the small-weight
side regulariser `side_dense` deferred to "a head that sees position" — it inherits the same ceiling.

## What this does and does not close

**Closes:** compartment localisation as a route to the weak columns. It is the seventh route closed on
the conserved profile, and the first one closed by pricing its own best case rather than by a null.

**Does not close:** Fracture and MCL, which are weak for other reasons — Fracture on 42 holdout positives
and MCL with no compartment partner to be confused with. Nor the non-spatial input levers (a different
plane or reconstruction, a segmentation prior over a finding rather than a side).

**Honest limitation.** Truth here is our report-mined supervision binarised at 0.75. If the report reader
mis-assigns a side when the text names a tear without one, the confusable regime carries more label noise
than the clean regime and part of this gap would be labels rather than the model. That is not the global
label-noise explanation already refuted at r = +0.081 — it is a conditional version of it, and it is
untested. It would make the mechanism smaller, never larger, so it does not change the verdict.

Related: `interpretations/kneemri/2026-09-18_the-column-profile-is-architecture-invariant.md` (the six
routes this makes seven), `experiments/README.md:2750` (`side_dense`),
`experiments/README.md:2807` (`sided_pool`),
`interpretations/kneemri/2026-08-23_side_resolution_is_not_score.md`.
