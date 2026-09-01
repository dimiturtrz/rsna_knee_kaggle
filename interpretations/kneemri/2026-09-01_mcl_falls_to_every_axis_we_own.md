# MCL falls to every axis we own, and the arm that proved it was read on the wrong column — knee-9707

MCL is the cheapest column on our own ruler: 0.8195 against the honest12 ensemble's 0.8750 mean, −0.0555,
the largest single per-label deficit we have. This closes the question of what moves it, by recovering a
result that has been on disk since 2026-08-23 and was never read against the bands it pre-registered.

## The arm ran, and it was judged on a column it told us not to read

`experiments/arms/focus_mcl.sh` still opens `CANCELLED BEFORE IT RAN`, and `experiments/README.md:1073`
repeats it. Both are wrong: `focus_mcl_s2026.metrics.jsonl` holds twelve epochs. The ledger elsewhere
(line 5591) knows it ran and records it as *"the `--focus-label MCL` arm refuted earlier the same day"*
at a solo of **0.5332** — the twelve-label mean.

The arm's own header forbids that reading, in as many words: *"Read the MCL COLUMN, not the mean: eleven
of twelve heads carry no gradient, so the mean is noise by construction."* `--focus-label` zeroes the
other eleven loss weights. A near-chance mean is the arm working as designed, not the arm failing.

## On the column it pre-registered, it is the informative null

| | MCL |
|---|---|
| `band_dense_s2026` (the matched control, e8) | 0.7931 |
| `focus_mcl_s2026` peak (e7) | **0.7861** |
| delta | **−0.0070** |

The pre-registered bands were written before the run. `0.783–0.802` reads:

> **JOINTNESS COSTS MCL NOTHING.** The informative null: it retires the objective as an explanation and
> points the whole MCL deficit at what the INPUT carries.

−0.0070 lands inside that band. The maximum tilt — eleven columns fully masked — moved MCL by less than
the noise floor. This is the null the arm was run to get, and it has been sitting unclaimed for nine days
while its own paperwork said it never ran.

## Where the null pointed, the data already was

The null hands the deficit to the input, and names the form: mm/px on the medial margin, a
crop-and-resolution arm. That arm does not need running either. Checkpoints on disk, MCL peak per run:

| run | crop / span | MCL |
|---|---|---|
| `crop80_tissue_full_s2026` | tightest | 0.7469 |
| `crop80_tissue_s2026` | tight | 0.7480 |
| `crop80_span_s2026` | tight | 0.7604 |
| `band140_s2026` | 140 mm through the same grid | 0.7772 |
| `band_dense_s2026` | (0.1, 0.9) | 0.7931 |
| `band_dense_hold_s2026` | (0.1, 0.9) | 0.8044 |
| **`band_wide_s2026`** | widest | **0.8076** |

**Tighter crop is monotonically worse on MCL, which is the opposite of what more mm/px predicts.** Three
crop80 runs sit 0.033–0.046 below the control; the widest band holds the best MCL this project has ever
recorded. Pushing more span through the same pixel grid (`band140`) is also worse. So the input axis does
not resolve the way the null's suggested follow-up assumed, and the ledger's own sweep already had
resolution flat (336 → 0.7695 over 54 runs, 448 → 0.7711 over 3).

## What is now closed

Every axis this project owns has been measured against MCL and none of them move it:

- **recipe** — 83 recorded runs; crop band spans 0.013, slice count 0.013, resolution 0.002
- **data volume** — the extended slot scheme, +83,297 slices, reads 0.7604 against canonical 0.7931
- **encoder family** — dinov2-small 0.7732 (n=48), base 0.7812, vjepa2l 0.7803, dinov3-small 0.7257
- **objective** — this null, at maximum tilt
- **crop and span** — the table above, in both directions
- **label quality is not the explanation** — MCL's label confidence is 0.789 (HIGH) at prevalence 0.174,
  while Synovitis at confidence 0.189 reads 0.8719

**And the existence proof the whole programme rested on is already met.** `focus_mcl` was priced on
another model reaching 0.8166 on MCL against our best-ever single member of 0.8042. The honest12
ensemble reads **0.8195**. Ensembling clears the bar that the weak-column programme was chartered to
reach, on the axis it was chartered to reach it — so the programme's remaining value was never the score.

## What this does NOT establish

- **The denominators differ and that is not a detail.** 0.8166 is the 4,407-study corpus ruler on twenty
  published members' weights; 0.8195 is our 909-study fold-0 report ruler. They are not the same
  measurement and the comparison is a flag, not an equality. What it does support is that no *single*
  member of ours needs to reach 0.8166 for the ensemble to.
- **"Every axis we own" is the operative clause.** Nothing here says MCL is intrinsically unreachable —
  it says the levers this repo has built cannot reach it. A different modality, a different pretraining
  corpus, or a medial-margin-specific representation are all untested and none is refuted.
- **The −0.0070 is one run against one control.** It sits inside a pre-registered null band, which is
  what makes it readable, but it is single-seed and the band is ±0.01 by construction.
- **The starvation confound applies to a LOSS, not to this null.** A focused run sees one supervisory
  signal where the control sees twelve (`half_data` measured −0.0234 for half the studies), so a *loss*
  would have been ambiguous. A null at maximum tilt is not: the arm was not starved into a tie.
