# The gold ruler raises everyone and can order no one

2026-08-30. `knee-9198` asked one question with two possible answers, and the answer decides whether
member selection — the greedy search, the pool ceiling, every roster call this project makes — has been
computed on the wrong ruler all along.

## The question, pre-registered

The board scores against **image-read radiologist consensus** (overview topic 733343). Every other number
here is measured against a report-derived **proxy**. `holdannot_s2026` reads 0.8487 proxy / 0.8932
annotation — same weights, same studies, **+0.045 from the ruler alone**. Because the metric is per-label
rank AUC and the two label sets rank studies differently, the proxy is not neutral toward label-side arms;
it is biased against them.

So: does the gold ruler only raise the LEVEL, or does it REORDER? Written before the numbers were read:

- **Level only** — the proxy is a shifted but faithful ruler, and selection stays where it is.
- **Ordering changes** — every selection decision was made on a ruler that disagrees with the board's,
  and member selection must move.
- **The readable claim is the ordering pattern, never a pairwise delta.** n=58, ~696 cells, per-label
  standard error 0.04–0.09 (`label_power`). Six members spanning 0.014 of proxy will not separate
  pairwise on 58 studies.

## The instrument was broken, and that is its own entry

The first six-member read was crossed: `AnnotatedReadings.readings` returned arrays in cache order and
`main` re-paired them against argv with `zip(..., strict=True)`, which guards length and not identity.
Five of six rows were reported under another member's name. It produced a confident ordering finding
(Spearman 0.43) and a contradiction against a correctly-closed P1, both of which were false. See
`knee-bk99` and commit `55673b6`. Everything below is post-fix, verified by equality against six
single-member reads where crossing is impossible.

## The numbers

| member | proxy | rank | gold | rank |
| --- | --- | --- | --- | --- |
| `calib_span_hold_s2026` | 0.8564 | 4 | **0.9152** | **1** |
| `calib_dense_hold_s2026` | **0.8634** | **1** | 0.9117 | 2 |
| `band_span_hold_s2026` | 0.8493 | 6 | 0.9023 | 3 |
| `band_dense_hold_s2026` | 0.8573 | 3 | 0.8956 | 4 |
| `blendhold_s2026` | 0.8576 | 2 | 0.8952 | 5 |
| `holdannot_s2026` | 0.8506 | 5 | 0.8932 | 6 |

**Spearman 0.14.** Read bare, that is the pre-registered "ordering changes" outcome, and it is the number
I was about to write up.

## Then the intervals arrived, and the finding did not survive them

`annotated_readings` printed bare means until today. It now prints the paired difference from a baseline
**named on the command line** — `calib_dense_hold_s2026`, the proxy's top member, a choice that pre-dates
this read and so is not the winner of the table it sorts:

```
paired against calib_dense_hold_s2026 (95% interval over studies):
  band_dense_hold_s2026        -0.0161 [-0.0355, +0.0019]   no pairwise claim
  band_span_hold_s2026         -0.0094 [-0.0277, +0.0053]   no pairwise claim
  calib_span_hold_s2026        +0.0035 [-0.0121, +0.0187]   no pairwise claim
  blendhold_s2026              -0.0165 [-0.0362, +0.0029]   no pairwise claim
  holdannot_s2026              -0.0185 [-0.0398, -0.0019]   clears
```

**The whole apparent reordering is one pair moving +0.0035 inside a band twelve times its width.** The
entire gold spread, best to worst, is 0.022 — narrower than the half-width of every interval on the page.
The single pair that clears runs the *same* direction the proxy already had (proxy #1 over proxy #5), so
it is not evidence of disagreement either.

**Spearman 0.14 is what ranking noise looks like.** Six members that cannot be told apart pairwise will
produce a low rank correlation between any two rulers, by construction. The statistic was never
measuring the ruler; it was measuring the denominator.

## What this settles

**The level shift stands. The reordering does not exist as a readable fact.** +0.045 is twice the entire
spread of the members being ordered, so the two claims are not the same size and never were.

**Member selection cannot move to the gold ruler, and no retrain fixes it.** The standing reading of
`knee-9198`'s remaining half was that selection *should* run on gold and today *cannot*, because only
`hold_annotations=True` checkpoints are readable there — a plumbing limit, priced at ~6 × 40 min of card
to retrain the shipping roster. That is the wrong diagnosis. The limit is the **denominator**: 58 studies
cannot resolve the 0.002–0.01 differences that selection turns on, whoever is readable on them. Retraining
the roster under `--hold-annotations` would have bought a ruler with the same blindness and cost four
hours of card to find that out.

**What would change it is more annotated studies, and nothing else.**

## Limits

- `calib_*` members carry `Supervision.MEASURED_AGREEMENT`, a constant fitted on these same 696 cells,
  and `--hold-annotations` drops the cells without refitting it — a structural edge on this ruler. It
  does not matter to the conclusion here, since neither `calib_*` member separates from the field anyway.
- `combined` (six-member rank average) reads 0.9155, **+0.0003 over the best single**. Also unreadable at
  this n, and stated only so it is not later quoted as an ensemble gain.
- The intervals are paired bootstraps over studies, seeded. They bound comparisons named in advance, not
  a search over the six.
