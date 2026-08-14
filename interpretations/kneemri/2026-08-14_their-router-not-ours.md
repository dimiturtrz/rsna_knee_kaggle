# Their router, not ours: finding the one difference in eleven

*2026-08-14, afternoon and evening. Every number here is CPU arithmetic. No GPU, no submission.*

For three days the graft's 0.011 leaderboard gap and its 0.025 out-of-fold residual had one leading
explanation and no test. This is how it was closed, re-opened by its own evidence, and then answered —
and the sequence matters more than the answer.

## The instrument had to be rebuilt before anything could be asked

`scratchpad/oof_oracle.py` compared our pipeline against a published member's own recorded predictions.
It was never committed and is gone, along with the weights it ran on. Two beads had argued for three
days from a number nobody could recompute.

Rebuilding it took one morning and immediately answered two of those arguments from facts sitting inside
checkpoints nobody had opened: the published members declare `crop_mm 130.0`, `band 0.20,0.80`, `img
336`, `slices 12`, `group 3`, `pool cls_mean`, the identical slot order, and byte-for-byte the same
`rules` dict `member_record.py` hardcodes.

**The lesson is not "commit your scripts".** It is that an instrument's absence is invisible: nothing
failed, no test went red, and the project kept reasoning confidently about a residual for three days
using a number that could not be reproduced.

## The refutation that was too broad

The tie-break hypothesis was: their stable `sort_values` keeps the first candidate in FRAME order and
our `max()` keeps the first in LIST order, so contested slots get different series.

One CSV read killed the ordering half. `Study.from_directory` probes `sorted(directory.iterdir())` and
the directory names ARE the series UIDs; `train_series.csv`'s frame order is that same sorted-UID order
for **4,407 of 4,407 studies**. Same order in, same first-maximal out.

That was written up as *"the tie-break is refuted"*, and it went into `SUBMISSIONS.md` in those words.

**The step from "same order" to "same choice" assumed their rule was first-maximal** — an inference
recorded in that same document, never verified, because we hold their checkpoints and not their router.
The cheap measurement answered the question it could answer and it was read as answering a larger one.

## Its own covariates re-opened it

The rebuilt oracle reports per-study residuals against free covariates, precisely so a defect that
concentrates says so. Over 24 studies:

```
corr(residual, contested_slots) = +0.664     ← strongest of four
corr(residual, series_count)    = +0.647
corr(residual, unrouted_series) = +0.471
corr(residual, empty_slots)     = -0.269
```

Contested slots are only 0.531 collinear with series count and survive controlling for it (**partial
+0.496, p ≈ 0.009**). The population whose divergence had been declared impossible was the one that best
predicted where the two pipelines disagree.

## The answer, by experiment rather than inspection

Decode the same studies under each rule, run a published member on both, compare against its own
recorded predictions:

| | 16 studies WITH a contested slot | 40 studies, unselected |
|---|---|---|
| `tie=first` (ours) | 0.1277 / 0.0236 | 0.1205 / 0.0094 |
| `tie=last` | **0.0252 / 0.0061** | **0.0671 / 0.0062** |

Replicated on **4 of 4 published members across 3 seeds**. And the confirming detail: once the rule is
corrected the routing signature does not shrink, it **disappears** — `corr(residual, series_count)` falls
from **+0.647 to −0.070**. The series-count correlation *was* the tie-break and nothing else.

## Ten axes say our decode is already theirs

Held `tie=last` fixed and swept everything else. Nothing beat the current pipeline:

```
antialias on the resize        0.067122 / 0.006468
percentiles (0, 100)           0.227633 / 0.038068
percentiles (0.5, 99.5)        0.089284 / 0.013494
per-SLICE percentiles          0.118455 / 0.011000
linspace without the dedupe    0.067122 / 0.006152   (bit-identical)
VOI LUT / DICOM window         0.504857 / 0.019107
crop size by floor()           0.068437 / 0.007994
overlapping windows            0.089594 / 0.016313
slices reversed                0.104531 / 0.015812
exact (odd) crop width         0.060341 / 0.006899   (max better, mean worse — no)
```

Read as evidence rather than as failure, that is the more reassuring half: our percentile values and
scope, our raw `RescaleSlope`/`Intercept` decode with no DICOM window, our crop rounding, our band
sampling and its documented dedupe, our resize filter, our slice order (their `rules.order = 'normal'`
is what we already do) and the disjoint-window policy are each the best of the alternatives tried.
Several were assumptions recorded in `SUBMISSIONS.md` that nobody had ever tested.

The arithmetic explanation is out too: a cross-device float difference is about **5e-6**, which is what
the *fingerprint* oracle measures between this machine and a T4. The residual is three orders larger.

## One contributor, not the cause — and where it stops

The tie rule is worth about a third of the mean residual. On the unselected sample **32 of 40 studies
still disagree above 0.01** against the 1e-6 a faithful pipeline reaches.

The eleventh axis was ambiguous (max better, mean worse), and the hunt stopped there on purpose: each
further axis costs a full sweep for a diminishing prior, and the remainder is probably not one thing —
sub-pixel crop placement, uint8 quantisation and float accumulation order can each contribute a little,
none separately findable at n=40. What would actually move it is hundreds of studies rather than forty,
and the residual's per-target **sign** structure rather than its magnitude, which would separate a head
or pooling difference from a pixel one.

## What was built, and why it is a strategy rather than a fix

Our members were **fitted** under first-maximal. A global switch would fix the graft and break
everything of ours. So the tie became the third per-member reading axis of the day, after `crop_mm` and
`centre`, each for the same reason: **a value recorded in a manifest that no consumer reads is a
guarantee that is not one.**

`Ensemble.reads` groups by it and `decoded_scheme` REFUSES a mixed group rather than resolving it, so a
fused entry holding both routers is two decodes instead of one silently wrong one. `MemberPackage` takes
the rule as an argument and never a default, because it is our inference about someone else's weights,
and burying it in a kernel would hide the claim from the manifest it contradicts.

## The pattern worth carrying

Three times today the cheap measurement beat the careful one, and twice it also over-claimed:

- a 4,407-study contested-slot census left the tie-break standing; one CSV read appeared to kill it; the
  oracle's own covariates showed the CSV read had answered a narrower question than the words used.
- the ledger's headline target (*"+0.0173 from MCL and Fracture"*) died to a corpus-scale re-read.
- a base-rate argument about the ruler was refuted by the direct test the same day it was filed.

The rule that would have prevented all three is the same one: **state which question the measurement can
answer, and check that it is the question being asked.**
