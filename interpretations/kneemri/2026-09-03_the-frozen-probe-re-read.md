# The frozen probe, re-read: two findings and one cache that was never two caches

2026-09-03. `kneemri.evaluation.feature_probe --choose-penalty`, fold-0 holdout, n=909, closed-form
ridge over memory-mapped caches — no card. Beads `knee-x4yx` (the penalty), `knee-2rmdc` (the readout),
`knee-n647` (the duplicate cache).

`knee-x4yx` said the probe compared trunks at a SHARED ridge penalty and that this had mispriced at
least two declines. The code fix was already in the tree — `FeatureProbe.chosen`, a five-decade grid,
selection on a nested split inside the training positions. What had never been done was the reading.
This is that reading, plus a second one the same instrument makes for free.

## One: the rank reversal is real, and it survives every readout

Each trunk at the penalty it is entitled to, `SliceMean` readout:

| trunk | penalty | mean AUC |
|---|---:|---:|
| ortholarge@focal | 1.0 | **0.8249** |
| radnet121@focal | 0.003 | 0.8197 |
| ortholarge | 0.1 | 0.8170 |
| dinov2-large@focal | 30 | 0.8048 |
| v3large@focal | 3.0 | 0.8046 |
| dinov2-large | 10 | 0.7996 |
| vjepa2-large | 10 | 0.7985 |
| v3large | 0.3 | 0.7968 |
| dinov2-small | 100 | 0.7896 |
| orthodiff | 0.3 | 0.7435 |

Under a shared penalty both `ortholarge` and `radnet121` lost to the DINOv2/v3 rows. At the penalty
each trunk wants, both lead them. The chosen penalties span 0.003 to 100 — five decades — which is
precisely the defect the bead described: a shared penalty is not a neutral choice, it is a choice that
silently favours whichever trunk happens to want it.

**The controlled pair.** `ortholarge` against `v3large` is the same architecture at the same read size,
differing only in OrthoFoundation's continuation on 1.25M knee slices: **+0.0203** at the focal cache,
**+0.0202** at the base cache, **+0.0165 / +0.0167** again under the `max` and `moments` readouts. An
in-domain SSL continuation gain, measured four times on one controlled contrast.

**The caveat that binds.** Every penalty peak was selected on the same 909-study holdout, so the LEVELS
are inflated. The ORDERING is the finding, not the numbers.

## Two: the slice readout is the unmeasured half, and the roster serves on the worst of the three

`slice_readout.py` calls itself "the unmeasured half of every trunk number this project has quoted."
It was right. Same instrument, three readouts:

| trunk | max | mean | moments | moments − max |
|---|---:|---:|---:|---:|
| dinov2-large@focal | 0.7936 | 0.8048 | 0.8082 | +0.0146 |
| v3large@focal | 0.7926 | 0.8046 | 0.8042 | +0.0116 |
| ortholarge@focal | 0.8146 | 0.8249 | 0.8279 | +0.0133 |
| radnet121@focal | 0.8061 | 0.8197 | 0.8240 | +0.0179 |
| ortholarge | 0.8038 | 0.8170 | 0.8203 | +0.0165 |
| v3large | 0.7816 | 0.7968 | 0.7983 | +0.0167 |
| dinov2-large | 0.7860 | 0.7996 | 0.8021 | +0.0161 |
| dinov2-small | 0.7792 | 0.7896 | 0.7921 | +0.0129 |
| vjepa2-large | 0.7985 | 0.7985 | 0.7985 | 0.0000 |

Eight of nine carry the ordering `moments > mean > max`. `max` is what every member on the roster is
fitted and served with. `moments` is the mean along the stack concatenated with its standard deviation
— twice the width, no learned parameters — and it is not in `SLICE_POOLINGS` at all, so the readout
that wins here cannot currently be trained.

The V-JEPA2 row is not an anomaly and was checked rather than explained: its cache is
`(4407, 6, 1, 1024)`, a slice axis of length one, because it consumes the stack jointly. Three readouts
over a single slice are the same reduction. It is the one trunk this axis cannot be tested on.

**Why this is a new axis rather than the closed one.** The aggregation chapter measured `attend` at
+0.0002 over `max` and `labelwise` at the same null. Those are learned pools over the same FIRST
moment. `moments` adds a statistic the first moment discards — how a channel varies along the stack,
which is the shape a lesion occupying a few slices has and a uniform structure does not.

**What would make me disbelieve it.** A ridge over max-reduced frozen features is not a learned head
over a max pool; the head can in principle learn what the spread carries, in which case the frozen gap
closes to nothing at train time. Note also that `dinov2-small` — the trunk the whole roster is built
from — carries the smallest testable gap, so this is an upper bound on what our own members would see.

## Three: two rows of that table measured nothing

`features_*_focal_max.npy` and `features_*_focal_attend.npy` are **byte-identical** — verified equal on
`ortholarge`, `v3large` and `radnet121`. `knee-n647` already documents the cause: `--slice-read` named
a readout in the cache FILENAME but never reached the encoder. Every `@focal-attend` row is a duplicate
of its `@focal-max` row, and reads as two instruments agreeing when there is only one. The `TRUNKS`
table in `feature_probe.py` still lists both aliases, which is what invites the misread.

## What this re-opens, and what it does not

`ortholarge` is MIT, so it clears the redistribution test that closed RadImageNet, and it is now the
best-probing trunk the project owns — while SERVING worse than every DINOv2 seat (`ortho_dense` 0.8452,
`ortho_pool_read` 0.8464, `ortho_lpft_head` 0.7537, against ~0.855). Probes best, serves worst is a
RECIPE question, not an architecture one, and `ortho_fresh_lowlr` is the control that makes the LP-FT
triple readable (`knee-6hfi`).

`radnet121` probing at 0.8197 stays unshippable regardless. The redistribution test does not care how
well a trunk reads.

Neither finding is a score. Both are card-blocked, and the card is the owner's.

## A note on the public-baseline audit

`research/2026-09-03_public_rsna_knee_baseline_audit.md` recommends keeping DINOv2-small and not
reaching for a larger DINOv2, citing a Kaggle Models tab where DINOv2-small reads 0.906 and
DINOv2-large reads 0.500. That table is a maximum over whatever strangers uploaded, not a controlled
comparison, and 0.500 is a collapsed head rather than a capacity result — the doc says so itself. On
our own controlled probe `dinov2-large` beats `dinov2-small` by 0.0152 at mean and 0.0100 at moments.

The recommendation still happens to match our serving experience, for a different reason: our larger
served trunks lose, which is a recipe failure we have not yet diagnosed rather than evidence that the
features are worse. The advice is worth following and the evidence offered for it is not the reason.
