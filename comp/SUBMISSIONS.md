# Submissions

Kernels are named for the METHOD, not for the attempt: `kneemri-<source>-<variation>`. A new version of
one slug is that same method retrained or re-tuned; a different source or aggregation is a different
slug. The slug is the claim, so a grafted score can never quietly become ours.

Budget: **5 submissions per day**. Kernel *pushes* are unlimited — probe there, submit only when the
question needs the leaderboard to answer it.

## Ledger

| date | kernel | v | method | local | LB | note |
|---|---|---|---|---|---|---|
| 2026-08-10 | `kneemri-probe` | 2 | passthrough `sample_submission` | — | not submitted | environment + corpus probe |
| 2026-08-10 | `kneemri-graft-verify` | 6 | fingerprint check only | 20/20 @ 4.65e-06 | not submitted | proves the architecture port |
| 2026-08-10 | `kneemri-graft` | 1 | **GRAFT** — 20 published CC0 members, our data layer, rank-averaged | their holdout 0.8398 | pending | **their result, not ours** |
| 2026-08-10 | `kneemri-trained` | 1 | **OURS** — 5 folds fitted by `kneemri.train` on report-derived labels | fold 0 @ 6 epochs: 0.7928 | — | our first real number |

## What counts as ours

The graft row is the field's floor measured on our harness. It is **not a result**: the weights are
`pilkwang/rsna-knee-weights` (CC0), and the only thing of ours in it is the data layer and the ensemble.
It exists so that every later number has something to be compared against, and so the submission path
was proven before a model depended on it.

`kneemri-trained` is ours end to end — our slot routing, our ordering, our laterality, our crop, our
head, our fitting. It starts below the graft, which is the honest starting position for a replication
plus one day's work.

## The two oracles

Anything claiming to reproduce the published pipeline has to pass both, because they check different
things and neither implies the other:

- **Fingerprint** (`kneemri-graft-verify`) — each member's output on a seeded synthetic bag. Proves the
  ARCHITECTURE is the function those weights were fitted to. Blind to the pixels: it feeds noise.
- **Out-of-fold agreement** (`scratchpad/oof_oracle.py`) — our pipeline against the member's own stored
  predictions on the studies it held out. Proves the PIXELS: routing, order, laterality, crop,
  percentile, resize. Currently max|diff| 0.025, per-study correlation 0.99, under disjoint windows.

The residual 0.025 is an OPEN QUESTION, not a rounding error — a faithful pipeline would agree to about
1e-6, as the fingerprint does. The likeliest cause is the slot tie-break when two candidate series have
equal slice counts.

## Gotchas paid for in full

- **Submission is notebook-only.** `kaggle competitions submit -f submission.csv` returns 400. The
  working form is `-k <owner>/<kernel> -v <version> -f submission.csv`.
- **Search for the artefact, not the container.** Three kernel runs died on mount-shape assumptions:
  the competition sits under `/kaggle/input/competitions/<slug>`, datasets under
  `/kaggle/input/datasets/<owner>/<name>`, and a `--dir-mode zip` upload is expanded on mount with its
  wrapping directory REMOVED.
- **Probe what you IMPORT, not only what you plan to use.** The runtime probe checked pydicom, timm and
  transformers (all present) and never checked `jaxtyping`, which our own modules import at module top.
- **A published checkpoint is a training artefact**, not a bare state dict: weights sit under `model`
  beside epoch, fold scores, config, out-of-fold predictions and its own fingerprint.
- **`machine_shape: NvidiaTeslaT4` gives two T4s and only four CPUs.** On a decode-bound pipeline the
  CPU count binds first: measured 12 ms per DICOM slice off the mount.
