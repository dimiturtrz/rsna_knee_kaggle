# Band TTA loses to the single band it averages, solo and in the seat — closed on records already on disk

2026-09-06. Subject: multi-band test-time augmentation for the foreign DINOv3 seat. Instrument:
`kneemri/evaluation/band_tta.py`. Card spent: **zero** — the three band records this reads were written by
`foreign_decode.sh`'s coarse sweep days ago, and the seat pricing reuses `price_served_base8.sh`.

## The idea, and why it looked good

The foreign seat serves a slice band; the coarse sweep priced 0.12-0.88, 0.15-0.85 and 0.20-0.80 and the
band mattered (0.8812 / 0.8781 / 0.8703 solo). A band is a view, not a fit, so averaging the three is TTA
in the cheapest possible form: three forward passes of one encoder, no training, no new checkpoint. The
prior was that a rank-mean over views buys what an ensemble buys — the views disagree where the study is
ambiguous, and the mean is better than any of them.

## What it reads

| reading | value |
|---|---|
| rank-mean over the three bands, solo | 0.8729 |
| best single band (0.12-0.88), solo | **0.8812** |
| the TTA record as a seat on the served base of 8, at its chosen 6x | +0.0128 |
| the single 0.12-0.88 band as the same seat, at 6x | **+0.0178** |

Both rulers agree and both point the same way: the average is **worse than its own best member**, by 0.0083
solo and by about 0.005 in the seat. The seat gap is the one that decides — it is above the 0.0011 board
resolution and above the +0.005 bar the foreign axis is held to — so this is a closure, not a flat.

## Why the mean loses (the mechanism, stated so a later use can re-test it)

TTA averages views of ONE model, and the views here are not equally good: 0.20-0.80 is 0.011 below
0.12-0.88 solo. A rank-mean weights them equally, so two-thirds of the average is drawn from bands that
are individually worse, and the disagreement it buys is not orthogonal error — it is the same encoder
seeing fewer slices. That is the difference between TTA and an ensemble, and it is why the ensemble
argument ("even a weaker member helps, it is about diversity") does not carry across: a weaker VIEW of the
same weights adds no independent error to cancel.

What this does NOT close: a WEIGHTED combination that keeps 0.12-0.88 dominant, and the fine sweep
(1090/1189/1387/1486) that is still looking for a better single band. The reading is about the equal mean
of unequal views, on this encoder.

## Consequence

The served band stays a single band, chosen by the sweep. No slot, no card, and the fine sweep keeps its
own pre-registration (a clearing band prices against 0.8862 fold-matched).
