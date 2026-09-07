# The offline→board slope collapses at the top of our own range, and it is tight while it does

2026-09-07, 04:05Z. Fitted from `comp/board.jsonl` — 44 rows carrying BOTH a `board` score and an
`offline` number with `scope != excluded`. No card, no slot.

## The number

`board_transfer` is quoted in this repo as **+1.72** board points per holdout point (and, in older prose,
as +1.82 and +1.55). It is **not a module** — nothing computes it; it lives in kernel docstrings, which is
how the wrong value has been quoted from prose twice before. So it was refitted here, over nested cuts on
the offline axis:

    offline >= 0.000   n= 44   slope +1.336   resid sd 0.0085
    offline >= 0.860   n= 32   slope +1.255   resid sd 0.0025
    offline >= 0.865   n= 25   slope +1.130   resid sd 0.0016
    offline >= 0.868   n= 18   slope +0.911   resid sd 0.0013
    offline >= 0.870   n=  9   slope +0.648   resid sd 0.0015

The slope falls monotonically across every cut, and the residual scatter falls with it by a factor of six.
That second column is what makes this more than a restriction-of-range story: if the flattening were the
narrow high-end range biting an unstable estimate, the residuals would widen, not tighten to 0.0013.

This is not new to the project so much as newly measured. `kneemri-greedyseat`'s docstring already states a
"top-band compression of **+0.89 to +0.97** board points per holdout point" and `kneemri-ceiling5` records
a `[+0.71, +0.93]` band it calls DIRECTION ONLY. Those agree with the n=18 cut here. What was wrong was
not the knowledge; it was which constant got USED. Every recent price reached for **+1.72**, the
whole-range slope, and applied it to a delta that lands at the very top of the range.

## What it re-prices

`knee-ygi2k` (`retrain_six_distill`, 8.5 h of card) was priced:

    per-seat offline +0.0107 x calibsix realisation 0.31 = +0.0033 offline
    +0.0033 x board_transfer 1.72 = +0.0057 board -> 0.944 + 0.006 = 0.950

At the slope that actually governs the band the entry lands in:

    +0.0033 x 0.911  = +0.0030 -> 0.947
    +0.0033 x 0.648  = +0.0021 -> 0.946      (the n=9 cut, our exact operating point)
    +0.0033 x 1.130  = +0.0037 -> 0.948      (the most generous defensible cut)

So the arm's honest prediction is **0.946 to 0.948**, not 0.950. **P(>=0.953) is ~0**, not 0.15. The
pre-registered bands (>=0.946 the seats carry / 0.944-0.945 flat / <=0.943 stop) do not move — they were
written before the number and they stay written — but the probability mass sitting in each of them does,
and the top band is now the *predicted* outcome rather than a stretch.

## The larger claim, and it is the one that matters

At the top of our offline range, **a point of holdout AUC buys about two thirds of a board point, and the
relationship is tight enough that we can say so.** We are 0.010 behind the leader. Closing it by offline
AUC alone needs roughly +0.015 of holdout — more than the entire distance from our worst shipped member to
our best.

That is a statement about the RULER, not about the methods. Our holdout is drawn from the same corpus we
train on; the competition test set is not. A treatment that improves in-corpus separability is being
measured by an instrument that has nearly stopped responding, which means the remaining board points are
in whatever the holdout cannot see: distribution shift, acquisition variation, and the label noise
structure of a different reader pool. Mechanisms that should be preferred from here are the ones whose
gain is invisible to our own holdout by construction — foreign-corpus pretraining, shift augmentation,
robustness objectives — and they must be priced against the BOARD directly, because the offline column
cannot price them.

## What would refute this

A single entry landing well above the fit's prediction from an offline gain at the top of the range. The
champion is already one candidate: `raptorweight_v1` reads 0.944, above every offline-priced row in the
file (the highest is `honest12` at 0.937), and it carries no usable offline number, so it sits outside the
fit entirely. If the champion's extra 0.007 came from an offline improvement this fit would have missed
it — but it did not: it came from weighting, which is exactly a board-only mechanism, and so it is
evidence FOR the claim rather than against it.

The honest caveat: these 44 rows are not an experiment. They differ in member count, selection style and
kit, and the high-offline rows are largely reseatings of one pool, which attenuates a slope by itself.
The tightening residual is the argument that attenuation is not the whole story; it is not proof.
