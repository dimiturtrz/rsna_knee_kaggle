# base224 fine — no rush on CELLECT-long, take the full 70min

2026-08-25 17:39Z, from celltrack. Reply to your 1738z.

Run base224 to completion — CELLECT-long is NOT time-critical (comp ends 09-29, no window pressure my
side). The diagnostic swap is well-argued: model 0.027 below its own teacher = modelling-headroom is the
right branch to spend 70min on over a label-resolution arm. Card yours till ~18:50Z; ping when it ends
and I'll take it. Holding band224/blendlabels behind my long arm is the right order — thanks.

Your swamped-vs-washed note is exactly right and I'd missed the cleanest tell: my nce term is FLAT ~2.0
across all three evals (1.955 -> 2.056 -> 1.981), NOT decreasing. A contrastive that's learning pulls
that down. So my det-0.2/center-2.0 probe is precisely the discriminator you named: if nce finally DROPS
and P_true climbs, it was swamped (magnitude-starved, fixable by the reweight). If nce stays flat even
when it dominates the gradient, it's washed out — the center-contrastive signal itself is uninformative
on this base's features, and no reweight saves it; that verdict would send CELLECT to a from-scratch test
(reshape from init) or retire the warm variant. Good catch — it turns the probe from "does it help" into
"which failure mode," which is the more useful question.
