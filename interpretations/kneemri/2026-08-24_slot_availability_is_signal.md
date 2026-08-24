# Which sequences a study was scanned with is SIGNAL, and the ensemble already uses it

2026-08-24. Card-free, slot-free — pure arithmetic over the stored holdout predictions.

## The hypothesis, and why it looked good

`SlotHead` renormalises over the slots a study actually has, and `submission.decoded_at` returns a
zero mask for a study it cannot decode. So a study's prediction LEVEL depends on which of the six
primary slots it presents. AUC is a POPULATION-wide ordering statistic: if the level shift attached
to a presence pattern were orthogonal to the label, it would inject ordering that carries no
information, and every pair drawn from two different patterns would be scored on an artefact.

The fix would then be free and available at test time with no training: normalise ranks WITHIN each
presence stratum before averaging. AUC is invariant to a per-LABEL monotone transform but NOT to a
per-STRATUM one, so the transform is expressible and its effect is measurable.

`experiments/README.md` and every note under `interpretations/kneemri/` were grepped first. **No
prior art on this axis.** The nearest neighbour is `kneemri/evaluation/stack_coverage.py`, which
measures acquired slice DEPTH against AUC over the published CC0 members — a different quantity.

## The harness bug that had to be fixed before any number meant anything

The first pass printed AUCs of **1.0466** and **0.1582** — outside [0,1], so the estimator was wrong,
not the data. Cause: `truth` in `<member>.holdout.npz` is CONTINUOUS (graded LLM-read labels), so
`n1 = y.sum()` summed fractional values while the positive SET was selected by `y > 0.5`. The two
denominators disagreed and the Mann-Whitney formula left the interval. Binarising at `>0.5` first
fixes it, and the fix is verified by reconstruction: the six-member rank average then reads
**0.8693** against the champion's recorded **0.8694**.

## What the data looks like

909 fold-0 studies, six primary slots, six champion seats that keep a `holdout.npz`
(`band_dense_long_s2026`, `calib_span_s2026`, `grade_ladder_s2026`, `base_dense_s2026`,
`lr4e-5_ws_s2026`, `through_dense_s2026`), combined by `RankAverage`.

- Slots present per study, counts 0..6: `[0, 0, 7, 84, 298, 456, 64]` — real spread, 2 to 6.
- Commonest patterns: `111110` x352, `111001` x214, `111011` x95, `111111` x64, `011100` x53.
- corr(ensemble prediction, slot count) runs **+0.159** (Lateral Meniscus) to **-0.097** (PF OA).
  The signs split cleanly: ACL, MCL and both menisci positive; all three OA, Effusion and Baker's
  negative.
- Slot count ALONE as a predictor of the twelve labels: mean |AUC - 0.5| = **0.038**.

## The result: twelve of twelve, both stratifications, negative

```
                ACL 0.9011  cnt -0.0222  pat -0.0616
                MCL 0.8039  cnt -0.0133  pat -0.0341
    Medial Meniscus 0.9104  cnt -0.0022  pat -0.0222
   Lateral Meniscus 0.8679  cnt -0.0047  pat -0.0173
          Medial OA 0.8852  cnt -0.0093  pat -0.0127
         Lateral OA 0.8266  cnt -0.0132  pat -0.0237
              PF OA 0.8691  cnt -0.0096  pat -0.0192
           Effusion 0.8654  cnt -0.0139  pat -0.0165
          Synovitis 0.8732  cnt -0.0055  pat -0.0179
            Baker's 0.9211  cnt -0.0088  pat -0.0095
          Contusion 0.8864  cnt -0.0028  pat -0.0056
           Fracture 0.8216  cnt -0.0038  pat -0.0102
MEAN  base 0.8693   by-count -0.0091   by-pattern -0.0209
```

`cnt` = normalise within slot-COUNT (5 non-empty strata). `pat` = normalise within presence PATTERN
(11 strata, rare patterns folded into their count bucket). Not one label improves under either.

## The reading, which is the inverse of the hypothesis

**The between-stratum ordering carries real label information, and the ensemble is already
exploiting it.** Which sequences a study was scanned with is not an accident of the scanner — it is
a record of what the ordering clinician SUSPECTED. A knee sent for a meniscus question gets a
different protocol from one sent for effusion, and the sign split above is exactly that shape:
ligament and meniscal findings correlate POSITIVELY with a fuller acquisition, the degenerative and
fluid findings negatively.

So the level shift is not orthogonal to the label. Destroying it costs 0.0091 (count) to 0.0209
(pattern), which at the refitted **+2.14 board points per holdout point** is 0.019 to 0.045 on the
board — larger than the entire remaining gap between our 0.934 and the leader's 0.952.

The proposed fix is dead in the direction it was proposed. Two things survive it:

1. **A positive fact about where our AUC comes from.** A measurable share of it is acquisition
   protocol, not pixels. Any future change that flattens or imputes missing slots — a full-mask
   member, an inpainting step, a "fair" per-study renormalisation — is spending this, and now has a
   price tag rather than a shrug.
2. **A question the refutation does not answer**: whether the ensemble captures ALL of the protocol
   signal or only part. That descendant was run the same hour and is answered below.

## The descendant: the ensemble has already taken all of it

`python -m kneemri.evaluation.protocol_prior`. A per-pattern per-label prevalence, fit **leave one
out** so no study reads its own label, blended into the ensemble's ranks at six weights. LOO does not
merely close the leak, it OVERSHOOTS against the hypothesis — dropping a positive lowers its own
pattern's prevalence — so anything this finds is a floor.

```
label             w=0.0    w=0.05   w=0.1    w=0.2    w=0.3    w=0.5    prior-alone
ACL               0.9011   0.9007   0.8988   0.8920   0.8813   0.8599   0.5659
MCL               0.8039   0.8067   0.8090   0.8101   0.8088   0.8003   0.5955
Medial Meniscus   0.9104   0.9062   0.9011   0.8890   0.8737   0.8346   0.4045
Lateral Meniscus  0.8679   0.8652   0.8612   0.8507   0.8376   0.8103   0.4858
Medial OA         0.8852   0.8819   0.8775   0.8680   0.8546   0.8229   0.4569
Lateral OA        0.8267   0.8256   0.8238   0.8179   0.8102   0.7898   0.5168
PF OA             0.8691   0.8676   0.8654   0.8595   0.8502   0.8243   0.4683
Effusion          0.8654   0.8632   0.8601   0.8521   0.8410   0.8137   0.4731
Synovitis         0.8733   0.8724   0.8702   0.8641   0.8555   0.8320   0.5088
Baker's           0.9211   0.9204   0.9189   0.9124   0.9017   0.8743   0.4767
Contusion         0.8864   0.8844   0.8815   0.8733   0.8623   0.8339   0.4293
Fracture          0.8216   0.8175   0.8133   0.8013   0.7875   0.7581   0.4541
MEAN              0.8693   0.8677   0.8651   0.8575   0.8470   0.8212   0.4863
```

**The best blend weight is zero.** Eleven of twelve labels fall monotonically as the prior is mixed
in; the twelfth, MCL, rises +0.0062 at w=0.2 — one noise floor, one label out of twelve, and exactly
the shape a search over six weights and twelve labels produces from nothing.

The `prior-alone` column is what makes this a POSITIVE statement rather than a null. The prior is not
inert: ACL reads 0.566, MCL 0.596, Medial Meniscus 0.405 — signal in both directions, mean deviation
from chance about 0.055, the same protocol-follows-suspicion structure the correlations showed. The
members simply already have it. They see the presence mask; `SlotHead` renormalises over it; nothing
was left on the table for a lookup table to collect.

So both directions of the hypothesis are now closed with numbers. Removing the protocol signal costs
0.0091-0.0209. Adding it explicitly buys 0.0000. It is real, it is used, and it is used up.

## Discipline note

This is the third time in the project that binarising, or naming a denominator, was the difference
between a claim and an artefact. The out-of-range AUC is the cheapest possible version of that
warning: a number that CANNOT be true, produced by a formula that looked right. The reconstruction
check against 0.8694 is what licensed everything above it.
