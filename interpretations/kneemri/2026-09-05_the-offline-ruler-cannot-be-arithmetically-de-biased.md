# The offline ruler cannot be arithmetically de-biased

**Date:** 2026-09-05 · **Beads:** knee-zsyp4 (biased ruler), knee-wko44 (labeler axis)

## The question

Every score this project reports is agreement with a lexicon reader's labels, and the holdout is scored
against the *same* table the members train on. knee-zsyp4 measured the consequence from the board side:
the `colmcl` constant-column probe (ref 56020054) puts the board's MCL column at 0.980 against our
offline 0.8095. If the deflation were a known function of the label noise, every per-column number in
the ledger could be corrected arithmetically and no new annotations would be needed.

## The closed form

Under class-conditional noise with rates `rho+ = P(reader says not-YES | gold YES)` and
`rho- = P(reader says YES | gold NO)`, and clean prevalence `pi`, the noisy positive pool is a mixture
of true positives with weight

    w_p = (1-rho+)pi / [ (1-rho+)pi + rho-(1-pi) ]

and the noisy negative pool is a mixture of true negatives with weight

    w_n = (1-rho-)(1-pi) / [ rho+ pi + (1-rho-)(1-pi) ]

A random noisy-positive beats a random noisy-negative in four disjoint ways, so with `a` the clean AUC

    AUC_noisy = w_p w_n a + 0.5[ w_p(1-w_n) + (1-w_p)w_n ] + (1-w_p)(1-w_n)(1-a)

which is **affine in `a` and invertible** whenever `w_p w_n != (1-w_p)(1-w_n)`. So the answer to "can a
noisy-label AUC be corrected to a clean-label AUC" is yes, in closed form, distribution-free, given the
flip rates. That much is settled and needs no literature.

## Applying it to our flip rates puts every column out of range

Flip rates measured over the 57 radiologist-annotated studies (684 cells, `ReaderAgreement.cells`),
AUCs from `band_dense_s2026`'s best epoch on the 909-study fold-0 holdout:

    finding                 P   N  rho+   rho-  ceiling (95% CI)      band_dense
    ACL                    23  34 0.000  0.265   0.859 [0.781,0.932]     0.874
    MCL                     9  48 0.000  0.188   0.750 [0.632,0.868]     0.793
    Medial Meniscus        26  31 0.038  0.226   0.871 [0.784,0.946]     0.905
    Lateral Meniscus       23  34 0.130  0.294   0.778 [0.669,0.876]     0.846
    Medial OA              15  42 0.067  0.190   0.804 [0.698,0.905]     0.880
    Lateral OA             11  46 0.182  0.152   0.757 [0.628,0.885]     0.816
    PF OA                  20  37 0.200  0.216   0.773 [0.659,0.879]     0.858
    Effusion               34  23 0.000  0.696   0.840 [0.771,0.902]     0.860
    Synovitis              27  30 0.519  0.100   0.736 [0.607,0.848]     0.876   OVER
    Baker's                12  45 0.083  0.111   0.832 [0.709,0.941]     0.907
    Contusion              18  39 0.111  0.513   0.675 [0.570,0.777]     0.871   OVER
    Fracture               17  40 0.235  0.150   0.789 [0.670,0.904]     0.809

`ceiling` is `AUC_noisy` evaluated at `a = 1`: the best score a **perfect** clean ranker could post
against labels this noisy. Inverting the formula on the point estimates returns a clean AUC above 1.000
in **all twelve** columns — MCL at 1.086, Contusion at 1.561, macro 1.142 against a measured 0.8578.

## What that is and is not evidence for

An out-of-range inversion has two explanations and this measurement does not separate them cleanly.

1. **The flip rates are overestimated.** They come from 57 studies. The ceiling's CI, bootstrapped over
   STUDIES (cells within a study share a report and a radiologist and are not independent draws), is
   about +/-0.11 wide. Under that CI only **2 of 12** columns — Synovitis and Contusion — exceed the
   upper bound, and those are exactly the two columns carrying the largest single flip rate
   (rho+ = 0.519, rho- = 0.513). Two of twelve at a 95% bound is above the 0.6 you would expect by
   chance but it is not a proof.
2. **The noise is not class-conditional.** The reader and the model read the same anatomy through the
   same report vocabulary, so where the reader over-calls, the model has a correlated reason to score
   high. Instance-dependent noise correlated with the score INFLATES measured AUC relative to what CCN
   predicts, which is the direction observed in all twelve columns.

The point estimates favour (2) unanimously; the intervals only support it in two columns. Treat it as
**directional**, not as a finding, and note the second denominator caveat: the flip rates are measured
on 57 studies and the AUCs on a 909-study holdout, so the CI prices the ceiling's sampling error and
not the population mismatch between the two.

## What it decides

**The cheap fix is dead.** There is no per-column correction factor that can be applied to the ledger:
the correction the arithmetic asks for is out of range in every column, so publishing corrected numbers
would replace a biased ruler with an impossible one. Member selection stays on the ruler it has been on,
and no reseat decision gets re-read on corrected numbers.

That leaves fixing the LABELS as the only route to an honest per-column number, which is knee-wko44 —
independently supported here by MCL's 9 gold positives against 9 false positives, a reader that doubles
the MCL positive rate in the column the board scores at 0.980.
