# Severity weighting closes at full budget — and the concentration I read into it was sampling noise

2026-09-13. Subject: `severity_drop40_long_s2026` against its matched control
`band_dense_drop40_long_s2026` (bead knee-evdfd, read out knee-fm87e).

## The pre-registered read

The arm was `severity_weight`'s line with two flags imported — `--slot-dropout 0.40` and
`--epochs 12 -> 24` — because the twelve-epoch harness that had produced the original −0.0048 is
*measured* to bind (+0.0084/+0.0088 holdout on two seated recipes, +0.0111 ledger75 on the matched
plain recipe), and the corpus lever had already been written off on exactly that defect and then
inverted its sign on the re-take. The objective deserved the same budget its control got.

The veto was checked first and does not fire. Neither arm peaks on its last epoch:

```
band_dense_drop40_long_s2026   0.8587 mid   0.8918 l75   peak e16 of 24   (CONTROL)
severity_drop40_long_s2026     0.8538 mid   0.8899 l75   peak e18 of 24
                                             delta -0.0019
```

Pre-registered band `<= 0.0000` **closes the family on a harness that cannot be blamed.** The control
is matched, converged, and peaks mid-run; the budget defence is spent. At twelve epochs the same
subtraction was −0.0006 at ledger75, at twenty-four with the regulariser it is −0.0019. Giving the
objective its budget did not rescue it — it moved the number the wrong way.

## The salvage I reached for, and why it was wrong

The arm's CONCENTRATION clause is binding on every branch, so I took the per-label read before writing
the closure. It looked like a salvage rather than a kill:

```
    ACL                    ctrl 0.9308  sev 0.9402  +0.0094
    MCL                    ctrl 0.8882  sev 0.8649  -0.0233
  G Medial Meniscus        ctrl 0.8986  sev 0.8959  -0.0027
  G Lateral Meniscus       ctrl 0.8775  sev 0.8875  +0.0100
  G Medial OA              ctrl 0.9023  sev 0.9066  +0.0042
  G Lateral OA             ctrl 0.8526  sev 0.8489  -0.0037
  G PF OA                  ctrl 0.8573  sev 0.8757  +0.0184
    Effusion               ctrl 0.9229  sev 0.9285  +0.0056
    Synovitis              ctrl 0.9191  sev 0.9240  +0.0049
    Baker's                ctrl 0.9138  sev 0.9060  -0.0078
    Contusion              ctrl 0.8899  sev 0.8988  +0.0089
    Fracture               ctrl 0.8491  sev 0.8024  -0.0467

graded mean +0.0052 (n=5) · ungraded mean -0.0070 (n=7) · the other ten average +0.0047
```

The story wrote itself: **Fracture and MCL carry the entire deficit, and they are exactly the two
columns where a severity grade has nothing to say** — Fracture is binary, and the label-contest pilot
found 0 rulebook errors on MCL in 100 studies. `__conf` is constant 0.95 on all 52,872 populated
cells, so the loss cannot down-weight itself where its own information is absent. The named
one-variable follow-up was to restrict severity weighting to the columns that HAVE rungs, priced at
~2.2 h of card, and the ten-column arithmetic put the member near 0.8965.

**Both halves of that story are false, and each was cheap to check.**

### The columns all have rungs

Counting the reader's levels per column over the whole training table — the premise the follow-up arm
rested on, never measured before it was written:

```
target                      0.08    0.28    0.68    0.82    0.94   n_pos  rung-spread
ACL                         2825     355     474     293     459    1226        0.613
MCL                         3209     430     495     219      53     767        0.355
Medial Meniscus             1809     250     568    1176     603    2347        0.499
Lateral Meniscus            2843     442     415     529     177    1121        0.528
Medial OA                   1662    1125     517     605     497    1619        0.626
Lateral OA                  1862    1457     443     387     257    1087        0.592
PF OA                       1600     815     543     658     790    1991        0.603
Effusion                    1306     434    1599     910     157    2666        0.400
Synovitis                    114    3712     218     346      16     580        0.403
Baker's                     1269    2023     548     426     140    1114        0.508
Contusion                   2193     916     519     709      69    1297        0.453
Fracture                    1606    2486     122     169      23     314        0.462
```

There is no rung-free column. Fracture carries 122/169/23 across the three YES grades — a spread of
0.462, mid-pack, *higher* than Effusion's or Synovitis's. MCL carries 495/219/53. The claim that the
mechanism has no information on those two columns is contradicted by the reader's own table. The
follow-up arm's one variable does not exist, and with it the 2.2 h of card it was going to cost.

### The two "losers" are the two rarest columns, and neither delta clears noise

CLAUDE.md names this error by name — *"a per-label delta read without its own sampling interval"* — and
I made it again. Hanley–McNeil standard errors on the holdout, per column:

```
target                  n_pos    ctrl     sev    delta    SE    d/SE
ACL                       152  0.9308  0.9402  +0.0094  0.0145  +0.65
MCL                        57  0.8882  0.8649  -0.0233  0.0288  -0.81
Medial Meniscus           482  0.8986  0.8959  -0.0027  0.0105  -0.26
Lateral Meniscus          140  0.8775  0.8875  +0.0100  0.0194  +0.52
Medial OA                 233  0.9023  0.9066  +0.0042  0.0138  +0.31
Lateral OA                128  0.8526  0.8489  -0.0037  0.0217  -0.17
PF OA                     312  0.8573  0.8757  +0.0184  0.0144  +1.28
Effusion                  241  0.9229  0.9285  +0.0056  0.0122  +0.46
Synovitis                  77  0.9191  0.9240  +0.0049  0.0217  +0.23
Baker's                   217  0.9138  0.9060  -0.0078  0.0135  -0.57
Contusion                 160  0.8899  0.8988  +0.0089  0.0174  +0.51
Fracture                   42  0.8491  0.8024  -0.0467  0.0377  -1.24
```

**Not one of the twelve deltas reaches 1.3 standard errors.** Fracture's headline −0.0467 is −1.24 SE
because it has **42 positives in the holdout**; MCL's −0.0233 is −0.81 SE on **57**. Those are the two
rarest columns in the table by a clear margin — the next scarcest, Synovitis, has 77 — and rarity
tracks the absolute delta across the twelve (rank correlation +0.45). The "concentration" was the two
widest error bars in the panel, read as a mechanism.

Nothing about the graded/ungraded split survives either: the graded mean +0.0052 and the ungraded
−0.0070 are averages of twelve numbers whose individual intervals all straddle zero, and the split that
produces them was chosen after seeing them.

## What is closed, and what this costs

**`--severity-weight` closes at full budget**, as pre-registered: −0.0019 ledger75 against a matched,
converged, mid-peaking control. It stays in the codebase as an opt-in switch — the mechanism behind it
is real (the three YES grades agree at 0.378 / 0.688 / 0.958, three intervals that do not overlap) and
a corpus with a wider severity axis could still want it. It does not earn a seat here.

**The per-column restriction is not a follow-up, it is loss-engineering.** Even if some column property
had predicted the two losers, hand-picking ten of twelve columns to weight is fitting the objective to
a specific weakness, which the project's own rules forbid. That no measured column property predicts
them — not rung spread, not retained positive mass, not shrinkage factor — settles it without needing
the rule.

**The aux-objective axis does not re-open on this arm.** It is not refuted in general; `knee-keemo`'s
supervision KINDS we do not have (localized, slice-level, paired-degradation) are untouched by this.
What is refuted is the cheapest member of that family, on its own best harness.

The saving is the point: one afternoon's arithmetic — a level histogram and twelve standard errors,
both seconds of CPU — retired a 2.2 h card arm whose premise was false and a finding that would have
gone into the ledger as a fact about the objective when it was a fact about 42 fracture cases.
