# The residual is near-isotropic, so the failure is not study-level — and that kills my own focal/peripheral story

2026-09-17. Bead knee-2co4r. No card spent: one pass over
`labelwise_depth24_drop40_long_s2026.holdout.npz`, 909 studies × 12 labels, residual taken in RANK space
so the labels share one scale despite a wide prevalence spread. Truth cut 0.75.

**The answer, stated first: a SHARED, STUDY-LEVEL aggregation failure — the model reading the wrong part
of the volume, so a study it gets wrong is wrong on all twelve labels together — is EXCLUDED. That
failure predicts a dominant first principal component and a concentrated set of hard studies, and
neither is present. This is the fourth refuted cause of the class-B per-column holes, and it retires a
story of my own before it could cost a card.**

## The measurement

| statistic | value | what a study-level failure would predict |
|---|---|---|
| cross-label residual correlation | mean **+0.090**, median +0.080, max +0.317 | strongly positive across the board |
| leading PC variance shares | **0.179, 0.116, 0.102, 0.089** — flat | one dominant PC |
| worst 1% of studies carry | 1.8% of total residual | far more than 1% |
| worst 5% | 8.1% | |
| worst 10% | **15.5%** (uniform would be 10%) | a large multiple of 10% |
| worst 25% | 35.4% | |
| ORACLE macro with PC1 projected out | 0.9340 (+0.0223 over 0.9117) | — uses truth, upper bound only |

The spectrum is flat and the error is spread almost uniformly over studies. There is a mild
concentration — the worst decile carries 1.55× its share — but nothing like a set of studies the model
systematically misreads.

**A per-label failure is NOT excluded.** Only the shared study-level one is.

## The story of mine this kills

My per-label table (my statistic, truth cut 0.75, macro 0.9117 — **not** ledger75's 0.9053) reads
Fracture 0.8424, Lateral OA 0.8792, MCL 0.8830 and Lateral Meniscus 0.8966 as the weak four, against
Baker's 0.9561, ACL 0.9458 and Medial Meniscus 0.9430.

I read that as focal/peripheral structures scoring worse and central/diffuse ones better. **That is the
GEOMETRY hypothesis, and this repo already refuted it with the sign REVERSED** — widening helps the
CENTRAL columns twice as much as the peripheral ones, MCL's best band is the NARROWEST (0.7908 at
0.2–0.8 against 0.7869 at 0.1–0.9 and 0.7823 at full), and the band acts through TRAINING rather than
inference, because the outer slices carry nothing the model would use even if it were shown them. The
story was not new, and it was backwards. **Do not re-open it.**

## What converges across two instruments

Three of the weak four — **Fracture, Lateral OA, MCL** — are exactly the three class-B holes that
`2026-09-11_seat_value_and_the_per_column_label_wall.md` priced independently nine days earlier on a
different instrument and a different ruler (MCL +0.121, Lateral OA +0.121, Fracture +0.115 below a
perfect-truth ceiling). Two instruments, two rulers, the same three columns. That agreement is the
strongest thing in this note.

The fourth weak label is not one of them. **Lateral Meniscus is class C — WALLED.** The pool's best
members already score at or above its perfect-truth ceiling (v2 contradicted on 89 of 89 dissent cells),
so measured gain there is learned label noise. That also weakens the lateral-meniscus thread in
`2026-08-20_pooling-is-closed-on-the-mean-but-not-on-lateral-meniscus.md`: the label it pre-registered
and found moving is the one that is walled.

## What this does NOT license

- **The per-label arithmetic is not a lever.** Lifting the weak four to ~0.93 computes to +0.0182 macro.
  That is an ARITHMETIC IDENTITY, not a mechanism. The label lane is closed twice by exhaustion
  (2026-08-26, 2026-09-03) and every label-correction arm is negative: `confident_span` −0.0116,
  `errors_dense` −0.0158, `v2_corrected` −0.0173, matching knee-dry3h's 26 of 26 negative on
  `--label-errors`. **Do not re-open the label lane on this arithmetic.** It is the specific move this
  project has made and retracted more than once.
- The oracle bound uses the truth and is not achievable; it says the residual has 0.0223 of structure in
  it, not that 0.0223 is available.
- One holdout, one split, one seed, one checkpoint. Rank-space residual is a CHOICE — a probability-space
  residual might show more shared structure, and that variant has not been run.
- Per-label AUC noise here is wider than the 0.0062 mean floor, and **no per-label interval was
  computed**, so the ordering of the weak four is not itself resolved.

## Where that leaves it

The class-B holes on MCL, Lateral OA and Fracture are real and now carry **four refuted causes**
(rarity — Spearman −0.333, p=0.29; external corpus coverage — MCL +4, Fracture +0, Lateral OA +0;
geometry, reversed; and shared study-level aggregation, here) plus **one closed** one (supervision).

No mechanism is identified. **That is an open QUESTION, not a lever, and it does not earn a card until
something names a mechanism that could plausibly carry 1.5%.** Writing it down is the point: the holes
keep attracting arms, and each of the five causes above was worth a card to somebody before it was
measured.
