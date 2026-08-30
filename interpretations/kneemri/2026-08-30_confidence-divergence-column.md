# The confidence column, and what its first run said about the roster

2026-08-30. knee-c2ew use 1, built and read the same day. `kneemri/evaluation/member_divergence.py`,
printed by `python -m kneemri.kaggle.seat_value` beneath `MemberDistance`.

## What it measures, and why a third column

`MemberDistance` reads rank correlation. Rank correlation is invariant to any monotone rescaling of a
member's scores **by construction**, so two members that order every study identically read as twins
there however confidently they predict — one saturated at 0.02/0.98, the other hedged at 0.45/0.55.
That difference is exactly what a LOGIT average is sensitive to, and the champion has shipped
`LogitAverage` since `kneemri-vote`. A ruler blind to the axis the shipped vote combines on is a ruler
that is silent about the shipped vote.

`MemberDivergence` reads Jensen-Shannon in bits between the members' predictions as Bernoulli beliefs,
per label then averaged — the same shape `RankAverage.agreement` uses, for the same reason. JS rather
than the KL the bead named, on the bead's own counter: KL is asymmetric, unbounded, and explodes where
the reference has near-zero mass, which is the rare-finding regime our per-label scores are weakest in.

## First reading: the two columns do NOT order the roster the same way

Against `honest12` (0.937, twelve members), roster of 12:

- **Candidates rank identically on both columns.** `swin_supervised` is both the most rank-distant
  (0.8346 correlation) and the most divergent (0.0218 nearest / 0.0383 mean bits); `contrast_pool` is
  nearest on both. On this trio the new column says nothing the old one did not.
- **The SEATED roster separates, and that is the reading.** `margin_seat` is the most divergent seat
  (0.0665 bits to its nearest neighbour) at only 0.8772 rank correlation — mid-pack by rank, an outlier
  by belief. `mae_matched` is by far the most rank-distant seat (0.4691) at 0.0598 bits. Two different
  members top the two columns. The column is therefore not a restatement of the rank one.

Read `margin_seat` conservatively: this module's own honest counter is that a member which merely
disagrees about CALIBRATION — same ordering, wider spread — scores high here and buys what a
temperature knob would. `margin_seat` at 0.8772 correlation is not a twin, but it is not the
high-divergence-AND-low-correlation pair the module says to look for either. The pair to distrust is
high divergence alone.

## Which member tops which column, checked against what the ledger already knows

The disagreement is not symmetric, and this is the one thing in the run worth carrying forward.

- `mae_matched` tops the RANK column (0.4691) and knee-nj4l's census already prices that reading: it is
  DINOv2-small like the other 111 members, ran FOUR epochs against a 12-epoch family, and peaked at
  0.6735. It is rank-distant because it is undertrained, not because it sees the knee differently.
- `margin_seat` tops the DIVERGENCE column (0.0665 bits) and the ledger reaches the opposite verdict
  about it independently: 0.8525 solo, −0.0053 against its own control, and greedy picks it SECOND of
  sixty-five — "the first candidate this project has measured into the strong-and-distant cell"
  (`experiments/README.md`, 2026-08-24).

So on this roster the rank column's top seat is an artefact of a short run and the divergence column's
top seat is the member a board-adjacent greedy already singled out. That is the first corroboration this
ruler has, and it is **n = 1 and directional** — one roster, no board split, and the module stays
unvalidated as a predictor until a seat decision turns on it and lands. It is not a licence to spend a
slot; it is a reason the column keeps printing.

Seat values in the same run were all under the +0.0015 bar (best +0.0005, `stack_increments`), so
nothing here is a shipping decision. The column is UNVALIDATED as a predictor and weaker than the rank
one: it was written from a mechanism, not from a board split, and no submission slot should turn on it
until one does and lands.

## The defect its own first run found

The support column was written as `truth[:, label] > 0` and read ~99% on **every one of the twelve
findings**. `RecordMix` does not build a binary indicator: over the 909-study fold-0 holdout the truth
vector takes seven values — `0, 0.08, 0.28, 0.68, 0.82, 0.94, 1.0` — a reader-agreement scale. A `> 0`
count is a statement about how rarely all readers agree on absence, and says nothing about prevalence.

Fixed at the source rather than in the caller: `PRESENT = 0.5` with the measurement in its docstring, the
denominator and threshold printed on every line, and the unit fixture made SOFT so a `> 0` count can
never pass the test again. At 0.5 the same columns read 68 to 558 of 909, which is the prevalence the
ledger quotes.

This is the CLAUDE.md denominator rule landing on a ruler on its first use: the fix is one line of
arithmetic, and it is the difference between a support column and a decoration.
