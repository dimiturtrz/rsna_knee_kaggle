# Every teacher this project can reach is at or below its student on the honest ruler

*2026-09-17, ~0.3 h CPU, no card. Nothing was run: three numbers already in the record are multiplied
together for the first time. The third of them is seven hours old, which is why the product was not
available before today.*

## The three facts

**(1) The ensemble premium is ≤0.002 on the board.** `56296698`, submitted 06:09Z today, is ONE member
(`labelwise_depth24_drop40_long_ff`) at **0.944** against the seven-member champion `tripleseat_v2` at
**0.946**. Two ticks of the competition's own 0.001 resolution, 1.5 sd on the top band's residual sd of
0.0013. See `2026-09-17_one_member_is_the_whole_ensemble.md`.

**(2) The distillation treatment effect is +0.0056, and the teacher's SOURCE does not matter.**
`experiments/README.md:8545-8600`. The leak-free reads:

```
seat_distill_lr4e5   0.8633   our own OOF teacher, --exclude-fold        +0.0055 vs band_dense 0.8578
seat_clean_distill   0.8634   the clean refit of the same term           +0.0056
foreign_distill_mix07 0.8656  a FOREIGN teacher at the same weight       +0.0061 vs its own control
```

Two independent reads of our own term agree to 0.0001, and the foreign term sits 0.0025 from them —
well inside the 0.0062 paired floor. The ledger's own verdict: *"DISTILLATION IS THE LEVER (+0.006 to
+0.011), THE TEACHER'S SOURCE IS NOT."* The +0.0107 that circulated for a day was the **teacher's**
delta written into a **student's** sentence, and it ordered 8.5 h of card before it was corrected.

**(3) The strongest teacher this project can reach is BELOW its student on the board.**
`experiments/README.md:5400` prices `raptor_v4_swa` — the foreign member the champion seats — at
**board 0.923 solo**. Our own solo is **0.944**. The teacher is **0.021 behind the student** on the only
ruler neither of them trained on.

## The product

Distillation moves a student toward a teacher. Its gain is bounded by what the teacher knows that the
student does not, and there are exactly two sources of that here:

- **An ensemble teacher** — bounded by fact (1) at **0.002**. A student distilled from our own seven
  seats cannot be handed more than the seats have over one of them, and that is two ticks.
- **A stronger single teacher** — refused by fact (3). There is no such teacher on disk. `raptor_v4_swa`
  is the best foreign model this project owns and the board puts it 0.021 the wrong way.

The gap to first place is **0.011 board**. Both bounds are below it, one by 5x. So the distillation
axis cannot reach first place — not at a better weight, not at a better temperature, not composed onto
a better recipe. **That is a statement about the available teachers, not about distillation**, which is
the distinction fact (2) makes payable: the term is real and measured, it is simply already spent.

## The correction this page also owes

A top-14 solo ranking by `ledger75` was built earlier today and its first four rows are **void**.
`comp/members.declines:815-821` had already ruled them in-sample against the identical 909-row set at
**98.6%** overlap (`foreign_overlap`, the `raptor_solo` mechanism):

```
raptor_v4_swa     0.9425  VOID, in-sample   board 0.923 solo
raptor_v4         0.9399  VOID, in-sample
raptor_v9         0.9265  VOID, in-sample
raptor_effv2l480  0.9139  VOID, in-sample
foreign_oof_corpus 0.9051 flagged (move +0.0122), already declined
```

Read as printed, that table says a foreign member is 0.037 ABOVE our best. Read honestly, the same
member is 0.021 BELOW it — a sign flip on the largest apparent number in the repo. The honest top of
the solo ranking is what it already was: `labelwise_depth24_drop40_long_s2026` at **0.9053**.

The failure mode is the one this repo keeps paying for: `ledger75` prices whatever holdout file it is
handed and knows nothing about who trained on those studies. The decline file knew; the ranking did
not consult it. `roster_check`'s rule caught this class again without being designed for it.

## What is NOT closed

- **Distillation as a composed increment.** +0.0056 is a real operator and the shipping recipe does not
  carry it. `CLAUDE.md` licenses chains of sub-floor increments explicitly. It is a knob at 1.5 h
  (221 s/epoch x 24), not a path, and it must be priced as one.
- **A teacher from outside this project.** Fact (3) is a statement about what is on disk. Any model that
  is genuinely above 0.944 on the board would reopen this immediately.
- **knee-vkn45** — teacher exposure is a fourth honesty filter and nothing enforces it. This page is
  more evidence for that bug, not against it.

## Denominators

Board numbers: the competition's held-out set at 0.001 resolution, `comp/board.jsonl`, refs 56296698
(0.944), 56232529 (0.946), and the 0.923 at `experiments/README.md:5400`. Offline numbers: `ledger75`,
909 held-out studies, fold 0 / split_seed 2026. Paired floor 0.0062. Top-band residual sd 0.0013 on
`ledger75`, n=15, detrended against slope +1.09. Gap to first: leader 0.957, ours 0.946, read live
2026-09-17T12:39:10Z, rank 65 of 3885. In-sample overlap 98.6%, `foreign_overlap` over fold 0's 909.
