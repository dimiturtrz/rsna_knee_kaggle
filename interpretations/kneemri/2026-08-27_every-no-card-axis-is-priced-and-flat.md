# Every axis reachable without the card is priced, and the champion sits on the best side of all of them

2026-08-27, evening. Source: `comp/board.jsonl` (58 submissions, complete since `board_check` closed the
24-row gap on 2026-08-25), `experiments/README.md`, and the two interpretations filed earlier today.
No compute was spent producing this — it is a read of the record we already own.

## What was asked

The card goes to the peer project tonight, so the lanes left are research, refactoring, bug-fixing and
preserving results. One submission slot remains, held for ~23:30Z. The question is what to spend it on,
and the honest prior answer was "a roster permutation" — which is what the last several slots bought.

## The answer, in one table

Every axis that can be moved WITHOUT training a new member has now been measured, most of them by a
controlled single-variable board pair rather than by a sort:

```
axis                evidence                                       verdict
------------------  ---------------------------------------------  ------------------------------
seat count          regime-stratified, offline curve + board       flat within a regime
vote rule           efficient_v4 0.919 -> vote_v2    0.919         flat  ( 0.000 at 2 seats)
                    dark6_v1     0.937 -> votefit_v1 0.938         flat  (+0.001 at 6 seats)
blend dose          votefit 0.938 -> blendseat 0.939 -> 0.939      saturates at dose 1
test-time jitter    votefit 0.938 -> jitterfit_v1 0.937            negative
fit regime          ema_dense ablation + allfit                    +0.004, ALREADY in the champion
calibrated targets  votefit 0.938 -> calibsix_v1 0.941             +0.003, ALREADY in the champion
```

The champion `calibsix` is at **0.941** and holds the winning side of every row. Its own supersets tie
it and do not beat it: `calibmix` at eleven seats reads 0.941, `pixgroup` at eight reads 0.940.

**So there is no zero-training submission with a positive expected score tonight.** That is a complete
answer rather than a shrug — it is the union of six measured axes, not an absence of ideas.

## The two axes that DID pay, and what they have in common

Both are **treatments applied to every seat at once**, not seat permutations:

- `calibsix` — `--calibrated-targets` on all six seats. Its own board note is explicit that no seat
  moves and no member enters or leaves. **+0.003.**
- the fit regime — `--fold -1` on every seat. **+0.004**, and the `ema_dense` arm separated it from the
  `--ema` that rides along with it, so the credit is data, not the averaging.

Every seat permutation tried since is worth between −0.004 and +0.001. The shape that carries is
**change what all the members are, not which members sit.** That is the design rule this board has
actually taught us, and it is worth more than any single entry on it.

## Where that points the card, tomorrow

The two paying treatments have never been CROSSED. `calibsix` is calibrated targets on six seats;
`blendthree` is blend supervision on three, and blend supervision is the only other treatment of the
all-seats shape (it changes the label SOURCE — external expert reads from Rijeka — where calibration
re-scales the label VALUES). Different mechanisms on different parts of the same pipeline is the
standard precondition for two effects to add rather than to overlap.

The honest discount: the blend dose curve is flat from one seat to three, so blend's own axis is
saturated and the cross may inherit that. What makes it worth arming anyway is that dose was measured
as seat REPLACEMENT while calibration was measured as an all-seat TREATMENT, and the board says only the
second shape pays. A calibrated blend twin has never existed, so the cross has not been measured in the
shape that works.

It needs training — a blend-supervised twin built with `--calibrated-targets` — so it is a card
question, not a slot question. Filed as the first thing to arm when the card comes back.

## What this costs the slot

It says plainly what the slot cannot buy. Firing it at another roster permutation would be the seventh
reading of an axis whose six previous readings all landed inside one board tick, and the ledger already
calls that class of spend out. Whether to fire it anyway on the best-argued null, or to hold it, is the
owner's call — this document is the evidence for making it, not the decision.

## Three false starts this session, all caught by the same rule

Worth recording because the rule earned its place:

1. A board-wide sort said the fit regime was the biggest structured effect on the board. The ledger
   already had it, with a CONTROLLED arm (`ema_dense`); my sort was confounded with date.
2. The seat-count curve predicted +0.0064 through `board_transfer`'s +2.14 slope. That slope was fitted
   on member QUALITY and cannot price a count delta.
3. A P1 was filed today claiming the vote rule had never been measured offline, because the ruler
   defaults to `RankAverage` while the champion serves `LogitAverage`. `comp/board.jsonl` holds two
   controlled pairs on exactly that swap. Corrected in `knee-brkd` before any compute was spent.

All three were prevented by reading `experiments/README.md` and `comp/board.jsonl` before spending —
CLAUDE.md's read-the-record-first rule. Three times in one window is not a coincidence about this
window; it is what the rule is for, and the third one was caught cheapest of all because the board is a
single file a query can answer in a second.

Related: `knee-brkd` (the corrected vote bead), `2026-08-27_seat-count-is-flat-both-rulers.md` (the
population mismatch that discounts every offline->board transfer), `knee-9g28`, `knee-e620`.
