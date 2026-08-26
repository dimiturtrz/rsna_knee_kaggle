# The holdout orders entries correctly; it just whispers

2026-08-16, from `python -m kneemri.evaluation.board_transfer` over `comp/board.jsonl`.

> **CORRECTION 2026-08-24 — THE CONSTANT IN THIS DOCUMENT IS 2.2x TOO BIG. The title is not.**
>
> Everything below fits the compression over entries spanning **0.0043 offline**, which is inside this
> project's own **0.0062** holdout noise floor. A slope fitted entirely inside the noise floor cannot be
> falsified by the data that produced it, and this one was wrong: the `weakseven` instrument
> (`knee-825t`) was built for no other purpose than to place a point outside that band, scored offline
> 0.8562 against the champion's 0.8694, and read **0.908** against 0.934. That is 0.0132 offline buying
> 0.026 board — a slope of **1.97**, or 2.21 against the chain floor. Re-running the tool over all twelve
> entries now prints **+2.14**.
>
> **The headline claim of this document SURVIVES and is stronger than when it was written.** Inversions
> are 4 of 55 orderable pairs and **0 of the 33 the holdout resolved at all** — every disagreement sits
> under 0.0053. The holdout has still never mis-ordered a pair it could see, and it ordered the largest
> gap ever put in front of it correctly and monotonically. It whispers; it does not lie.
>
> So: trust the SIGN above 0.0053 exactly as this document says. **Halve the MAGNITUDE.** Any figure
> below, or in any document written between 2026-08-16 and 2026-08-24, that converts an offline delta
> into board points at 4.67 is 2.2x too generous. Read the constant from the tool, never from prose.

## The question

Every recipe decision in this project is made on a 58-study holdout scored against LLM-read report
labels, and only five submissions a day can check one against radiologist annotations. knee-7d1
recorded the conversion as "the proxy appears to compress ~2x", fitted to two entries, with its limits
stated. Nine entries now carry both numbers.

## What was measured

Pairs, not a regression. A line through the points would be a claim about the LEVEL of two scores that
were never meant to agree; what a ruler is FOR is ordering candidates. So each pair of entries is asked
the only question that decides a slot: the holdout preferred this one, did the board?

**The denominator did most of the work, and getting it wrong produced a different result.** The first
pass compared all nine entries and reported 7/34 inversions and a median ratio of 1.16. Four of those
entries are not on the same scope: two quote the MEAN of their members' individual holdouts, which sits
below the ensemble it describes by however much averaging bought, and two are a bare number the ledger
never says the scope of. On paper a one-member entry at 0.8518 then outranks a five-member entry at
0.8269 while the board has them the other way round — and that "inversion" is two denominators, not the
board. Restricted to the five entries whose offline number is the holdout AUC of *what was submitted*:

```
5 entries on the ensemble scope, 9 orderable pairs (excluded: 2 member_mean, 2 unknown)
compression: median +4.67 board points per holdout point
inversions: 2/9 pairs the board ordered the other way, 0/5 of the pairs the holdout resolved at all
  trained_v8 over efficient_v4: holdout +0.0029 (UNRESOLVED, under the floor), board -0.008
  chosen_v2  over efficient_v4: holdout +0.0031 (UNRESOLVED, under the floor), board -0.001
```

## Two results

**1. The ruler does not mis-order anything it can resolve. 0 of 5.** Both inversions are pairs the
holdout separates by less than its own 0.0053 noise floor — two entries it called equal, which the
board is free to order either way. The failure mode we feared (selecting the wrong recipe offline) has
not happened once in the range where the holdout expresses a preference at all.

**2. The compression is much larger than the ~2x on record.** The five resolved pairs convert at 1.2,
2.4, 4.7, 5.3 and 8.1 board points per holdout point. So a holdout gap the ruler can *just* see, 0.0053,
is worth roughly 0.01-0.04 on the board.

Those two combine into the useful sentence: **the ruler's resolution floor and the competition's result
bar are about the same line.** Anything the holdout can resolve is worth a submission; nothing it
cannot resolve should be argued about from offline numbers.

## What this does NOT say

- **n=5 entries, and four of the five resolved pairs share `solo_v1` as one endpoint.** The
  one-member entry is the low anchor of almost every ratio quoted; it is not five independent reads.
- **The pairs are not matched.** They span six days and the kit changed underneath them — fp16 serving,
  the image budget, overlap TTA, a member pool that doubled. Only the two 17:45 entries on 2026-08-16
  are a same-day matched pair, and their ratio is 2.37.
- **It does not license reading knob-sized offline deltas.** A ratio of ~4 applied to a delta the ruler
  cannot resolve multiplies noise by four. The floor is what makes the conversion meaningful.
- The earlier note on knee-7d1 — that the board charged 2.4x what the holdout did for collapsing to one
  pixel group — survives this and is the same-day matched point above. What changes is the framing: that
  is not the holdout *mis-ranking* diversity, it is the holdout ranking it correctly and quietly.

## Consequence

`comp/board.jsonl` is now the machine-readable half of `comp/SUBMISSIONS.md`, and every future scored
entry adds a line. Two of the existing rows say `"scope": "unknown"` — the ledger recorded a number
without recording what it was a number OF, which is the same failure this project keeps paying for.
Recovering them is worth a bead, not a guess.

## Update 2026-08-26 09:54Z — the n=5 caveat is retired, and the ruler still never mis-orders

Re-run on the complete record (`board_check` now reconciles 62 of 62 live submissions, against the 34
this file was written from):

```
21 entries on the ensemble scope, 171 orderable pairs
compression: median +1.72 board points per holdout point
inversions: 8/171 pairs the board ordered the other way,
            0/108 of the pairs the holdout resolved at all (a gap >= 0.0053)
```

**The property survived a 12x growth in the sample.** It was 0 of 5 resolved pairs when this page was
written and 0 of 33 by the end of it; it is now **0 of 108**. Every one of the 8 inversions still sits
under the floor — the largest is `peak8_v1` over `gradeswap_v1` at +0.0037 holdout — which is the
signature of a ruler that is quiet where it should be quiet rather than one that is wrong.

**The compression settled at 1.72, down from the 4.67 this page measured.** That figure came from five
entries spanning 0.0043 offline, four of whose resolved pairs shared `solo_v1` as an endpoint; the
warning written here at the time — *"n=5 … it is not five independent reads"* — was correct, and the
ratio fell by two-thirds once the sample stopped being one anchor. 1.72 is the number to convert with.
Quoting 4.67, or the +1.82 that an incomplete `board.jsonl` fitted before 2026-08-25, is quoting a
superseded denominator.

### Why this was worth re-running rather than assumed

The suspicion it was checked against: our offline reads sit ~0.07 below our board scores
(`soloseat_v1`, offline-era member at 0.8512, boards at 0.925), while a competitor reports *"OOF AUC =
LB AUC within noise"*. A ruler that disagrees with the board by 0.07 is a **faulty-TEST** suspect, and
every flat band this project has declared was read on it.

It is not one. The ruler compresses and **does not invert**: a difference it can resolve is a difference
the board agrees with, 108 times out of 108. So the offline holdout is sound for the one job the plan
needs it for — deciding whether an arm moved — and `raddino_dense`'s incoming number can be read against
its pre-registered bands without a board confirmation.

### The one real gap that remains

**27 of the 48 scored entries are one-sided** — `soloseat_v1`, `votefit_v1`, `blendseat_v1`,
`jitterfit_v1`, `blendthree_v1` and every other recent one. They carry a board score and no offline
number *by construction*, because their members are `--fold -1` full-fit and have no uncontaminated
holdout. So 1.72 is fitted entirely on the **older** half of the board, before full-fit members existed.
Nothing indicates the constant has drifted, and nothing could: the regime that would reveal a drift is
exactly the regime that produces no offline number. Convert with 1.72, and treat a conversion involving
a full-fit member as an estimate carrying an unmeasurable term.
