# The offline ruler cannot cross the all-data boundary

**2026-08-25.** Three `record_mix` runs, no card, no submission slot. The instrument re-derived a
shipped entry without being told about it — and in doing so priced that entry fourteen board ticks
wrong.

## The measurement

`record_mix --seats honest` refits the greedy on half the 909 fold-0 studies and scores each prefix
on the other half, both directions averaged. Run over eleven members — the five fold-0 twins of the
champion's six, plus the six mounted members the 77-member honest curve had seated:

```
 1 members: held-out 0.8583   in-sample 0.8598   bias +0.0015
 2 members: held-out 0.8652   in-sample 0.8669   bias +0.0018
 4 members: held-out 0.8705   in-sample 0.8720   bias +0.0015
 6 members: held-out 0.8722   in-sample 0.8729   bias +0.0007
 8 members: held-out 0.8731   in-sample 0.8731   bias -0.0000
 9 members: held-out 0.8728   in-sample 0.8727   bias -0.0001
11 members: held-out 0.8721   in-sample 0.8720   bias -0.0001
best honest count 8 at 0.8731, against the in-sample peak 0.8731 at 8
```

The eight it names, in greedy order: `small_batch6`, `margin_seat`, `calib_span`, `through_dense`,
`band_dense_long`, `sided_pool`, `aprior_dense`, `crop80_span`.

**That is `peak8_v1`, member for member.** The entry has been on the board since 2026-08-22 at
**0.932**, offline 0.8731. The instrument was given eleven names and no history and re-derived a
shipped ensemble exactly. Whatever else follows, the offline ruler is doing the thing it says it does.

## And the same instrument prices that entry fourteen ticks wrong

The five champion twins score **0.8687** (LogitAverage; 0.8684 RankAverage). `peak8` scores
**0.8731**. That is **+0.0044**, which at the measured compression (`board_transfer`, +1.72 board
points per holdout point over 171 pairs) predicts `peak8` should sit about **eight ticks above** the
champion.

`peak8` read 0.932. `votefit_v1` reads 0.938. **Six ticks the other way — a fourteen-tick miss.**

This is not the inversion clause. That clause covers 8 of 171 pairs, every one of them under the
0.0053 floor, none of them resolved by the holdout at all. This is a resolved offline gap of 0.0044
pointing confidently in the wrong direction by more than the entire span of the board.

## Why, and it is not a ruler failure

The champion's six members are **all-data**: `_ff` refits and `final_ws_e12`, fitted at `--fold -1`
on all 4,406 studies. They have seen every holdout study, so they cannot be scored offline at all.
The 0.8687 is not the champion's number. It is the number of **five different, weaker models** — the
same recipes fitted on 3,497 studies — standing in for it.

So the comparison was never champion-versus-`peak8`. It was handicapped-copy-of-champion versus
`peak8`, and the handicap is the whole disagreement.

**The all-data dose does not explain the size of it.** That axis was measured directly
(`2026-08-24_the_all_data_dose_response.md`): one all-data seat bought about +0.001 board, the next
five about +0.003 together, +0.004 to +0.005 in total. The miss here is fourteen ticks. Three times
the dose, and the honest position is that **the size of the boundary effect is unknown and larger
than any offline delta this project can produce.**

## The rule

> **An offline number may only be compared to another offline number fitted on the same studies.**
> Across the all-data boundary there is no comparison — not a weak one, not one with a correction
> applied, none. Every candidate that mixes all-data and fold-0 members is priced blind, and the
> only ruler that speaks to it is the board.

`comp/champion.json` already says the pin is *"fully blind"* and that seat scoring is impossible on
it. What was missing is the second half: the blindness is not merely an absence of a number, it is a
**trap that produces a confident wrong one** as soon as a twin is substituted to fill the gap.

## What the board says directly, which is the evidence that should have led

`lastseat_v1` is the mixed candidate at n=7 — the all-data six plus one fold-0 member
(`small_batch6_s2026`) — and it read **0.937** against `votefit`'s 0.938. One fold-0 seat cost a
tick. The dilution mechanism is plain: members that held out 909 studies, averaged against members
that saw everything, drag the strong half toward the weak. Six such seats is that experiment six
times over.

Together with the five seventh-seat trials (`dark7` 0.937, `lastseat` 0.937, `augseat` 0.936,
`v3seat` 0.936, `farseat` 0.937, `allfit` 0.937, against `dark6` 0.937 and `votefit` 0.938), the
count axis is **closed in the all-data pool**: seven has been tried five ways and every one is flat
or down, `_ff` refits are closed (knee-97ib) so no eighth all-data member can be built, and the
fold-0 members that would fill the seats cost a tick each.

## The second finding, which is smaller and useful

**Selection overfit scales with the size of the candidate pool, not with the number of seats.**
Over 77 members the honest bias runs +0.0035 at six seats. Over these eleven it is **+0.0007 at six
and −0.0000 at eight**. Same greedy, same split-half protocol, same 909 studies. A greedy given
seventy-seven ways to fit the split-half noise finds some; a greedy given eleven cannot.

The consequence for procedure: **curate the candidate pool by mechanism first, then run the greedy.**
The broad-pool honest peak (twelve seats, 0.8720) is *lower* than the curated-pool honest peak (eight
seats, 0.8731) — the extra sixty-six candidates did not add a better ensemble, they added selection
noise the honest split then had to subtract back out.

## What this does NOT say

It does not say the offline ruler is broken. Inside the fold-0 pool it inverts on 0 of the 108 pairs
it resolves, and here it reconstructed a shipped entry from scratch. It says the ruler has a
**boundary**, that the boundary is exactly the `--fold -1` line, and that substituting a twin to
reach across it is the specific move that turns a blind spot into a wrong answer.

It does not retire `peak8` as a bad ensemble. `peak8` is the honest optimum of the fold-0 pool and
0.932 is a real score. It is beaten by an all-data six for a reason that has nothing to do with how
its members were chosen.

It says nothing about `merged_blend`, `blendlabels` or `merged_corpus`, which are all fold-0 arms
compared to fold-0 controls on the same 909 studies — the one place the ruler is sound.
