# The transfer constant is a curve, and we are standing on the flat part

*2026-08-26, on CPU, while `margin_calib` held the card. No slot was spent to learn this.*

## What was believed

`board_transfer` reports one number — **+1.72 board points per holdout point** — and that number prices
every arm in this repo. It is quoted inside shipped kernels, inside `paired_floor`'s docstring, and in
every plan that has translated an offline delta into a board move. The 0.950 target has been arithmetic
on it all week: 0.939 → 0.950 is +0.011 board, so at 1.72 we need **+0.0064 offline**, and +0.0064 is
close enough to a good arm that the goal has always looked one lucky flag away.

## The measurement

The constant is the **median of pairwise board/holdout ratios** over every orderable ensemble pair. That
estimator is well chosen — the docstring's argument against a fitted line (pairs are neither independent
nor matched) is right, and I did not replace it. What it is not is **local**. It mixes pairs spanning the
entire history, from `solo_v1` (offline 0.8518, board 0.868) to `peak8_v1` (0.8731, 0.932), and every arm
we would price today sits at the top of that range.

Re-read the same estimator at floors the record itself supplies, plus an OLS fit as an independent check:

| regime | estimator | value | 2000-draw interval |
|---|---|---|---|
| all pairs, resolved gap | median ratio | **+1.72** | [1.42, 1.89] |
| all ensembles | OLS | +2.22 | [1.21, 3.14] |
| offline ≥ 0.860 | OLS | +1.45 | [0.76, 2.10] |
| offline ≥ 0.860, resolved | median ratio | +1.40 | [1.11, 1.79] |
| offline ≥ 0.8648 — **where we are** | OLS | **+0.83** | [0.55, 1.47] |
| offline ≥ 0.8648, any gap | median ratio | +0.88 | [0.59, 1.11] |
| offline ≥ 0.8648, resolved | median ratio | **+0.74** | [0.49, 0.85] |

The whole-range cell reproduces the tool's own +1.72 exactly, which is what makes the rest a like-for-like
comparison rather than a different statistic. Two estimators that share no functional form agree on ~0.8
in the current regime, and **both current-regime intervals exclude 1.72**. The fall is monotone across
three nested regimes.

## The caveat, which is not small

The top band spans **0.0083** of offline while the offline noise floor is **0.0053–0.0062**. Dividing a
board gap by an offline gap barely wider than the noise in that gap attenuates any ratio estimator toward
zero, and restriction of range does the same to the OLS slope. So the three regime intervals overlap, and
**saturation itself is not resolved**. The structural slope may still be 1.72.

That does not rescue the pricing, and the reason is worth stating carefully. An arm is priced from an
**observed** offline reading, and that reading carries exactly the noise that attenuates the estimate. The
regression of board on a *noisy* offline is the thing that predicts a board move from a number we actually
measure — so **+0.8 is the correct predictive coefficient whether or not the saturation is real.** The
structural question is interesting; the pricing question is already answered.

## What it changes

0.939 → 0.950 is +0.011 board:

| coefficient | offline needed |
|---|---|
| +1.72 (what the repo assumes) | +0.0064 |
| +1.11 (optimistic end of our regime) | +0.0099 |
| +0.83 | +0.0133 |
| +0.74 | +0.0149 |

Our best offline ensemble ever recorded is `peak8_v1` at **0.8731**; `gradeswap_v1` at 0.8694 reads 0.933.
0.950 wants an offline somewhere around **0.883–0.888**. Nothing in this repo is within +0.010 of the best
number it has ever produced. **The goal is roughly twice as far as every plan here has been written to
assume**, and no queued arm — `calibsix`, `margin_calib`, a `blendhold` six-seat — is within a factor of
three of closing it. Every price I computed earlier today was computed at 1.72 and is overstated ~2×.

It also retrodicts two things that were sitting there unexplained. `blendthree_v1` swapped three of six
champion seats for blend-supervised twins and read **0.939, dead flat**. And ten entries now sit inside
**0.936–0.939** across count, member identity, fold status, regularisation, encoder family, architecture
family, vote rule, serving views, corpus, and supervision. Both are what small offline gains look like
when they are multiplied by 0.8 and then rounded to three decimals.

## What it does not change

**The ordering.** 0 of 108 resolved pairs are inverted. The holdout still ranks candidates correctly; it
is only the *scale* that was wrong. Selection is as trustworthy as it was yesterday — which is why this is
a re-pricing and not a crisis about the ruler.

## The fix, and the habit

`board_transfer` now re-reads its own estimator at quantile floors drawn from the record and prints the
fall, with each cell's pair count beside it (the top cell is always the thinnest — 8 pairs, and 0 above
the upper quartile). `--near <offline>` reads the regime an arm would actually be priced in. The
whole-record line now says in its own output that it is the wrong number to price with.

The habit is the one `CLAUDE.md` already records twice — a constant quoted from prose while the instrument
said something else. This is the third instance, and it differs from the first two only in costing more:
the pinned board gap read 0.008 against the API's 0.015, `board_transfer`'s slope was quoted wrong from
prose twice, and now the constant itself was right about a regime we left days ago. The rule is not "quote
the tool" — I *was* quoting the tool. It is that **a summary statistic over a range wider than the decision
is not evidence about the decision**, and a tool that reports one number invites exactly that error. The
tool now refuses to.

Beads: `knee-ps8o` (P0, the finding), `knee-8yi5` (the arm whose price this halves).
