# Public-board overfit: what in our picks was chosen ON the public board (2026-10-15)

The question: the 0.001 tie order keeps us at rank ~148 on the public board today, and a new public notebook
can break 0.957 any day. That is the visible risk. The larger risk is that the private split (~70% of 1,322
studies, ~920) does not reward what the public split (~400) did. Earlier pick work
(`2026-10-10_postprocessing_and_pick_risk.md`, `2026-10-11_pick_pair_in_pile.md`) priced pick risk as SAMPLING
noise only: private = a fresh draw of studies. It never priced SELECTION, meaning anything we or the community
kept because it read higher on those 400 studies.

## Three layers of selection, separated

**1. The tie order inside the 0.957 block: pure public noise, ~0 private value.** Our position ~25 of ~150 at
0.957 comes from hidden decimals inside a 0.001 bin. The paired private sd between our own near-identical
entries is 0.0007-0.003 (`hedge_corr`, 10-10). That is the bin width or more, so the order inside the block
reshuffles on the private board. Every flat 0.957 variant we submitted (soloquarter, flip, solo224, probit)
and then read for its decimals was a public-only fit. Chasing decimals further only adds overfit.

**2. Community selection of the pile (h957 recipe, R7/nartaa weights): real, size unknown, and shared.** The
public recipes climbed 0.937 → 0.957 through dozens of published reblends, each kept because it read higher
on the same ~400 studies. Paired sd between pile variants is ~0.001-0.002. Best-of-~20 selection on that is
worth ~2 sd, so roughly 0.002-0.004 of the climb may not carry to private. Every fork shares it. So it moves
the PILE against independent teams and leaves our rank INSIDE the pile mostly alone. Both medal lines are
inside the 0.956 block today (07:07Z full LB: 1,210 teams >= 0.956; top 5% and top 10% both cut at 0.956).
So the medal question is our private position RELATIVE TO THE PILE, and that is layer 3.

An independent ledger narrows the size (public notebook `hoshinoyumemi/knee-0-939-to-0-956-what-moved-the-score`,
read 2026-10-15 07:33Z, fourteen of that author's submissions, one addition at a time). Of its 0.939 → 0.956 climb,
+0.011 came from two newly trained models (nartaa CoAtNet-384 +0.005, heliosli R7 +0.006), each a single addition
rather than the best of many tries. Those are the part most likely to transfer. The part that was SELECTED is the
per-finding outer weights "published after leaderboard probes by their author", +0.002. So layer 2 is most likely
~0.002 rather than 0.004. It is one author's public-only ledger, so it locates the selected part but does not measure
its private loss.

**3. Our edge over a pure-pile fork: the only increment with off-public evidence.** It is our two soups (1/12 each, 1/6 together, in `s6r7solo`).
Two independent rulers agree on its sign:
- offline, fold-honest, nartaa replica as the pile proxy: two soups at 1/6 each add +0.0015 / +0.0016
  [+0.0005, +0.0026], P>0 1.00 on both replicas (`2026-10-10_postprocessing_and_pick_risk.md`, 21:17Z);
- board: `s6r7solo_v1` 0.957 with soups at 1/12 each against `r7half_v1` 0.956 without them (SUBMISSIONS: r7half). The offline read is at 1/6 each, a larger share, so it is the sign that agrees, not the size. The board pair is not a clean ablation either: R7 also moves from 1/3 to 1/2.
That is the part of our score that was NOT fitted to the public board. Caveats: the replica is weaker than the
real nartaa, so the true increment against the real pile is likely smaller, and the holdout labels lean
toward our own members (`2026-09-05_the-holdout-is-the-distorted-ruler.md`).

## What follows

- **Stop spending slots on pile reweightings judged on decimals.** In expectation they buy nothing on private
  and add selection. A slot is worth spending on a change with off-public evidence, or on an instrument that
  answers a question, never on the next decimal.
- **The repeatable defence against "a new notebook breaks 0.957" is a transform, not a weight hunt:** take
  the new public best and add our two soups at 1/12 each in rank space (as in `s6r7solo`). That increment has the off-public evidence
  above. The binding constraint is Kaggle GPU quota: ~17 min, one R7-class visible run until the 10-17
  reset. That run is worth more held for this transform than spent on any shelf instrument.
- **Picks.** The 10-09 rule (pick 1 = lowest ours share holding the board best) protects against the pile
  HOLDING on private (sampling world). Selection is a second failure world: the pile's public lead is partly
  fitted, and independent strong teams rise past it. Here the pick wants weight on increments measured OFF
  the public board. The owner's pair (`s6r7solo_v1`, `s6r7_v1`) both sit at ours 1/6, and both carry the soup
  increment, so neither world finds them naked. What they lack is any coverage of a bad pile reweight: both
  ride nartaa 1/2 + R7 1/3 from public recipes. A pick 2 with LESS public-tuned structure (fewer reblend
  steps, weights not fitted to the board) would hedge that, at the cost the 10-09 table states for a larger
  ours share. No measurement here prices that trade. It is the owner's call, and this note records that the
  sampling-only analysis did not see it.

## What would change this

A held-out read of the PILE itself. The only fold-honest nartaa read we own is the replica
`raptor_ft_repl2x2_a_swa` (0.8787 @ledger75, against our soup's 0.9089), and it disagrees in direction with
the board (nartaa 0.949 board vs soup 0.945). One of the two rulers is wrong on the foreign family. If it is
the board, the pile's public lead is partly selection, layer 2 is larger than 0.002-0.004, and a more-ours
pick 2 gains value.
