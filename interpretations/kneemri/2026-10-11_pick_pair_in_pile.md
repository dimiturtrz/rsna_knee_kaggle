# Final-pick pair at a zero public lead (2026-10-11 07:49Z)

The board moved under the pick table. `twosoupmirror_v1` scores 0.954, which is the public pile's own score
(rank 297/5725, 187 tied, 07:27Z). The 10-10 table (`2026-10-10_postprocessing_and_pick_risk.md`) priced every
candidate at a **+0.003** public lead. At a lead of 0 the question changes.

Same harness as `joint_risk.py`, re-pointed. Fold-0 analogs, n=896, nartaa replicas a and c. The pile is the
replica alone. 400 shared draws, 390 public and 910 private. `pick_pair_now.py`, kept outside the repo.

| Pick 2, with pick 1 `twosoup1/3` | P(it < pile) | corr with pick 1 | P(both < pile) | E[max − p1] |
|---|---|---|---|---|
| itself (pick 1 alone) | 0.47 / 0.48 | 1.00 | 0.47 / 0.48 | 0 |
| `soupseven1/3` (ours7 + ff soup) | 0.47 / 0.48 | 0.96 / 0.95 | **0.42 / 0.43** | 0.00025 |
| `twosoup1/4` | 0.47 / 0.48 | 0.98 / 0.99 | 0.43 / 0.45 | 0.00026 |
| `twosoup1/2` | 0.48 / 0.49 | 0.98 | 0.45 | 0.00043 |
| soup alone at board 0.946 (lead −0.008) | 0.88 / 0.89 | 0.90 | 0.47 / 0.48 | 0.00006 |

**Reading.**

- **The tie is a coin flip.** With no public lead, whether an entry finishes above the pile on private comes down
  to sampling, about 0.47 for every ⅓-share blend. The share no longer sets the risk. It did when we led by +0.003.
- **Pick 2 buys a little.** `soupseven1/3` is the least correlated with pick 1 because it seats a different member
  set. It lowers P(both below the pile) by about 0.05.
- **The OAI-free soup alone buys nothing as a sampling hedge.** It starts 0.008 behind, so it falls below the pile
  in nearly every draw. Its case is the licence ruling only.

**What this harness cannot see.**

- The real pile is nartaa + goodpjw reader + S6. Several of its notebooks tune a per-finding share on the public
  split, and the 10-10 table priced that tilt at about +0.05 of private risk. Our entries carry none of it. So the
  0.47 here is an upper bound on our side of the coin, by the amount of the pile's overfit.
- The draws resample one fold-0 holdout. A shift between the public and private distributions is invisible to them.

Every delta in the table is directional. The pair ranking is the only usable output: **twosoupmirror_v1 +
soupsevenmirror_v1**, if the latter scores 0.954.

**Caveat added 07:58Z: the analog has nartaa as the weak leg.** The section of `2026-10-10_postprocessing_and_pick_risk.md`
dated 00:07Z already found this. The replica standing in for nartaa reads **0.8787@ledger75** (n=896); the soup reads
**0.9089**. On the board it is the other way round: nartaa 0.949, ours 0.945/0.946. So the decorrelation figures above
are measured where the pile is 0.03 weaker than our side. That inflates every "ours-heavy" offset. It is why
`twosoup1/2` shows the largest E[max − p1], while the board read share as flat (mirrorquarter 0.952, mirrorthird 0.953,
mirrorseat 0.953).

What survives this is narrow. Two pick-2 candidates sharing the ⅓ share differ only in which ours members they seat.
The comparison between them is less exposed to the weak-leg bias than any comparison across shares. So
`soupseven1/3` against `twosoup1/3` is the one row here worth carrying, and it is still only directional. The
share rows (`1/4`, `1/2`) are for sign only, and the board has already said their sign is flat.
