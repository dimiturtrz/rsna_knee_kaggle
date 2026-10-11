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

**09:34Z: `heliossoup_v1` scored 0.955 (ref 57063827).** This is the first entry above the pile. Public rank
went from 328 to 116 of 5732; public rank counts our best submission, so that holds with no further action.

It does not change the pick pair above. Three reasons:

1. **The ruling.** heliosli's S6 is derived from SKM-TEA. Under the 2026-09-15 ruling (foreign-corpus-derived =
   instrument only) it cannot be a final pick. That holds until the owner says otherwise.
2. **What the score proves.** The kernel has an S6 fallback-tolerant cell, and Kaggle serves no hidden log. So
   0.955 does not prove S6 was in the hidden run. `s6third_v1` (S6 at 1/3) is the second point. If S6 is
   live, moving its share should move the score.
3. **Size of the gain.** +0.001 over `twosoupmirror_v1` is half the top-band residual sd (0.002). It is a
   direction, not a measured gain.

What it does say: something in our model set clears the pile on public, and the move was a fourth producer
at 1/6. If the owner relaxes the ruling for S6, the pick table above needs an S6 leg, and there is no offline
replica of S6 to price that leg with. Any pricing would come from the board alone.

## 10:06Z — s6third_v1 scored 0.956: the S6 share curve has two points

`s6third_v1` (ref 57063964; nartaa384 1/2, heliosli S6 1/3, the two soups at 1/12 each) read **0.956**,
public rank 87 of 5735. The curve is now 0.954 at no S6 (`twosoupmirror_v1`), 0.955 at 1/6
(`heliossoup_v1`); this entry is the 1/3 point.

- **S6 ran on the hidden rerun.** If the fallback cell had fired, both S6 entries would have collapsed
  to the soup blend and read about 0.954. Getting one point more for each step up in share is not
  what a fallback produces.
- **The size is still directional.** +0.002 over the pick base is about one top-band residual sd, and
  each step is +0.001, which is the board's own resolution. Two consistent points beat one, but they
  do not make a finding.
- **The pick pair is unchanged under the 2026-09-15 ruling.** S6 is SKM-TEA-derived, so it is an
  instrument and never a pick. If the owner relaxes the ruling, `s6third_v1` becomes the strongest pick
  candidate. The next point on the curve (S6 at 1/2) would be the instrument that says whether the
  curve is still climbing or has peaked.

## 10:56Z — the 09-15 ruling is revoked; the picks change

The owner lifted "foreign-corpus-derived = instrument only" (entry is inference code + model; the fight is
for a medal, not first). `heliossoup_v1` (0.955) and `s6third_v1` (0.956) move from `excluded` to `ensemble`,
and `comp/champion.json` now pins `s6third_v1`. KneeCoT stays banned.

Pick plan: the best S6 entry plus `twosoupmirror_v1` (0.954) as the non-S6 hedge — the two differ only in
whether S6 serves, so the pair covers S6 failing on the private split. Slot 3 goes to `kneemri-s6half`
(S6 1/2), the third point of the share curve, which can only raise the S6 pick or confirm `s6third_v1`.

## 19:32Z — the pile now holds both medal lines

Full public board, 5771 teams. **447 teams are tied at exactly 0.954, ranks 141-587.** At 07:27Z the same
tie held 187. The silver line (top 5%, r288) and the bronze line (top 10%, r577) both fall inside it.
The gold line (10 + 0.2%N, about r21) is 0.961. Ours is 0.956, rank 95; 78 teams are above 0.956.

What this does to the picks:

- **Being at the pile on private means no medal at all.** A pick that finishes level with the pile ranks
  with it, at 141-587 on public. Below it is outside bronze. There is no partial safety in being "near".
- **`twosoupmirror_v1` as the non-S6 hedge is the coin of the table above, and the stakes are now a medal.**
  It covers one case only: S6 failing on the private rerun. In that case `s6third_v1` falls back to the
  soup blend and lands in the pile anyway. So the hedge buys nothing that the S6 pick's own fallback
  does not already give.
- **Pick 2 should therefore be a second entry ABOVE the pile, decorrelated from pick 1.** Not a
  non-S6 entry inside it. `s6seven` (the seven-member ours side at the 1/3 S6 share) is that candidate,
  if it reads at least 0.955 when the 10-12 slot spends.
