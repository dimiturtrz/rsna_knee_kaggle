# Post-processing is spent on the holdout; per-finding share is the one lever left (2026-10-10)

Ruler: the 896 studies shared by our six seated members' holdout records and two honest replicas of
nartaa's recipe (`raptor_ft_repl2x2_{a,c}_swa`), ledger75 cut (midpoint on Medial Meniscus and Baker's).
The replicas read 0.8787 here while the real checkpoint beats our seven on the board (0.949 vs 0.945),
so every number below that involves the foreign side is an ORDERING, never a level.

## What was tried, and what it read

| Method | Fit | Δ macro vs its base |
|---|---|---|
| cross-label stack, ours 12 logits → each label (logistic, C 0.01/0.1/1) | 5×5-fold CV | −0.019 / −0.007 / −0.008 |
| per-label member reweight (6 members) | 5×5-fold CV | −0.002 / −0.001 / −0.003 |
| members + cross-label | 5×5-fold CV | −0.003 / −0.004 / −0.009 |
| seat, cross-label stack over 24 | 5×5-fold CV | −0.011 / −0.004 / −0.009 |
| truth-correlation coupling (score + α·corr·others), α 0.05–0.4 | 5-fold CV | −0.0003 … −0.011, monotone in α |
| aggregation rule over our six: rank / median / max / min / power-mean | none | −0.0003 / 0.0000 / −0.010 / −0.007 / −0.002…−0.007 |
| max(ours, foreign) rank vs ½ rank mean | none | −0.0007 [−0.0036, +0.0023] |
| per-label share of the pair (logistic on two z-scores) | 5×5-fold CV | **+0.006** |

The models already carry the label correlations (truth corr up to 0.38: Effusion–Synovitis,
OA trio, Medial Meniscus–Medial OA) — re-imposing them only adds the fold's noise. The one positive is
per-finding SHARE, which the board is already being asked about (`mirrortilt`, `mirrortiltthird`), and
its size here is a replica's, not the real checkpoint's.

## `mirrorlogit` is a share knob, not a confidence lever

Its docstring reads logit-at-½ over rank-at-½ as +0.004 "cross-head confidence". On the same 896 studies
our averaged logits are 1.3-2.2× wider than a replica's per label (1.87 vs 1.21 overall), so a logit mean
hands ours roughly ⅔ of each study's vote. Rank at ours ⅔ reads **0.9078 / 0.9076** (replicas a / c);
logit at ½ reads **0.9076 / 0.9075**. Scale-free z-logit at ½ gains only +0.0009 over rank. The whole
"gain" is the ruler's monotone preference for ours share, which this ruler has because it understates
nartaa, and which the board does not share (¼, ⅓, ½ read 0.952 / 0.953 / 0.953). Not submitted; voided
as a candidate. Its private risk (0.24) is also the ⅔ row's (0.25), as the identity predicts.

## The public wave holds no new seat (read 06:35Z)

`rabari9999/rsna-knee-sota-0-954`: 0.954 is the notebook's stated TARGET, no score shown; members are
nartaa-384, nartaa-224 (0.30 inside the CoAtNet side) and goodpjw2008's reader — `triseat`'s, which read
0.952 — and the only change from its own `lb-0-95` is the reader share on 8 findings (Lateral OA 0.28→0.50)
and dropping the probit half of the fusion. A per-finding share retune found on the public board.
`heliosli/*` (and `evgendvorkin/rsna-versia-7`, verbatim heliosli): 0.7 nartaa + 0.3 an OrthoFoundation-L /
DINOv3 compact model; its weights sit in `heliosli/rsna-knee-ofl-ssl-weights` (403 to us) and a
`rsna-knee-compact-v1` whose public copy is one README, licence `other`, titled "Private Weights". Not
mountable, not licence-clean.

Re-read 07:28Z, four newer notebooks, still nothing: `jenillangaliya/…dual-stage…lb-0-951` and
`rainycloudss/…apex-grandmaster-stack-v9` are nartaa-384 0.70 + goodpjw2008's ConvNeXt reader 0.30, i.e.
the wave `triseat` already seats; `omn1v3r53/omniverse-sota…` mounts nartaa + `tonylica/…bend-dinov3-0917-repro-assets`
(mattiaangeli's DINOv3 family, closed by `bend_v1`) but its code runs a single-slice ResNet stub with a
hash-of-UID fallback, and its 0.9572 ablation table has no run behind it; `innshrestha/…0-943` is the
older raptor/DINOv2 pile at 0.943.

## Errors

- **Fracture** is our weakest label (0.842; 39 positives) and the only one where even the weaker replica
  beats us (0.870). Every combination that moves weight foreign helps it on the holdout, CIs wide
  (rank ¼-ours +0.028 [0.000, +0.061]). Two public notebooks special-case Fracture the same way
  (`mattiaangeli/bend-the-knee-to-dinov3-ensembled` keeps it from the base family; `prvsiyan/…read-the-report…`
  a "Fracture max" rule). `mirrortiltthird` gives ours ¼ there.
- **Medial Meniscus** false negatives (29 of 475 positives ranked in our bottom 30%) are 24 at soft truth 0.68
  and 6 at 0.82 — the grade just over the cut. Label ambiguity, not a ranking a post-process can fix.

## Private risk by candidate (knee-kl48k)

`P(private below the pile | public lead +0.003)`, 300 bootstraps of 910, both replicas agree to ±0.01:
ours ¼ 0.04 · ⅓ 0.09 · tilt@⅓ 0.14 · ½ 0.18 · tilt@½ 0.21 · logit ½ 0.24 · ⅔ 0.25 · ours alone 0.31.
Spread of the lead across 1.5T/3T/Siemens/Philips/GE is ≤ 0.0045 for every blend — no stratum flips one.
So lower ours share buys private safety monotonically; a per-finding tilt costs ~+0.05 risk over its
centre share.

## Medal geometry (full public board, 5651 teams, read 07:56Z)

Gold (rank 21) 0.960 · silver (rank 282) **0.951** · bronze (rank 565) **0.950**, and the bronze line sits
INSIDE a pile of **522 teams at exactly 0.950** (ranks 338-859) — the public nartaa/reader notebooks. Above
it: 0.951 ×127, 0.952 ×54, 0.953 ×28 (us), 128 teams above 0.953. So the pile's private score is one
number shared by ~500 teams, and a pick that lands below it on private falls past rank ~860 — out of the
medals, not one tier down. `P(private below pile)` IS the medal-loss probability, to first order.

Re-priced at each candidate's ACTUAL public lead over the pile (¼ +0.002, ⅓ +0.003, ½ +0.003): ¼ **0.12**,
⅓ **0.09**, ½ 0.18/0.19, tilt@⅓ 0.14. The quarter's lower variance does not buy back its lost tick;
`mirrorthird` stays pick 1.

Re-read 10:07Z (5658 teams): gold moved to **0.961**; silver still 0.951, bronze 0.950. The 0.950 pile grew
522 → **538** in two hours (ranks 343-880), 0.951 ×129, and the 0.953 band holds 30. So the pile is still
filling from below while the band we sit in barely moves. Against the two tiers that remain in reach: silver
holds as long as private does not drop us under the 0.951 pile (ranks 214-342), and 0.954 alone would move us
from rank ~130 to ~98, which is no change of tier. **0.954 buys private margin, not a medal.** The medal is
protected by the pick's lead over the pile, which is what the private-risk column already prices.

Re-read 18:06Z (5690 teams): gold 0.961 (rank 21), silver 0.951 (rank 284), bronze 0.950 (rank 569). The pile
grew 538 → **608** in eight hours (ranks 362-969), 0.951 ×134, 0.954 ×39; 139 teams sit above 0.953 and we
slipped to rank 153 without a score change. The bronze line now sits a hundred ranks deep inside the pile, so
the pile's private number decides bronze for ~600 teams at once, and the silver line moves with the 0.951 band.
Nothing in this changes the picks; it lengthens the fall below the pile (past rank ~970).

Re-read 20:16Z (5700 teams). The lines have not moved: gold 0.961 (rank 21), silver 0.951 (rank 285, inside
the 0.951 band at ranks 232-371), bronze 0.950 (rank 570, inside the pile). The pile is 608 → **619** in two
hours (ranks 372-990), 0.951 ×140, 0.954 ×42. 142 teams sit above 0.953, and we are rank 157 with no score
change. The fill is mostly at or below our score (pile +11, 0.951 +6, against +3 above us), which is the probit
wave settling. The silver line is 114 ranks
under our band's floor (171), so private shake must drop us past the whole 0.952 band to cost silver. Same
reading as at 18:06Z; nothing in it changes the picks.

## A mirror-family second pick hedges nothing

Correlation of the private-minus-pile delta with `mirrorthird`'s, same 400×910 bootstraps, both replicas
agreeing to ±0.01: ours ¼ **0.99**, ½ **0.99**, tilt@⅓ **0.95**, ours alone **0.94**. Two picks that fail on
the same draws are one pick. The levels here favour ours (the replicas understate nartaa), so the expected
max of a pair is not readable — only the correlation is. Pick 2 is therefore worth a different seat
COMPOSITION if one reaches 0.953 on the board; failing that it is `mirrorseat` (½) as the upside pick, and
the hedge it buys is small.

## The hedge that IS uncorrelated: OAI eligibility (16:00Z)

nartaa's CoAtNet weights were trained on OAI, and the forum has asked three times (747455, 744056, 743416)
whether public OAI-trained weights are eligible, with no host answer. Both current picks and the whole 0.950
pile sit on that checkpoint. If the host rules against it, every one of those entries is voided on the same
draw, so the pile vanishes and the medal lines fall to whatever is left — and the correlation table above
cannot see that event, because it is not a sampling draw. The best OAI-free entry on the board is
`basinsoupff_v1` at 0.946. `kneemri-oaifree` is the first entry built for that event: our seven at ½ plus
goodpjw2008's ConvNeXt reader alone (ImageNet `convnext_tiny`, trained on LLM report labels read from the
competition's own reports, verified from its mount), no nartaa. Pre-registered: ≥ 0.950 makes it pick 2,
0.948-0.949 says the reader is a real second family but pick 2 stays `mirrorseat` unless the host rules,
≤ 0.947 says the hedge sits at basinsoupff level. Its ours share is priced on the honest half of our holdout
by `kneemri-readeroof` (the reader's own fold map reproduced: 562 studies it never trained on, 40 leak-control
rows).

### The reader priced honestly (`kneemri-readeroof`, read 17:46Z)

On the 562 studies outside the reader's training folds (ledger75 cut), the reader alone reads **0.8886**, our
six **0.9106**, the two nartaa replicas 0.8780 / 0.8785. The 40 leak-control rows read the reader at 0.9215,
+0.033 over its honest number, so the reproduced fold map is right. Ours share against the reader:
0 0.8886 · ⅓ 0.9031 · ½ **0.9080** · ⅔ **0.9108** · ¾ **0.9114** · 1 0.9106. Half-half minus ours alone is
−0.0026 [−0.0069, +0.0018]: the reader adds nothing at ½ that this ruler can see, and only a small share of it
helps. Per label the reader trails ours everywhere but Fracture (+0.021) and Synovitis/Baker's (±0.001), the
same Fracture pattern as nartaa's. Pre-registered rule (knee-kl48k 17:07Z): `oaifreeshare` moves to ⅔ when the
argmax lands above ½ — it did, so v2 holds ⅔ (+0.0028 over ½). A tie-break, not a finding: the board has
not reproduced offline share preferences in the mirror family.

Three-way, the reader inside the foreign ⅔ beside a replica lifts **+0.0085** at reader 0.45 of it
(0.8976 → 0.9061, both replicas agree). The real checkpoint already saw this as `triseat`/`trihalf` at 0.952,
flat on the board, so it is recorded as a mechanism, not a slot.

### Pick 2 buys no sampling protection, so it should buy the ruling (18:38Z, before `oaifree` scored)

Same 562 honest rows, 400 paired draws of a 390 public and a 910 private sample, the pile = replica a alone.
Private lead is read as `board lead − e_pub + e_priv`, with one draw shared by every candidate (`joint_risk.py`).
Correlation of the private-minus-pile delta with `mirrorthird`'s: `mirrorseat` 0.98, `oaifreeshare` 0.90,
`oaifree` **0.83**, the reader alone 0.52. `mirrorthird` alone falls below the pile in **0.072** of draws, and
`P(both picks below the pile)` is **0.072 for every second pick tried**: `mirrorseat`, and `oaifree` or
`oaifreeshare` at any board score from 0.947 to 0.950. Whenever `mirrorthird` falls, our shared seven take the
second pick down with it. So **no second pick lowers the medal-loss probability on a sampling draw**. What they
differ in is expected upside, `E[max(pick1, pick2) − pick1]`:

| Pick 2 | 0.950 | 0.949 | 0.948 | 0.947 |
|---|---|---|---|---|
| `mirrorseat` (board 0.953) | 0.00044 | | | |
| `oaifree` | 0.00054 | 0.00033 | 0.00021 | 0.00012 |
| `oaifreeshare` | 0.00045 | 0.00027 | 0.00015 | 0.00007 |

Levers on the OAI-free entry itself (19:06Z, same 562 rows, `oaifree_levers.py`). Adding a soup as a seventh
seat moves ours+reader at ⅔ by +0.0007 (`basinsoup2_k4`, 0.9108 → 0.9115) or +0.0008 (`basinsoup_dmlb_k7`).
Giving the reader ⅔ of Fracture moves it +0.0006 [−0.0009, +0.0023]. All three are knob-sized, so none gets a
slot. The OAI-free side is saturated at what the reader adds.

Every gap here is ≤ 0.0004, which is a tenth of the board's resolution. The only event a second pick can
actually hedge is the one no bootstrap sees: a ruling that voids OAI-trained weights. Amended BEFORE either
score was seen: **pick 2 = the best OAI-free entry if it reads ≥ 0.948** (it gives up ≤ 0.0002 of expected
upside). Below that, the choice between `basinsoupff_v1` and `mirrorseat` is the owner's: insurance against
upside. Caveat: the replicas understate nartaa, so these are orderings, not levels.

**Scored 20:37Z: `oaifree_v1` = 0.944**, the `<= 0.947` branch. That equals every ours-only seven-seat entry
(`foldseven`, `eightseat`, `swapseat`), so the reader's half is worth zero on the board, and it sits under
`basinsoupff_v1` (0.946), our best OAI-free entry. The table above priced `oaifree` as a pick only from 0.947
upward, which the board has now ruled out. Pick 2 is the owner's call: `basinsoupff_v1` insures against an OAI
void at 0.007 of public score, and `mirrorseat` keeps the upside. The readeroof +0.0104 reader lead was a lead
over a replica pile, not over ours: the half-half was already −0.0026 under ours offline. The board reads that
flat, not negative.

Threads re-read 20:06Z (747455, 744056, 743416), still no host ruling. The precedent participants quote is
the host's Aug 27 guidance in 743416: "straightforward registration or click-through data-use agreements are
generally acceptable; institution-specific approvals are not" (KneeCoT was barred under it). OAI through the
NIMH Data Archive is a login.gov account plus click-through terms, so the precedent points toward allowed. One
commenter also attached a reply from the NDA itself about OAI's intended accessibility, though its text is not
in the thread dump. A void of OAI-trained weights therefore looks less likely than an even bet. That shrinks
the value of the insurance but does not remove it, so the rule above stands. The commenters' read that only
prize (top-10) solutions get audited cuts the same way.

### The "0.954" probit wave (read 18:17Z)

`rabari9999/rsna-knee-lb-0-954-rank-127-silver` is byte-identical to `jcgweb/rsna-knee-probit954`. Mounts:
nartaa's SWA weights and goodpjw2008's reader only — `trisota`'s members. Fusion is per-leg rank → `ndtri`,
then z-blends nartaa-384 with nartaa-224 crop (`C224_WEIGHTS`, 0.02-0.22 per finding) and the reader
(`CNXT_WEIGHTS`, Lateral OA 0.52, Baker's 0.40, Contusion 0.26, the rest ≤ 0.10), then `ndtr`. The board
partly backs the title: rabari9999's team (Netram Faran) reads 0.954 at rank 136, but jcgweb's own team reads
0.950 with the same notebook, and so does sameerk's (d5l-probit). So 0.954 is one team's best, which may come
from a different version; the notebook alone is not shown above the pile. Nothing new to seat: ours over the
reader-wave already read 0.952 three times (`triseat`, `trihalf`, `trisota`), so a probit wave plus ours is
priced flat and gets no slot.

The combiner priced on its own (20:27Z, `probit_fusion.py`, 562 honest rows, 300 paired draws). Swapping our
rank-average for probit z-space blending (ndtri of the mid-rank percentile) moves the macro by **+0.0002 to
+0.0008**. That holds for `mirrorthird` (1/3), `mirrorseat` (1/2), `oaifree` (1/2) and `oaifreeshare` (2/3),
on both nartaa replicas. The best case is `mirrorthird` on replica c at +0.0008 [+0.0002, +0.0014]. Every
delta is under the board's 0.001 resolution, so it is knob-sized and gets no slot. The fusion is not where
the wave's 0.954 comes from.

Public notebooks re-read 15:30Z: `romantamrazov/…dinosaur-v5` and `kozykappa/…gold-gated-triple-reader` are
nartaa + the same reader at the 0.950 pile. Nothing new.

**Soup as nartaa's partner, priced 20:50-21:06Z (fold-0 analog, nartaa replicas a/c, n=896).** The ours third of
the mirrorthird blend split as `a*ours6 + (1-a)*basinsoup_k4` in rank space, nartaa replica at 2/3:
a=0 (soup only) +0.0011 [-0.0002, +0.0025]; a=1/4 +0.0011; a=1/2 +0.0010 [+0.0003, +0.0017]; a=3/4 +0.0006, all
against a=1 (ours only), identical on both replicas. A PLATEAU from a=0 to a=1/2, knob-sized, below the 0.002
top-band residual sd: directional only. Two kernels built to read its ends on the board: `kneemri-soupmirror`
(a=0, slot 5 of 10-10) and `kneemri-soupsevenmirror` (a=1/2, for 10-11). No interior share points are worth
a slot: the board's 0.001 resolution cannot split a plateau that is 0.0001 wide.

**Two soups as the partner, 21:17Z (same rows, same replicas).** `basinsoup_k4 + basinsoup_dmlb_k7` at 1/6 each:
+0.0015 / +0.0016 [+0.0005, +0.0026] over ours6, P>0 1.00 on both replicas; ours6 + both soups at 1/9 each
+0.0012 / +0.0014. Best of seven compositions read on these 896 rows, so mildly selected, and its margin over the
soup alone (+0.0004) is noise. Kernel `kneemri-twosoupmirror`. A caveat on the board side of the argument:
`soup16_v1` serves the SAME soup checkpoint and boarded 0.944, so the soup's 0.946 against the seven's 0.944 is
inside serving noise; the offline analog, not the board solo, is the evidence for this line.

**Pick 2 priced on sampling alone (21:27Z).** The final score is the better of the two picks, so pick 2's worth
over pick 1 alone is `E[max(0, d)]`, with `d ~ N(board gap, sd(cand - mirrorthird))` and sd taken from the
`hedge_corr` paired draws. `mirrorseat` (tie, sd 0.0007) is worth +0.0003, and a soup-side mirror that ties
(sd about 0.0010) is worth +0.0004. `basinsoupff_v1` (-0.007, sd 0.0024-0.0034) is worth 0.0000 and wins the
private draw 0.2-2% of the time. All three are under the board's 0.001 resolution, so **sampling does not
decide pick 2. Eligibility does.** `basinsoupff_v1` matters in only one world: OAI-trained weights are ruled
out. That ruling also removes every nartaa-based team in the 0.950 pile, so the bronze line falls well below
0.950 and a 0.946 OAI-free entry can medal. In every other world `basinsoupff_v1` is worth zero. The owner's
question is therefore one number, P(OAI ruling), and not a score comparison. A side effect for tonight's
instrument: if `soupmirror` reads 0.953 it replaces `mirrorseat` as the OAI-exposed pick-2 option. The edge is
free but noise-sized.
