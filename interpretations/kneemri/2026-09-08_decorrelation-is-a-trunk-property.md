# Decorrelation is bought by changing the trunk, not the study cache — and everything that has bought it so far cost more quality than it bought

2026-09-08. Sense-making over D174–D177, the disk reclaim, and two identity defects. No arm was run for
this document; every number in it is a re-reading of members already on disk.

## 1. The bar the pool is now judged against

D174 §3 priced 242 members one at a time into `nine` (base 0.9024 @ledger75, k=9). **Zero of 242 clear
+0.0015.** The best add in the clean pool is `labelwise_dense_s2026` at **+0.0011** (solo 0.8959 @ledger75,
nnCorr@pool251 0.8925). Fitting the adds gives

```
add = -0.0260 + 0.0443*solo - 0.0153*nnCorr     R^2 0.582, residual SD 0.0006
```

so at nnCorr 0.80 a member needs **solo 0.8953** to clear the bar. The sentence that follows from the
surface is the one that should govern arm selection from here: **a new member does not have to be BETTER
than what we have; it has to be AS GOOD while far more decorrelated.** Every knob arm this project has
run moves solo by a few thousandths and leaves nnCorr at 0.89+, which the surface prices at ~0.

## 2. The three-class split (D176), which is the decisive measurement

41 foreign members, 11 of them dark. Split by what the member actually changed:

| class | n | nnCorr median | solo max | best add into `nine` | verdict |
|---|---|---|---|---|---|
| **cache** (foreign STUDIES, our trunk) | 11 | 0.8880 | 0.8928 | **+0.0007** | buys no decorrelation |
| **encoder** (foreign TRUNK, our studies) | 12 | **0.7767** | 0.8505 | +0.0004 | buys decorrelation, cannot pay for it |
| **hisprog** (his pipeline) | 18 | 0.7430 | 0.8984 | +0.0062 | **VOID**, 703/909 = 77.3% in-sample |

Two things fall out and they point the same way.

**A foreign study cache does not decorrelate a member.** `merged3_corpus` prices **-0.0003** and
`merged_corpus` — the SECOND site, the member carrying the +0.0043 solo point that the whole
site-count argument rested on — prices **-0.0005**. So the site-count closure, first measured on solo,
**survives on the seat ruler with the same sign**. Two rulers, one conclusion: adding somebody else's
studies to our trunk produces a member that agrees with our pool.

**A foreign trunk does decorrelate.** nnCorr median 0.7767 against 0.8880 is the largest structural move
any axis has produced. It has simply never been affordable: solo caps at **0.8505** across eight
families, and the surface needs 0.8953.

**The empty corner is occupied only by void members.** hisprog sits at nnCorr 0.7430 / solo 0.8984 —
exactly D174 §3's empty corner, at 4x the bar — and every one of those members is declined VOID at 77.3%
in-sample. `comp/members.declines:794` pre-registered this ratio as *"the shape of the leak"* and said
"twice"; the measurement came back **5.6x** against the honest +0.0011. **A leak signature that predicts
its own magnitude in advance is the strongest evidence this project has that the corner is a leak
artefact and not an opportunity** — which is the only reason the +0.0062 is not the headline of this file.

## 3. What that licenses, and what it does not

Licensed: **the next arm on the diversity axis has to change the TRUNK.** Not the cache, not a knob.
D176's own words: *a trunk this pool has never seen, fitted on our own studies and our own fold — the
corpus is a way to get such a trunk, not the thing that pays.*

Not licensed, and stated as the standing non-claim: **nobody has shown a foreign-pretrained trunk fitted
on our fold reaching solo 0.89.** Nothing in this project has; the highest is 0.8505. So the trunk lane
is where the mechanism is, and it is also the lane with no existence proof.

The one axis nobody has priced on this ruler is **supervision** — what a member is TAUGHT, at fixed trunk
and fixed recipe (distillation, soft labels, rank weighting, ordinal heads, column-weighted losses). The
ledger points here rather than at recipe: `radiodino_corpus` ran at `v3base_dense`'s **exact** recipe
(336 px, 4e-5 backbone, batch 2 / accumulate 3, 12 epochs), converged 12/12, landed 0.8214 and priced
**-0.0008** as a seat. That closes the recipe confound — its ceiling is a trunk property, not a mis-fit —
and leaves supervision as the only card-cheap axis still open. M183 asks for it.

## 4. The licence table, complete across four corpora

the competition rules (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) §2.5a:128-130 exempts incompatible-licence INPUTS by name, and the
competition's own WINNER LICENSE TYPE is CC-BY-NC 4.0 (:66-67), so **NC is defused for all four**. ND is
the clause that decides.

> **RETRACTED 2026-09-11 — the first two rows of this table were wrong, and the first one was wrong in
> the direction that cost the most.** The table below is kept as written because it was cited onward and
> a silently edited table cannot be traced; read it only through this note. See `knee-lsqnm` (P0) and
> `research/deep_dives/2026-09-11_shippable-corpus-licence-survey.md`, which reads every licence as primary text.
>
> - **fastMRI+ does NOT clear.** The "algorithm development named in" allowance is real but it is not a
>   derivatives grant: the NYU terms permit use "for internal research or educational purposes only" and
>   forbid distributing "any subsequent variables or data files derived from the fastMRI Dataset". Training
>   on it is internal research and stays legal; uploading a member fitted on it and running that member in
>   a competition kernel is distributing a derivative and does not.
> - **Rijeka's "already shipped" is not a verdict, it is a description of a mistake.** CC BY-NC-ND is ND on
>   sharing, and shipping a member fitted on it is sharing a derivative.
> - **The blocked artefact is WEIGHTS, not labels**, which this table never considered. So a trunk that
>   sees only foreign PIXELS under a self-supervised objective is blocked by the same clause as a
>   supervised one — `ortho_fastmri` (knee-i3ahx) is an instrument, never a member.
> - **The row that survives is the ND column itself.** ND, not NC, is what decides — that much was right,
>   and §2.5a's exemption of incompatible-licence INPUTS does defuse NC for all four. The error was
>   reading "no NC" and "allowance for algorithm development" as adding up to a permission to redistribute.

| corpus | instrument | ND? | verdict |
|---|---|---|---|
| fastMRI+ | NYU DUA + MIT annotations | **no ND** | **CLEARS** — algorithm development named in |
| Rijeka | CC BY-NC-ND | ND on *sharing* | already shipped |
| MRNet | Stanford RUA cl. 6 | ND on *creating* | blocked |
| SKM-TEA | Stanford University Dataset RUA | ND on *creating* | blocked, same death as MRNet (D175) |

SKM-TEA's clause is quoted from a sibling AIMI instance rather than from our own accepted copy. Closing
that gap is **one login**, with credentials already held; owner action, no card, no slot. *(2026-09-11:
the survey resolves this from the SKM-TEA paper's own Appendix A.5, which carries an explicit
"create derivative works" prohibition — the strongest anti-shipping clause of the four. The login would
confirm, not decide.)*

## 5. What fastMRI+ can and cannot be pre-registered on (D177)

Census over the READ denominator, 1,173 studies (not the 974 that carry a box — 199 were read and found
clean, which is supervised negative, not missing):

- **positive supervision on three columns only**: Effusion n+=142, ACL n+=101, MCL n+=4.
- **MCL's positive class is four studies.** It is not a supervisable column.
- eight columns are coarse-only and receive **zero positives**; `Synovitis` is not asked at all.

So a `canonical + fastMRI` arm **cannot be read per-column on nine of our twelve** — a per-column delta
there measures what 1,051 fresh negatives do to a prior, which is the hazard `fastmri_corpus.py:48-53`
names itself. The readable per-column axes are **ACL and Effusion**. This does not kill the arm; it fixes
what the arm may claim, and it is a second independent reason (after D176 §5) to pre-register it on the
**trunk** axis — does the member land below nnCorr 0.85 at solo >= 0.88 — rather than on a macro delta
averaged over nine columns the corpus can only push toward absence.

`core/data/corpus/slots.py` now carries `FASTMRI_MERGED_SCHEME = "mergedf"`, the only licence-clean merge
destination, with `mergedf_labels.csv` built (5,578 rows, 38 columns, 4,407 canonical + 1,172 fastMRI).
**Its arm was cancelled by evidence 40 minutes after the scheme was built** — §2's `merged_corpus` at
-0.0005 is evidence against its mechanism, not for it. The scheme stays as committed code because it
costs nothing and is the only destination that exists.

## 6. Two identity defects, both now repaired in the tree

**`foreign_dinov3_matched` was not a member.** Its predictions equal `foreign_oof_corpus`'s to float32
rounding (max |delta| **7.689e-06**) — the same member under two names. It was the DINOv3 family's only
claimed-honest comparator, and the +0.0074 LEDGER / +0.0030 HIS reading that declined the family was the
void member's number. `comp/members.declines:871` now records the decline's ground as **provenance**, not
reading.

**`READINGS`' docstring contradicted its own map.** It said `Joint Effusion ` is coarse; the line above
reads `Reading(positive=EFFUSION_TERMS)`, and `EFFUSION_TERMS`' docstring argues the positive case
correctly with the box-size measurement (median 36.7 mm, lower quartile 31 mm over the 142). Nothing that
runs was affected. It would have cost D177's census its **largest positive column**. Repaired.

The transferable shape: **both defects were prose disagreeing with code that was right.** Neither would
be caught by any gate in this repo, and both were caught by somebody counting against the map rather than
reading the paragraph.

## 7. Infrastructure facts learned, worth more than they look

**A `run_train` arm's checkpoint lands in the MODELS ROOT while its log lands in `experiments/`.**
`$MODELS` is `D:/data/volumetric/mri/models/rsna_knee_abnormality/`, so `calib_dense_s2026.pt` (89 MB), its `.holdout.npz`
and its `.metrics.jsonl` are at the root and only `calib_dense_s2026.full.log` is under `experiments/`.
An `ARM DONE` marker with no checkpoint beside the log therefore looks like a dead arm and is not one.
This cost a false alarm; CLAUDE.md's rule (*the success marker is the CHECKPOINT on disk*) is right and
the path it implies is wrong.

**Disk: 45 G -> 225 G free**, by deleting only merges regenerable by concat from source arrays that stay,
after verifying zero live processes referenced them. The 12sl/224px merged caches that most `merged_*`
members were fitted on were deliberately kept, so no existing member became unreproducible.

## 8. `ourtrunk44` — the honest foreign trunk, fitted on our fold, priced

**IT IS NOT THE ARM §3 ASKED FOR, and this section said it was.** `experiments/arms/ourtrunk44.sh` opens
*"Cell D of the 2x2 — HIS 44-slot corpus, HIS recipe, OUR trunk"*: `foreign_run` over his training
script, `--labels labels_llm_soft.parquet` (HIS soft label table), his 44-slot corpus, `--ckpt timm`.
Only the trunk and the fold are ours. It is **honest** — his script holds our fold 0 out at
`knee-mri-training-the-twelve-finding-model.py:393-397` and the arm passes `--fold_file fold0_909.json
--fold 0`, so this is nothing like the 98.6%-in-sample hisprog shape — but a member fitted on his labels
prices his labels' TRANSFER, not our recipe's ceiling. **D178 §4's branch — a trunk fitted on OUR studies
with OUR labels — remains unrun**, and it is the arm this file twice claimed had already run.

`raptor_ft_ourtrunk44` converged (best ep14 0.8342, ep15 0.8322 over 16 epochs).

- **`nnCorr@honest12` 0.7469**, against the roster's own boundary 0.8947. js **0.0415** against 0.0150 —
  nearly 3x the roster's internal divergence. maha 4.760 vs 4.366. The tool calls it **DISTINCT**, the
  only cell that is a seat.
- **`solo@own896` 0.7857 at the midpoint, 0.8173 at ledger75.** An earlier draft of this section quoted
  **0.8440 and called it cut-invariant, on the argument that this member's stored truth is binary. It is
  not binary** — `np.unique` over its truth returns the five graded levels `0.08 0.28 0.68 0.82 0.94`,
  and the roster's returns seven (those five plus a hard 0 and 1). So the recut moves it like everything
  else, and the two numbers it moves between are 0.7857 and 0.8173 — which is, letter for letter, the
  midpoint-vs-ledger75 pair CLAUDE.md's C-rule names as the trap. The 0.8440 belonged to another
  statistic and should never have been written beside a `solo`.

**So the standing non-claim survives, with one more point under it.** D176 said nobody has shown a
foreign-pretrained trunk fitted on our fold reaching solo 0.89; this is the arm that asked, and it lands
0.8440. Decorrelation is real and purchasable exactly where D176 said — the trunk — and it still does not
pay: the surface wants solo 0.8953 at nnCorr 0.80, and this is half a point of solo short of the foreign
family's own 0.8505 cap while being the most decorrelated member the project has produced.

**THE NUMBER IS NOW COMPUTED, AND IT IS ZERO.** Priced into `honest12` — the pool whose twelve members
are named on the board (`comp/board.jsonl:71`, board 0.937) and all twelve of which carry a holdout
record — under `LogitAverage` at ledger75, base recomputed on the same rows:

```
BASE   k=12 @ledger75 on 896 rows   0.9032
+cand  k=13                         0.9033
ADD ourtrunk44                     +0.0000
```

**And the 13-study gap was never an obstruction.** The 896 ids **nest inside** the 909 — `set(cand) -
set(roster)` is **empty** — and the two truth tables are byte-identical on every shared row (max |Δ|
exactly 0.0). So the intersection IS the candidate's own row set, the base needs no re-derivation beyond
subsetting, and `seat_value`'s divergence refusal was protecting against a hazard that this particular
pair does not have. **That refusal cost a day of treating a solved alignment as the project's blocker.**
The tool is right to refuse in general — two row sets that merely overlap cannot be differenced — but it
reports divergence without reporting NESTING, and nesting is the case where the answer is free.

**Against the surface, it is a hit, not a miss.** The plane wants `add = -0.0260 + 0.0443*solo -
0.0153*nnCorr`; at solo 0.8173 and nnCorr 0.7469 it predicts **-0.0012** against an observed **+0.0000**,
a residual of +0.0012 — two SD of the fitted 0.0006, and the same sign and size as the residuals D178
found across all 51 supervision rows. **A surface fitted over nnCorr 0.71-0.98 with a median of 0.9285
predicts, to two SD, a member sitting at the extreme low end of that range.** That is the most
load-bearing thing this file records: the plane extrapolates.

**So the fourth axis closes with the other three.** Cache, supervision and every knob leave nnCorr where
it was; this member moves nnCorr all the way to 0.7469 and STILL adds nothing, because it pays for the
decorrelation in solo — 0.8173 against the 0.8953 the surface demands at nnCorr 0.80. Decorrelation is
purchasable and it has never once been affordable.

**Priced twice, on two pools, by two agents.** D179 priced it into `nine` (base 0.9024 @ledger75, k=9)
at **−0.0010**; this section priced it into `honest12` (base 0.9032, k=12) at **+0.0000**. Two disjoint
base pools, two independently written scripts, one conclusion. The 0.0010 between them is the pools, not
a disagreement.

**And the arm's OWN pre-registration fires the opposite branch to the one this file argued.** Written
before it started: *"if it lands below 0.85: the CORPUS carries it, the trunk term is small."* It landed
at 0.7469, twelve points under the low edge of its predicted 0.87–0.92 band. So on its own pre-registered
reading the decorrelation here is a property of **his corpus**, not of the trunk swap — the reverse of
§2's encoder-class reading, and pre-registered rather than fitted afterward. The same run refutes D35's
seat gate, which predicted **~+0.0033** for a member at that nnCorr against a measured −0.0010. Of the
two models this run tested, **the D174 plane survives and the D35 seat gate does not.**

**Two population labels, stated because this file would otherwise invite the substitution it warns
about.** `nnCorr@honest12` (max over 12 seats) is NOT comparable to D176's encoder-class median
`nnCorr@pool251`, and `solo@own896` is NOT comparable to a `@ledger75` solo over 909. The 0.7469 and the
0.7767 are close enough to look like the same quantity and they are not.

**And `gold_macro@gold58` is not a solo at all.** The run log's headline `BEST GOLD macro-AUC 0.8342` is
measured on **58 studies** — `raptor_gold_ourtrunk44.npz` is `(58, 12)`. At that n, MCL's 0.678 and
Effusion's 0.950 are the same number wearing different error bars. Quote it as `gold58` or not at all.

## 9. calib_dense, priced

`calib_dense_s2026` converged 12/12 and prices as a **TWIN** against `honest12`: nearest **0.0081 bits**,
the closest of any candidate measured on that roster, rank 0.9396 / js 0.0081 / maha 4.011 against the
roster's own 0.8956 / 0.0150 / 4.447. It is not a seat, and it is the cleanest confirmation of §1 there
is: a member built by moving a label-side knob lands *closer* to the pool than the pool is to itself.

## 9. The residual tail is structureless, the trunk term is a PENALTY, and the corner was never refused (D180)

Researcher D180, card-free, 242 priced members, `nine` base 0.9024 @ledger75 on the 896 gold-dropped
frame. The plane refits to its published coefficients to four decimals and is unbiased on its own pool
(residual mean +0.00008 against SD 0.00060), so everything below is read against the fixed surface.

**The positive tail is not a family.** Fifteen distinct lineages in the top twenty, and the only
repetition — `calib_span` x4, `through_dense` x3 — is one member exported at several resolutions, not
several members agreeing. Composition: top20 = 9 no-arm / 9 supervision / 2 recipe, **0 trunk, 0
geometry**; bottom20 = **10 trunk** / 7 no-arm / 2 supervision / 1 geometry.

**The grouping that DOES hold is on the negative side, and it inverts a year of framing.** Alternative
trunks sit ~0.0005 BELOW what their own (solo, nnCorr) buys: non-default trunk n=36 mean residual
-0.00049, best +0.00060; swinbase -0.00058, cnxbase -0.00080, ortholarge -0.00017; default n=141
+0.00019. A trunk swap pays for its decorrelation twice — once in solo, and again in something the
surface does not name. It is a penalty term, not a lever.

**Every recipe knob dies against its own permutation null** (400 draws, p95): epochs +0.0018 vs +0.0083,
slices +0.0001 vs +0.0088, band +0.0000 vs +0.0068, image px +0.0001 vs +0.0070, backbone lr +0.0044 vs
+0.0099. Only trunk survives, and it survives collapsed to ONE degree of freedom (alt vs default,
+0.0695 vs null +0.0081), so it is not a level-count artefact. It weakens but never dies as the two
extreme-leverage members are dropped (+0.069 -> +0.030 -> +0.013).

**M186's caveat is upheld and it changes the edge of the conclusion.** The geometry regressors carry
CV 0.12-0.25 with a 64-96% modal share — 96% of resolved sizes are 224, and only 52 of 242 members set
the flag at all. A null on a regressor with that spread is a fact about the SAMPLE. What can be said is
the size: **the geometry variation the pool does contain moves nnCorr by 0.008 where the corner needs
0.09** (12 vs 20 slices: 0.9211 vs 0.9131 @pool251, solo flat at 0.8778 vs 0.8788; at 336 px the median
nnCorr is HIGHER, 0.9450, n=5). A tenth of the required move, on the axis with the most spread of the
three. It does not touch knee-nas5's within-group pixel collapse, and it says nothing about a geometry
change of KIND — a different plane, a mm-based field of view, true 3D.

**The correction that matters most: "the corner is unreachable" is not something either surface could
ever say.** A full quadratic fits this pool better (R^2 0.78, resid SD 0.00043 against 0.58 / 0.00060)
and diverges from the linear plane by 4x toward the empty corner (+0.0045 vs +0.0176 at solo 0.8959 /
nnCorr 0.60). We hold exactly one out-of-region point to arbitrate them and it prefers the linear one:
`ourtrunk44` measured -0.0010, linear -0.0012 (+0.0002, a third of one SD), quadratic -0.0021 (+0.0011,
~2.5 SD). So the plane stays the instrument and D179 s4 stands — but **both surfaces are extrapolating
out there and what is established is EXHAUSTION, not refusal**: no member built along four axes has
reached the corner. That is a strong argument and it is not the same claim. Both sides had been using
the stronger form.

**The break-even geometry.** Gradient (0.0443, -0.0153), ||g|| 0.0469. The nearest real member to the
+0.0015 line is `merged_labelwise_cf` at euclidean 0.0304, needing **+0.032 of solo OR -0.093 of
nnCorr**. +0.032 solo means solo 0.920, which is 0.024 past the best member this project has ever
produced (`labelwise_dense_s2026`, 0.8959 @ledger75). -0.093 nnCorr means nnCorr@9 0.77 at solo ~0.89, where the
only members under 0.80 nnCorr sit at solo 0.46-0.82 and `ourtrunk44` paid 0.078 of solo to buy 0.7469.
Zero of 242 have a measured add >= +0.0015 (best +0.0011); zero are PREDICTED to under the quadratic
either (max +0.0007).

**So M185's second branch fires: member generation is closed as this project has posed it, in these two
coordinates.** The lever moves to what a member SEES — and per the 0.008-vs-0.09 reading, only to a
change of kind. M187 asks the inventory question that decides whether that is an arm or a code change:
which SEE-axis flag is settable today and has never been set off its default by any committed arm.
