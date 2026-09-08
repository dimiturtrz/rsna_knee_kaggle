# Decorrelation is bought by changing the trunk, not the study cache — and everything that has bought it so far cost more quality than it bought

2026-09-08. Sense-making over D174–D177, the disk reclaim, and two identity defects. No arm was run for
this document; every number in it is a re-reading of members already on disk.

## 1. The bar the pool is now judged against

D174 §3 priced 242 members one at a time into `nine` (base 0.9024 @ledger75, k=9). **Zero of 242 clear
+0.0015.** The best add in the clean pool is `labelwise_dense_s2026` at **+0.0011** (solo 0.8959,
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

`comp/COMPETITION_RULES.txt` §2.5a:128-130 exempts incompatible-licence INPUTS by name, and the
competition's own WINNER LICENSE TYPE is CC-BY-NC 4.0 (:66-67), so **NC is defused for all four**. ND is
the clause that decides.

| corpus | instrument | ND? | verdict |
|---|---|---|---|
| fastMRI+ | NYU DUA + MIT annotations | **no ND** | **CLEARS** — algorithm development named in |
| Rijeka | CC BY-NC-ND | ND on *sharing* | already shipped |
| MRNet | Stanford RUA cl. 6 | ND on *creating* | blocked |
| SKM-TEA | Stanford University Dataset RUA | ND on *creating* | blocked, same death as MRNet (D175) |

SKM-TEA's clause is quoted from a sibling AIMI instance rather than from our own accepted copy. Closing
that gap is **one login**, with credentials already held; owner action, no card, no slot.

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
`$MODELS` is `D:/models/rsna_knee_abnormality/`, so `calib_dense_s2026.pt` (89 MB), its `.holdout.npz`
and its `.metrics.jsonl` are at the root and only `calib_dense_s2026.full.log` is under `experiments/`.
An `ARM DONE` marker with no checkpoint beside the log therefore looks like a dead arm and is not one.
This cost a false alarm; CLAUDE.md's rule (*the success marker is the CHECKPOINT on disk*) is right and
the path it implies is wrong.

**Disk: 45 G -> 225 G free**, by deleting only merges regenerable by concat from source arrays that stay,
after verifying zero live processes referenced them. The 12sl/224px merged caches that most `merged_*`
members were fitted on were deliberately kept, so no existing member became unreproducible.

## 8. calib_dense, priced

`calib_dense_s2026` converged 12/12 and prices as a **TWIN** against `honest12`: nearest **0.0081 bits**,
the closest of any candidate measured on that roster, rank 0.9396 / js 0.0081 / maha 4.011 against the
roster's own 0.8956 / 0.0150 / 4.447. It is not a seat, and it is the cleanest confirmation of §1 there
is: a member built by moving a label-side knob lands *closer* to the pool than the pool is to itself.
