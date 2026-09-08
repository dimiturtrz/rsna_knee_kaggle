# Supervision, not architecture, sets a member's correlation with the pool

2026-09-08, from the night of 09-07/09-08. Four readings, all offline, no card and no slot spent on any of
them. Two closed axes, one refuted hypothesis (mine), one ruling I lost and one arm killed before it ran.

## The member

`hisrecipe_distill` is `coatnet_rmlp_2_rw_384` at 384 px, 16 epochs, distilled — a trunk family the pool has
never held, trained by the foreign lane on **our** teacher (`experiments/arms/hisrecipe_distill.sh:63`,
`teacher_clean_full.npz`). It carries no released foreign checkpoint, so unlike `raptor_v4_swa` it is
licence-clean. That combination is what the roster has wanted since the contamination finding: foreign-family
diversity without a foreign checkpoint.

Its OOF was converted to a `HoldoutRecord` with `kneemri.evaluation.foreign_holdout`, which joins on the study
IDENTIFIER and scores against `report_labels_v2.csv` — his soft-label truth table is discarded deliberately.
896 studies against our 909; the two columns are therefore not literally paired and are not written here as if
they were.

## 1. Solo: inside our band, at the bottom of it. The fleet-wide architecture axis is closed.

`ledger75`, both rulers:

| member | LEDGER (midpoint) | LEDGER75 | rows |
|---|---|---|---|
| `stackfit_s2026` (our best solo) | 0.8631 | 0.8962 | 909 |
| `small_batch6_s2026` | 0.8598 | 0.8895 | 909 |
| `band_dense_s2026` | 0.8578 | 0.8877 | 909 |
| `hisrecipe_distill` epoch-ensembled | 0.8498 | 0.8880 | 896 |
| `hisrecipe_distill` SWA | 0.8496 | 0.8880 | 896 |
| `hisrecipe_distill` final (ep15) | 0.8495 | 0.8875 | 896 |
| `hisrecipe_distill` gold-selected | 0.8488 | 0.8865 | 896 |
| `hisrecipe_distill` ep10 | 0.8480 | 0.8870 | 896 |

−0.0080 against `band_dense` and −0.0133 against `stackfit_s2026`, both outside the Hanley-McNeil floor
(0.0053) and the three-seed floor (0.0062). **Retraining the fleet on a foreign trunk is not a lever.**

**A second result falls out of the same five checkpoints, at no cost.** Epoch selection, SWA and
epoch-ensembling together span **0.0018** — a third of the floor. The gold-58 trace climbing through ep14 to
0.9198 did not translate into an 896-row gain. The epoch-budget question on his family is answered without
running it: 58 rows cannot resolve what 896 can, and this is the second time in one night that the two
disagreed in a way that would have cost a card night.

## 2. Correlation: he is MORE correlated with our pool than our own members are with each other

`seat_swap`, five champion parents, frame 896, gold rows dropped, published rank transform. Measured
independently by both lanes on different harnesses:

- **parents' own mean pairwise corr: 0.871** (reproduces D87's 0.870 — the frame checks out)
- **`hisrecipe_distill` against the rest: 0.890–0.896** (this lane), **0.894** (the other lane)
- **`raptor_v4_swa` against the parents: 0.801** (D87)

A trunk family the pool has never held is **0.02 more correlated with us than we are with each other**, and
0.09 more correlated than the foreign member it was proposed to replace.

> **The teacher sets the correlation, not the trunk.** Distilling any architecture on our teacher reproduces
> our error structure inside it. The architecture is a container; the supervision is the content.

This is a hypothesis, not a measured law — the measurement that would settle it is a within-group /
between-group correlation split over the pool tagged by **supervision** (teacher, different teacher, no
teacher, different truth table) rather than by trunk family, with `raptor_v4_swa`'s 0.801 as the out-of-group
anchor. Filed as a job, not claimed here.

It also explains, if true, why the diversity axis measured closed: every member in the pool shares a teacher
or a label table, so the pool has a floor on how decorrelated it can get, and the closure is a statement about
our supervision rather than about diversity.

**A LEDGER75 tie is not a decorrelation signature.** His gap to `band_dense` collapses from −0.0080 to +0.0003
under the 0.75 recut (his family gains +0.0382 from the recut against our +0.0299). That looked like "these two
members are not making the same errors" and it was read that way for about ten minutes. The correlation says
the opposite. A ruler effect and a decorrelation signature are different things and only one of them is
measurable by correlation.

## 3. The seat: no seat, and a proxy number that must not be quoted

Champion proxy, five parents at 2/3 rank-mean, one anchor at 1/3:

| anchor at 1/3 | champion proxy | five alone | the seat is worth |
|---|---|---|---|
| `raptor_v4_swa` | 0.9238 | 0.8997 | +0.0241 |
| `hisrecipe_distill` | 0.9017 | 0.8996 | +0.0020 |

As an eighth seat rather than a substitute, `seat_swap` against each of our five reads −0.0008 / −0.0003 /
+0.0002 / −0.0002 / −0.0004 on the proxy. **0 of 5 pass the gate.** The substitution costs **−0.0221**, so
the "zero-cost licence-gate closure" is not available.

**The +0.0241 is a measurement of the proxy's deficiency, not of the raptor seat, and it is recorded here so
it is not quoted later as a fact about the seat.** Five of the six champion members are `--fold -1` and left
no holdout record and never will, so four are stood in for by weaker `_s2026` siblings. A pool of same-recipe
siblings is more correlated internally than the real champion six, which inflates whatever distinct thing is
added to it — and `raptor_v4_swa`, the most distinct member we own, absorbs that inflation maximally. Against
it, D89 measured the same seat at **0.000 on the board across eight entries** (7-with-seat 0.944 = 6-all-ours
0.944), and +0.0241 offline would be ~+0.034 on the board at the 1.4 slope, three times our whole distance to
the leader, from one seat at one-third weight. The board is the ruler that pays. **The licence gate stays
open; the substitution is settleable only as a submission.**

Both lanes built this frame independently and both hit the same wall. That is a property of the champion, not
of either script.

## 4. The instrument was broken before any of this could be measured

`seat_swap` accepted `--anchor` and then ignored it: `proxy()` read the module constant `ANCHOR`. Every
substituted-anchor reading died on `KeyError: 'raptor_v4_swa'`, and an anchor that happened to also be a
seated parent would have **silently scored the default and reported it as the substitution**. A value recorded
in a flag while the consumer reads a constant — the plumbing class exactly. Fixed in `3be0494b`
(`knee-mtwmy`); `anchor` is now a `SeatFrame` field threaded through `proxy` and the ruler-move message.

## 5. An arm killed before the card: there are no slices to reallocate

A candidate arm was proposed and refined twice on the reading that `augmentation.py:74`'s *"fill rates are
100% for the axial fluid-weighted slot but 55-60% for sagittal non-fat-sat and both T1 slots"* meant ~8 slices
of padding per T1 slot, free to move to coronal fluid at zero VRAM.

The quote is exact. The reading is wrong, and the next sentence of the same docstring says so: those fills are
**study-level slot PRESENCE**, and `SlotScheme.presence` **masks** an absent slot rather than zero-filling it.
A 57% fill means 43% of studies do not have that sequence at all; for the 57% that do, all 20 slices are real
image. Depth is uniform — `images_336px_12sl.npy` canonically, `images_336px_20sl_0.1-0.9band_merged.npy` in
the band family — across **six** slots (`core/data/corpus/slots.py:61-66`), not five. There is no 64-slice
budget and no unequal split to rebalance.

Cost of the kill: one grep. Cost had it run: 1.4 h of card on a trade that does not exist.

**The same docstring hands over something real, reframed.** `--slot-dropout` is fully plumbed
(`options.py:81` → `run_setup.py:192` → `augmentation.py:69,226-230`) and stamped into the member record, and
**no member on disk has ever set it** (default 0.0). What it addresses is a train/serve distribution shift on
~43% of studies — absence was never in the training distribution — which is the same class as the only two
mechanisms that have ever paid here (whole-stack +0.0147, overlap TTA +0.0146: "the model reads more of the
study"), at the other end ("the model stays correct when there is less of the study").

*(Corrected 2026-09-08: this line first read "whole-stack +0.0185". The ledger carries whole-stack at two
sizes against two different controls — **+0.0147 against its own matched control** (`README:294`) and
**+0.0185 against `rank24_s2026`** (`README:829`), the ORIGINAL baseline the running-total column is
measured from. Quoting the running-total figure as the arm's size inflates it by a third. The matched
number is the one that belongs in a sentence about a mechanism, and even it is read DOWN by `README:874`:
the +0.0147 control was scored without the overlap TTA it gets at inference, and given both their own best
reading `lr5x_d6` (0.8475) edges `wsfold0_e6` (0.8457). So the two mechanisms named here are not additive
and the pair is the same lift bought twice — which is the whole reason `knee-eeo` says whole-stack members
gain nothing from overlap TTA.)*

Pitching it as a **diversity** arm prices it at **+0.0006** by the measured seat-value regression
(`uniform 1/9 gain = -0.0424 - 0.0291·corr + 0.0765·solo`; a corr move of 0.871 → 0.85 at flat solo), which is
an eighth of the floor. Pitching it as **subpopulation repair** prices it at `gap × incomplete_fraction` — at
43%, a 0.03 stratum gap is **+0.013 macro**. Same arm, same 1.4 h, and the framing decides whether it is a
pre-registered null or the largest solo lever on the board.

**So it is gated on a CPU measurement, not run on an argument:** macro-AUC on the complete stratum vs the
incomplete one, per column, stratified by which slot is absent. Gap ≈ 0 kills the arm for free and closes the
axis on a measurement instead of a precedent. Two hazards were flagged before it started: there is **no cached
presence table** (`presence` takes the routing decision, not the image cache, and nothing on disk holds one),
and **Rijeka is one slot by construction**, so a split taken over the merged corpus puts every Rijeka study in
the incomplete stratum missing five slots and returns a large, plausible gap that is a statement about a
foreign corpus. Denominator: `n_incomplete / n_ours`.

## What this night closed

- **Fleet-wide architecture change**: closed on a measurement (−0.0080 to −0.0133 solo).
- **Epoch budget / SWA / epoch-ensembling on the foreign family**: closed at 0.0018 total span.
- **`hisrecipe_distill` as a seat, at 1/3 or uniform**: closed (0 of 5, substitution −0.0221).
- **Decorrelation via a foreign trunk**: refuted — the trunk was foreign and the correlation went *up*.
- **Depth-by-fill reallocation**: does not exist; presence is not padding.
- **Still open, and now the two live levers**: depth past the batch-12 ceiling (`deep32_batch6`, armed,
  DeviceBudget derives 22.25 GB at 32 slices / batch 6 against ~31.8 usable), and slot-dropout as
  subpopulation repair, gated on the stratum gap.
