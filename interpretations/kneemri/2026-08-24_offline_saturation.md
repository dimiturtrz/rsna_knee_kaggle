# The offline holdout is saturated at ~0.872, and that makes the slope the only question left

2026-08-24. Written while `base_dense_ff` trains, before the instrument entry is fired, so the
prediction below is on the record ahead of the reading.

## What was being looked for

The board sits at 0.934 and the goal is 0.950. The gap is 0.016. The search was for a lever that
could plausibly deliver it — per the house rule, picked by whether the MECHANISM could deliver 1.5%,
not by whether it might help.

Per-label decomposition of the greedy-optimal seven (909 fold-0 studies, each label worth 0.0833):

| label | AUC | n | headroom to 0.90 |
|---|---|---|---|
| MCL | 0.8102 | 171 | +0.0075 |
| Fracture | 0.8229 | 68 | +0.0064 |
| Lateral OA | 0.8310 | 233 | +0.0058 |
| Effusion | 0.8664 | 558 | +0.0028 |
| Lateral Meniscus | 0.8690 | 226 | +0.0026 |
| PF OA | 0.8714 | 409 | +0.0024 |
| … | | | |
| Baker's | 0.9171 | 217 | −0.0014 |

Three labels hold **+0.0197** between them — more than the whole deficit. MCL at n=171 is 19%
prevalence, so it is not a rarity problem; 171 positives is plenty. MCL is a thin band on the medial
side and Lateral OA is one compartment, so both are spatially localized, and a rank average over a
whole stack dilutes a finding that lives on a few slices on one side.

That argument produced a concrete design: a per-diagnosis query over SLICES. `SlotHead` already gives
each diagnosis its own query over acquisition SLOTS with an anatomy prior, but slice pooling below it
uses a single shared query and is label-agnostic. So a query may choose the sequence and not the slice.

## Why it was not run

The ledger had already closed it, from two directions, and `SliceAttention`'s own docstring had
already named it: *"A per-diagnosis query over slices is the obvious next step and is deliberately not
taken until a shared one earns it."*

It did not earn it. The slice-aggregation stage is measured saturated from three directions at once:

- more inputs to it: 20 cached slices vs 12 under max, **−0.0001**
- more slots: nine slots, 83,297 unseen slices, **+0.0002**
- every combining rule: max 0.8457, order-aware gate 0.8461, mean 0.8473, trained 64-wide gate 0.8476
  — four operators spanning **0.0019**, under a third of the 0.0062 noise floor

And `aprior_dense` refutes the premise underneath the design. Initialising `SlotHead.prior` from the
plane-by-finding anatomy table read −0.0021, with train loss tracking the control to the fourth decimal
at every epoch: the head had ALREADY recovered the mapping from study-level labels alone. So
"supervision never says where" is false about this head, and localization must buy a real localization
signal rather than a better allocator. A per-diagnosis slice query is one more allocator on a stage
saturated four ways. Pre-registered null; not run.

## Everything else is closed too

- **decode/head** — four consecutive knobs: −0.0009, −0.0044, +0.0014, −0.0021
- **SSL as init** — five collapses. `mae_matched`, the only encoder that has seen nothing but this
  domain (402,600 knee-MRI slices, 8 epochs masked reconstruction), reads 0.6735 against `adapt_null`'s
  0.8240 through the same `--encoder-weights` path. Bead `knee-eoiq` was still open at P1 against this;
  closed as refuted 2026-08-24.
- **backbone size — INVERTED.** `v3base_dense` 0.8455 and `dinov3_base_lora` 0.8268 both sit BELOW
  small-based `band_dense` 0.8578. Capacity is not the binding constraint.
- **resolution** — `res392` variants 0.8459–0.8523, flat to below.
- **slice order** — `depth_true` 0.8461, second-worst of four operators, so the extra structure is
  slightly costly rather than merely flat.
- **ensemble width** — greedy peaks 0.8719 at 9–11 members against the champion seven's 0.8716.
  **+0.0003.** Width is saturated as well, which matters because width was the last axis that did not
  require a new mechanism.

Offline lands at ~0.872 whichever direction it is pushed.

## The consequence: the slope is the gating question

The decisive unknown is no longer which lever to pull. It is what an offline point is WORTH.

`board_transfer` records a median of **+4.67 board points per holdout point**. But every board reading
we own sits in a 0.0043 offline band — six pairs spanning 0.8648–0.8691 offline against 0.927–0.934
board — and 0.0043 is INSIDE the 0.0062 holdout noise floor. A slope fitted there is fitted to noise
and cannot be falsified by the data that produced it.

The two readings imply opposite plans:

- **slope really ~4.67** — board 0.950 needs offline ~0.875. Nothing in this saturated space reaches
  it, and the entire member-search axis is dead by arithmetic. Budget must go to a mechanism that moves
  offline by 0.003+, and after the audit below **exactly one candidate remains unspent**: a 3D encoder
  that sees the volume before any per-slice feature exists (named untouched by `depth_true`'s row).
  External data is blocked (`knee-ombx`, MRNet RUA is non-commercial against a $77k purse). The reader
  grade/certainty split, listed here as the second candidate when this was written, **is closed** — see
  below.
- **slope really ~1** — the holdout is simply a poor guide above 0.868, board gains live where it
  cannot see, and offline deltas we have been dismissing as noise are UNDER-priced rather than over.

Separating them costs zero GPU, which is the whole point: it is a question about the ruler, and rulers
are read with submissions, not with training.

## The instrument, and its prediction

`kneemri-weakseven` names the seven packaged members with the weakest solo holdout AUC (0.8215–0.8462):
joint **0.8562**. That is 0.0086 below the chain floor — twice its total range, three times its span —
and it is the furthest a point can sit while every member stays rebuildable from `comp/members.roster`.
Count (7) and vote (`RankAverage`) are held fixed so only member QUALITY moves; a one- or two-member
entry reaches further offline (0.8512–0.8636) but changes the vote WIDTH, which confounds quality with
ensemble size and makes the point unreadable against the chain.

Pre-registration, for the offline gap of −0.0086:

| board | reading |
|---|---|
| 0.885–0.890 | slope 4.67 holds out of band; offline is a real ruler; chain flatness is saturation near the top |
| 0.905–0.920 | slope in the large is ~1–2; the steep local slope was six points inside the noise floor; `board_transfer` must be refitted or retired |
| above 0.925 | the board is nearly indifferent to member quality at this width — rank averaging does the work, not the members, and member search closes entirely |
| below 0.880 | the map is STEEPER out of band, so offline gains are under-priced, which licenses spending slots on small offline wins currently dismissed as noise |

There is no outcome on which this entry becomes a champion, and none on which it teaches nothing. That
is what makes it worth a slot under `comp/champion.json`'s own rule — a submission is a verification
of a path, not a measurement of a method, and a slot that expires is the reading we did not take.

FIRED 05:55Z, submission 55734428, slot 1 of 5.

## Addendum, same morning: the reader branch closes, which leaves one mechanism

`knee-mxj` was the second of the three unspent candidates above, priced at one LLM pass over 4,407
reports, on the argument that the reader emits ONE scalar carrying both severity and certainty and
should be made to emit two. Read rather than grepped, `api_labeler.py` **already emits two**: its schema
requires `{verdict, severity}`, severity is an integer 1–3 with its own vocabulary, `_verdicts()` zeroes
it whenever the verdict is not YES, and `to_scores` maps the pair to the value. That is where the three
YES levels come from, and `options.py:339` names them in the repo's own words — *"0.378 mild / 0.688
moderate / 0.958 severe"*. The graded value IS the severity field; `__conf` IS the certainty field. The
split the bead was written to buy is the split that already shipped.

What is genuinely degenerate is `__conf`, constant per verdict — and that axis was closed on 2026-08-21
when both cheap derivations of a better weight read BACKWARDS on real data (graded −0.146 over 266
positive calls, hedged +0.052 over 34), with one mechanism explaining both: the reader has already read
the hedge, so a regex over the same sentence re-derives its own output worse.

The consumer side is spent three ways against a single control (`experiments/README.md:1315`): target
VALUE `calib_span` −0.0020, loss WEIGHT `severity_weight` −0.0048, target SHAPE `grade_ladder` −0.0013.
`severity_flat`, which deletes the gradation, reads −0.0044 — so the gradation is live signal AND every
way of leaning harder on it is negative.

A population check ran first and pointed the same way. Counting the three YES levels per label against
`ordinal_signal` (baseline mean 0.741, `band_dense_long_s2026`) tested whether the three weakest-AUC
labels rank weakest on severity because their grade axis is too thin to measure. It is not thinness:
Effusion has the second-thinnest severe tail (5.9% of 2666 graded) and the BEST severity ranking
(0.859), while PF OA has the fattest (39.7% of 1991) and the second-WORST (0.676). Lateral OA is
likewise well-populated (23.6% of 1087) at 0.681. Fracture's 0.606 rests on 314 graded rows and must not
be read at all. So for the OA pair the axis is present, well-populated, already the target value, and
four arms deep — the model simply does not follow it. That is a statement about imaging capacity, not
about supervision, which is the same place `attend_dense`/`focal_dense` left the decode family.

One correction the close forced: the gate `knee-mxj` specified cannot be run as written. `ordinal_signal`
excludes annotated rows BY DESIGN (`GRADED = (0.5, 1.0)`, exclusive, because `Targets.load` overwrites
annotated studies with hard 0.0/1.0), so "re-run it against the 58 image-read annotations" asks the tool
for exactly the rows it drops. Any future severity gate scores on the graded non-annotated band.

**Consequence for the plan.** The two branches above are unchanged in kind but sharper in stakes. If the
weakseven reading says the slope really is ~4.67, member search is dead by arithmetic and the 3D encoder
is not one option among three — it is the only unspent mechanism left on the offline side. If the slope
is ~1, the holdout stops being the instrument that decides anything above 0.868 and the budget moves to
board-read levers. Either way the reading, not another arm, is what picks.
