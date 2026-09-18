# State of the levers — what could still carry 1.5%, priced — 2026-09-18T01:50Z

## Why this file exists

Tonight's close (`2026-09-18_the_pool_holds_no_per_finding_specialist.md`) says the next member has to be
BUILT rather than selected, so the next question is which mechanism to build. Answering it took seven
greps and one file read, and **every one of them landed on a number this repo had already recorded.**
That re-derivation has now been paid for at least three times. This is the table, so the fourth time is a
grep of one file.

Card cost of this audit: zero. It is arithmetic over things already on disk.

## The levers, each against the 0.011 board gap to first place

| axis | bound measured here | where |
|---|---|---|
| **pool diversity** — some existing member leads a finding | **0 of 12 columns clear** their own 95% paired interval over the shipping solo | `2026-09-18_the_pool_holds_no_per_finding_specialist.md` |
| **capacity ladder** — DINOv2-base/large over small | **−0.0010**, five isolating pairs median 0.0000 | knee-5dx47; `base_dense` 0.8512 vs `base_honest_batch` 0.8502 |
| **planes** — does the pipeline read coronal/axial at all | **already consumed**: six slots span all three planes | `core/data/corpus/slots.py:58` |
| **slot occupancy** — a finding starved of its primary sequence | **no column's weakness tracks slot presence**; mean 4.57 of 6 slots per study | README:10353; `2026-09-05_the-weak-columns-are-not-a-missing-sequence-and-not-a-combination.md` |
| **distillation** — a teacher above the student | **teacher headroom 0.002**, term itself +0.0056 (under the 0.0062 floor) | README:16585 |
| **in-domain MRI pretraining** — SSL on fastMRI/SKM-TEA | **published wash at our labelled scale** (+0.073 at 80 subjects, −0.005 at 800; we hold ~4,400) | README:7425, Zhang et al. MedIA 2024 |
| **external data as LABELLED augmentation** — the one axis README:7425 left explicitly open | **closed on our own measurement**: `pool_specialise` FLAT, `noisy_student` 0.8548, `--corpus-oversample 2.22` does not lift the mean | README:15216, :12447, :10737 |
| **recipe fragments** — any piece of the champion family | **closed entirely**; sign unanimous across seven leave-one-outs, `backbone_lr` at its optimum both sides | README:16503, knee-rn9py, knee-lm2ol |
| **auxiliary objectives** — the twelve `--*-weight` arms at full budget | **−0.0019** against its own matched converged control, the `<= 0.000` branch | knee-evdfd (read 2026-09-18T02:14Z) |
| **backbone-lr** — the one knob whose corner value was an ENDPOINT, 4/4 upward in the corpus | **−0.0022** at 8e-5 against its own control, converged, both peaking e18 — the sign REVERSES at 24 epochs with dropout | knee-e9f83 (read 2026-09-18T02:30Z) |
| **adaptation depth** — more than six encoder blocks adapting, the segment that RISES +0.0129/block from 4 to 6 | **+0.0007 midpoint / −0.0012 ledger75** at seven blocks against its own control — the gradient does not survive past six | knee-p724l (read 2026-09-18T03:19Z) |
| **readout depth** — which ViT block the head reads, the axis `get_intermediate_layers` exists for | **−0.0227** at block 9 of 12, 3.7x the floor, converged, seat −0.0034; an interior peak at 10/11 would need a sharp non-monotonicity | README:1200. **Spelled `encoder_layer`, not `--layer`** — a grep for the flag name finds nothing and reads as unarmed |
| **through-plane at the dense corner** — each slice's other two channels hold its depth NEIGHBOURS, free by construction | density gradient, not a flat: **12 slices −0.0136\* on Medial Meniscus** (`through224` 0.8505), **20 slices −0.0020** (`through_dense_s2026` 0.8558, seated). The named cause is SPAN LOSS, which shrinks with density | README:4635; knee-4by7q prices 36 slices and BLOCKS on slicecover36's premise |
| **DINOv3 trunk, ours** | best honest member 0.8724 against seats 0.9045 | `2026-09-17_member_candidates.md` |
| labels · cut · head_lr · resolution · overlap TTA | closed previously | prior ledger |

**No unbounded single mechanism remains on available assets.** Every priced axis is bounded below the
gap, most of them below the 0.0062 paired floor.

## The pool's ORACLE is also below the gap

The per-finding close says no single member leads a column. A weaker question is whether the pool leads
*collectively*: take the best member in every one of the twelve columns at once, and grant it every
advantage — the 374-way selection is not deflated, the leaders are read on the studies that chose them,
and combining them is assumed free. That oracle's mean margin over the shipping solo is the mean of the
twelve margins in `2026-09-18_the_pool_holds_no_per_finding_specialist.md`:

```
(0.0016 + 0.0113 + 0.0097 + 0.0057 + 0.0302 + 0.0002 + 0.0125 + 0.0120
 + 0.0124 + 0.0096 + 0.0033 + 0.0062) / 12  =  +0.0096 offline75
```

**+0.0096 is the ceiling of every recombination of all 374 members**, and the board gap is 0.011. So the
gap is not reachable by any reseating, re-weighting or per-column routing of what exists, even with an
oracle doing the routing — and the real figure is smaller, because 0.0302 of that sum is MCL's
selection artefact on the table's widest interval. This is the quantitative form of "the next member has
to be BUILT": not a preference, a bound.

## The composition escape is NOT open by default

Global CLAUDE.md says sub-floor increments still compose. This project has measured it both ways and the
distinction is per pair, not per size:

- **Composes** — the five small fragments of the control recipe sum to +0.0253 and `ctl_stripall` removes
  all five at once for +0.0269, inside the pre-registered band (README:16503).
- **Does not** — `distill_stack_clean` IS `labelwise_drop` plus distillation and reads **−0.0101** at the
  0.75 cut. Distillation subtracts from the pooling stack rather than adding to it (README:15893).

So "compose the surviving sub-floor terms" is a hypothesis owing its own arm each time, never a free sum.
The one composition explicitly still unspent is **+0.0056 distillation onto the SHIPPING recipe**
(README:16599) — and the refutation above is evidence against exactly that pairing.

## And the offline ruler no longer ORDERS members at the top

Three solos have now shipped from the top of the offline ranking. Their offline75 spread is 0.0014 —
far inside the 0.0062 paired floor, i.e. an offline TIE — and their board spread is 0.004:

| member | offline75 | board | entry |
|---|---|---|---|
| `labelwise_ext_drop40_long_s2026` | 0.9045 | 0.937 | `extsolo_v1` |
| `labelwise_depth24_drop40_long_s2026` | 0.9053 | 0.941 | `l75solo_v1` |
| `headlr_ship_s2026` | 0.9059 | 0.937 | `headlrsolo_v1` |

The body line fitted below 0.900 over-predicts all three, 3 of 3 LOW. So above offline75 ~0.904 the
ruler is an upper bound and not a rank: **an arm selected on offline75 up there cannot be priced before
its slot.** The consequence for the schedule is that the board's five slots a day are now the only
instrument with resolution at the top, which is what makes a slot spent on a reseating expensive.

That was an inference from three board points. It is now **measured**, by pointing
`kneemri.evaluation.rank_stability` at the shipping cluster itself — the instrument had only ever been
run on the 0.851-0.863 family (README:12502), where "resolves its top" means the top of *that* band.
Twelve arms, the whole `eb3ba199` population above offline75 0.8998, 909 shared studies, 500 resamples,
ruler ledger75, 2026-09-18T03:47Z:

| | 0.851-0.863 family | the shipping cluster |
|---|---|---|
| score span | 0.012 | **0.0052** — inside the 0.0062 paired floor |
| top arm holds rank 1 | **96.8%** | **29.2%** |
| arms that rank first at least once | 6 of 13 | **11 of 12** |
| Kendall tau, midpoint vs ledger75 | +0.6667 | +0.6061 |

`headlr_ship_s2026` leads by 0.0006 and holds that lead in under a third of resamples; the arm ranked
sixth wins outright 5.6% of the time. **909 studies cannot order the band we ship in.** Two things
follow, and the second is the expensive one:

1. Offline75 cannot choose which solo to ship. Picking the top row is a coin flip with ~29% odds, and
   the 0.004 board spread across the three shipped solos is what that coin flip looks like from the
   board side.
2. **Every keep/kill taken inside this band is not a null — it is no reading.** The three knob closes
   recorded tonight (aux objectives −0.0019, `backbone_lr` −0.0022, adaptation depth ±0.001) all sit
   an order of magnitude inside a span the ruler cannot resolve. They remain correct as *refusals to
   ship* — nothing there earns a slot — but they may not be cited as evidence that those mechanisms do
   nothing. The distinction matters the next time one of those axes is proposed with a stronger
   mechanism: the prior is uninformative, not negative.

The bar this sets for any future corner arm: **a mechanism must be worth enough to clear the band, not
to win inside it.** With the top-12 span at 0.0052 and rank 1 held at 29.2%, an offline-measurable
result at the corner needs to be a step, not an increment — which is the same conclusion the lever
table reaches from the other side, and the reason the remaining two places to look are supervision
quality and slice-to-study aggregation rather than another knob.

## What is actually live

1. `headlr_ship_ff` — the `--fold -1` refit of the shipping solo, FINISHED 02:16Z and unread. Dark arm;
   its reading is a board slot, not a ledger row, and it is priced as an instrument: its honest twin
   `headlr_ship_s2026` shipped at board 0.937, so carrying the refit term off that predicts 0.940-0.945
   against a 0.946 champion. P(beats champion) ~0.10.
3. `slicecover36` (knee-jb0ci) — QUEUED behind a 107 GB corpus decode started 02:32Z, because no
   36-slice cache exists and `run_train` never builds one. Watcher `after_sc36.sh` gates the arm on the
   decode's pid. Honest price ~1.5-2 h decode + 2.4 h train, expected value +0.001 — justified as an
   INSTRUMENT on the aggregation residual below, not as a gain.
2. ~~the auxiliary-objective pair~~ — CANCELLED 2026-09-18T02:14Z. Both had already run on 2026-09-13 and sat
   unpriced; `reading_check --only` refused the re-arm and the pair reads −0.0019, the closing branch
   (knee-evdfd). The card is free behind `headlr_ship_ff`.

## What this says about the schedule

The honest reading is that **offline ranking is exhausted and so is the asset pool**: the ruler ceiling
(offline75 ~0.904) and the per-finding close and this table are three views of one fact. A lever that
could carry 1.5% would have to add information the project does not hold — labelled studies, or a trunk
that has seen MRI at a scale our licences do not reach. The trunk one was closed on premise before the
card was spent; the labelled-data one was closed the harder way, by three arms of our own, and the
caveat README:7425 left open ("not a verdict on external data as LABELLED augmentation") was answered
NO eight days later at README:15216 and sat unjoined to it until now.

What that leaves is not "find the big lever". It is: keep the standing entry protected, spend card only on
arms with a pre-registered branch, and treat any claim that an axis is open as owing a line in this table.

## Where a 1.5% lever can still live, now that the knobs are gone

knee-dry3h priced 22 defaulted leaves and found the corner at its optimum on every one, with a single
exception: `--backbone-lr`, the only leaf whose corner value was an ENDPOINT rather than an interior
point, with 4/4 upward sign agreement across the corpus. That exception is now measured and reads
**−0.0022**. The upward agreement came from 12-epoch 224 arms without slot dropout — knee-dry3h's own
honest limit (2) — and the sign reverses at the corner's 24 epochs with dropout.

So three independent lines now say the same thing: a 4x encoder buys nothing, 22 defaulted knobs are at
their optimum, and the last endpoint reads down. **The residual is downstream of the encoder and
downstream of the optimiser.** Two places remain, and both are named rather than hoped for:

- **supervision quality** — what the labels mean, not how the model is fitted to them;
- **slice-to-study aggregation** — how twelve findings are read out of six slots of slices.

`slicecover36` is the input-side instrument for the second. It is not a candidate ship (expected +0.001);
it decides whether the coverage cell is open at the corner on a harness that cannot be blamed, which is
what the two prior readings — both 12-epoch, both under a max aggregator whose noise floor grows with the
size of the set it maxes over — could not.
