# Thirteen foreign checkpoints are four members and nine copies

*2026-08-28 · `python -m kneemri.evaluation.teacher_distance --teachers <teacher_fleet>` over 400 shared
studies · beads knee-v6ox, knee-nj4l*

## What was measured, and why it cost nothing

Every one of the thirteen public raptor checkpoints was fitted on all 4,406 studies with `--fold -1`, so
their QUALITY is unmeasurable here — a holdout AUC over our corpus is memorised, and the board is the only
honest ruler at one slot each. Five slots a day makes that unaffordable for thirteen.

Their DIVERSITY has no such problem. Distance is a property of the predictions, needs no labels and no
slot, and diversity is the term the ensemble bonus is actually paid for. One shared decode of 400 studies
fed all thirteen (the pass is decode-bound — 3% GPU at 18 GB), then `RankAverage.agreement`, the same
function the shipped vote averages in.

## Isolation, nearest-neighbour agreement, most isolated first

| member | family | nearest neighbour |
| --- | --- | --- |
| `raptor_ft_effv2l480` | `tf_efficientnetv2_l` @480 | **0.8223** |
| `raptor_ft_cnv2b336` | `convnextv2_base` @336 | **0.8429** |
| `raptor_ft_cnn336` | `convnext_small` @336 | **0.8492** |
| `raptor_ft_coatnet384` | `coatnet_rmlp_2_rw_384` @384 | 0.8863 |
| `raptor_ft_coatnet_v6_full_swa` | same | 0.9055 |
| `raptor_ft_coatnet384x` | same | 0.9087 |
| `..._v7_swa`, `..._v8_swa`, `..._v9`, `..._v10`, `..._v5_swa` | same | 0.9310 – 0.9363 |
| `raptor_ft_coatnet_v4_full` | same | **0.9955** |
| `raptor_ft_coatnet_v4_full_swa` | same | **0.9955** |

## Three readings

**1. The seated member's nearest neighbour is itself.** `v4_full` and `v4_full_swa` agree at **0.9955**.
They are one model — the SWA average and the raw tail of the same run. Any fleet blend that seats both is
seating one member at double weight while its docstring says one seat each. That is the arithmetic
drowning failure `RaptorSeat.fleet`'s own test warns about, arriving through identity rather than count.

**2. The inventory is four families and the count was never the number.** Ten of thirteen are
`coatnet_rmlp_2_rw_384` @384 differing only in fitting corpus, and they sit at 0.93+ from each other.
The three non-coatnets are the only genuinely isolated members, and their isolation is a **family** fact:
each one's nearest neighbour is another family, not another recipe. Seating all thirteen at 1/19 would
have given the coatnet cluster 10/19 of the blend — more than our entire roster — to say one thing.

**3. It corroborates the monoculture finding from the other side.** We found our own six vary only recipe
and read flat across seven board entries. Here, ten checkpoints that vary only fitting corpus agree at
0.93+, and the three that vary architecture drop to 0.82–0.85. Same law, someone else's pool: **recipe
distance is roughly a tenth of architecture distance.**

## What it selects

`raptor_ft_effv2l480` — most isolated of the thirteen, a fourth family (`tf_efficientnetv2_l` @480, a
different resolution as well as a different block layout), and its pairing with the currently-seated
`v4_full_swa` is 0.8223, among the most distant available.

That is the **second foreign arm** `raptorseat`'s own pre-registration named as its follow-up: *"whether a
second foreign arm adds again or whether one foreign decode is the whole of the effect."* Two foreign seats
at 1/8 each, not four at 1/10 — the three non-coatnets have **no measured quality at all**, and seating
three unmeasured members at 3/10 of the blend risks reading a dilution as a diversity verdict.

## What this does not say

Nothing about quality. `effv2l480` is the most distant member of the fleet and may still be the worst one;
distance and quality are independent here and only the board can price the second. It also says nothing
about how many board ticks a unit of distance buys — `raptorseat` bought +0.002 at 0.8223-ish distance and
one seat of seven, which is one point, not a slope.
