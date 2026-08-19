# The ordinal ruler orders the seat the binary one inverted

Four submissions put four different members into the SAME seat (`lr1e-4_ws`) of the champion four,
everything else held. That experiment's headline, recorded in `comp/champion.json`, is that the offline
BINARY ruler picks the winner and does not resolve the field: read alone the three challengers order
`band_span` > `depth_regions_narrow` > `calib_span`, which is exactly BACKWARDS from the board, and read
in the seat they order `calib_span` > `band_span` > `depth_regions_narrow`, which gets the top right and
inverts the middle.

This note reports what a second ruler, already built and costing nothing, says about the same four.

## The measurement

`kneemri.evaluation.ordinal_signal` asks a different question of arrays already on disk: among studies a
member never trained on, how often does it rank a 0.94-graded study above a 0.68-graded one? It needs no
card and no labels file. All four members are scored over the SAME 334,942 cross-level pairs, so this is
one common ruler rather than four.

| member in the seat | binary alone | binary in seat | ordinal, labels | ordinal, pairs | board |
|---|---|---|---|---|---|
| `lr1e-4_ws` (incumbent) | — | 0.8648 | 0.735 | 0.766 | 0.928 |
| `band_span` | **0.8565** (1st) | 0.8661 (2nd) | 0.736 (3rd) | 0.767 (3rd) | 0.927 |
| `depth_regions_narrow` | 0.8554 (2nd) | 0.8653 (3rd) | 0.743 (2nd) | 0.769 (2nd) | 0.928 |
| `calib_span` | 0.8545 (3rd) | **0.8670** (1st) | **0.759** (1st) | **0.791** (1st) | **0.929** |

On the three challengers the ordinal ruler reproduces the board order exactly, on both of its means,
while the binary ruler read alone reproduces it exactly backwards.

The disagreement is not confined to this seat. `lr4e-5_ws` scores 0.8525 binary against `band_span`'s
0.8565 but 0.748 ordinal against its 0.736 — the two rulers cross. `band_full`, the worst member the
ledger holds (0.8427 binary, the arm that closed the span axis), is also last on ordinal at 0.726/0.765,
so the rulers do agree where the gap is large.

## The confound this had to survive

`calib_span` trains with `--calibrated-targets`, which reshapes the graded levels the ordinal ruler asks
about. A model trained to respect that grading could score high on this ruler BY CONSTRUCTION, and that
mechanism would explain the whole table without implying any board skill.

It is directly testable, because a second member carries the same flag. `calibrated_targets` scores
0.739 / 0.776 — mid-pack, below `band_dense` (0.751 / 0.778) and `lr4e-5_ws` (0.748 / 0.770). So
training with calibrated targets does not by itself win this ruler, and the by-construction explanation
does not account for `calib_span`'s 0.759 / 0.791. It is weakened rather than eliminated: `calib_span` is
the calibration applied at the shipped operating point, and no member isolates that.

## What this is worth, stated honestly

**It is a lead, not a finding.** Three items ordered correctly is 1 chance in 6 under a random ordering.
That is not evidence by this project's bar, and no amount of the ordering being "exact" changes the
count of items it was taken over.

**The board's own ordering may not be real.** The three entries differ by 0.001 and 0.002. Nothing here
establishes that the hidden-test metric resolves differences that small, and if it does not, then what
the ordinal ruler reproduced is the board's sampling noise rather than its ranking. This caveat is
larger than the result and is the first thing a follow-up has to address.

**The pairs are not independent.** 334,942 is a count of cross-level PAIRS drawn from 909 holdout
studies. It is the denominator of the statistic, not a sample size, and quoting it as though 334,942
independent observations stood behind a 0.759 would be the exact subset-versus-corpus error this repo
has paid for repeatedly.

## The forward test, pre-registered here

The next SINGLE-MEMBER seat swap submitted must be chosen where the two offline rulers DISAGREE, and
chosen by the ordinal one.

- **Confirms**: the ordinal-preferred member scores above the binary-preferred member on the board.
- **Refutes**: the board follows the binary ruler, or splits the two by less than the entries above
  differ from each other — in which case this note records a coincidence over three items and the
  correct reading is that neither offline ruler resolves the field, which is where
  `comp/champion.json` already stands.

Until that returns, nothing here changes what gets shipped. The roster is chosen by seat value against
`comp/champion.json` as before.

## Why it was worth measuring at all

knee-mxj argues that ordinal severity is the supervision that compounds with the ranking term, because
the competition metric IS a ranking metric. If that argument is right, a ruler built on severity
ordering should track the board better than one built on present-versus-absent — which is what the three
challengers show and what the confound check keeps from being circular. That does not make knee-mxj
right; it makes this the cheapest evidence the project has for it, and the forward test above is what
would turn it into evidence at all.

## Limits

- One holdout split, 909 studies, one seed per member. The seat experiment's four board points are the
  only external arbitration, and three of them sit within 0.002.
- The ordinal ruler excludes annotated rows by construction, so it speaks only about the LLM-derived
  grades. A member better at the annotated cases is invisible to it.
- `lr1e-4_ws` is the one member the ordinal ruler places badly: last of four at 0.735, against a board
  0.928 that ties `depth_regions_narrow`. Four points, one of them wrong, is the honest count.
