# The strong are all twins: the monoculture, measured over 117 members

2026-08-30. `python -m kneemri.evaluation.pool_census`, card-free, no slots.
Answers the open form of knee-nj4l.

## The question, and why it was still open

knee-nj4l says the champion is six copies of one lens and the way out is a member whose errors are
somewhere else. Its own note, added the same week, says the proposition **has never been measured**: the
diverse-seating curve that appeared to refute it rested on `mae_matched` (DINOv2-small, four epochs
against a 12-epoch family, peak 0.6735) and `focus_mcl` (0.5332 on the twelve-label mean — near chance on
eleven of twelve columns). Those two are decorrelated because they are undertrained and narrow. The pool
contained no strong-and-distant member to seat, so the curve could not test the proposition its conclusion
was stated about.

## The measurement

117 members share a holdout (150 records on disk; `RecordMix.narrow` keeps the largest agreeing pool).
For each: its own mean AUC, its NEAREST rank agreement to any other member, its nearest belief divergence.

```
nearest-agreement range among the 14 strong: 0.9011 to 0.9496
nearest-agreement range across all 117:      0.0688 to 0.9664
```

`strong` = within 0.0062 — this project's measured arm-vs-arm floor — of the pool's best solo, 0.8642
(`stack_increments_s2026`). Fourteen members qualify.

**The distance axis has 0.90 of range in the pool and 0.05 of it among the strong.** Not one of the
fourteen strong members has a nearest neighbour below 0.9011, while the pool reaches down to 0.0688.
Every strong member the project owns is a twin of another strong member. That is the monoculture as a
measurement rather than as a configuration dump, and it is the mechanical reason seven differently
seated entries land inside 0.004.

## The cell is occupied and the occupancy means nothing

Three members clear both cuts — `labelwise_dense` (0.8613, agreement 0.9011), `finding_calib_dense`
(0.8600, 0.9179), `small_batch6` (0.8598, 0.9159). Read that as an artefact of the cut, not a result: the
`distant` cut is this pool's MEDIAN agreement, which by construction puts about half the pool below it, so
the three sit 0.002 to 0.019 under a line drawn through the middle of a tight cluster. The members this
project has actually measured as far — `margin_seat` at 0.8772, `mae_matched` at 0.4691 — are not in the
strong set at all. Occupancy under a median is a weak reading; the RANGE is the one that does not move
when the threshold does, which is why the tool now prints both.

## What it licenses, and what it does not

**A pre-registration bar for the non-DINOv2 lane.** The cnx / swin / vjepa arms are justified by
diversity, and diversity has until now been an argument rather than a number. It can now be one BEFORE a
slot is spent: a new member earns the diversity claim by landing **solo >= 0.8580 AND nearest agreement
< 0.90** — outside the band all fourteen strong members occupy. Anything inside that band is another
copy of the lens however different its trunk is, and the census will say so for free.

**It does not license a slot.** Nothing here is a board reading. `seat_value`'s own doc holds: the
offline ruler ranks correctly and understates what a weak member costs, and every seat value in the
current pool sits under the +0.0015 the board has never been shown to resolve.

## Two caveats that bound the claim

**Ours-only.** Seven foreign records are excluded by default, and correctly: `foreign_overlap` measured
896 of our 909 holdout studies inside the labels `raptor` was distilled from, which is why `raptor_solo`'s
0.8938 was voided on 2026-08-29. So the one genuinely different-family model the project holds is not in
the 117 — and its offline number is not trustworthy enough to put it there. The census describes OUR
pool, which is the pool a seat search actually draws from.

**Nearest, not mean.** A member could sit far from 116 others and near exactly one, and this reading
would still call it a twin. That is deliberate — the question is whether a member contributes an ordering
nothing else already contributes — but it means the census cannot be read as a statement about average
diversity.

## The non-DINOv2 lane, priced on the same axis — and one member of it that bought nothing

All nine of the project's non-DINOv2 members with a holdout record are already inside the 117; the ranked
table simply never reaches them. `--show`:

```
the 8 member(s) named, ranked by how far they sit from their nearest neighbour:
  vjepa_warm_s2026         solo 0.8337   nearest agreement 0.8335   divergence 0.0233 bits
  cnx_dense_s2026          solo 0.7927   nearest agreement 0.8410   divergence 0.0150
  cnx_own_recipe_s2026     solo 0.8186   nearest agreement 0.8496   divergence 0.0112
  swin_calib_dense_s2026   solo 0.8224   nearest agreement 0.8704   divergence 0.0187
  swin_supervised_s2026    solo 0.8331   nearest agreement 0.8797   divergence 0.0087
  swin_wholetop_s2026      solo 0.8274   nearest agreement 0.8797   divergence 0.0087
  raddino_dense_s2026      solo 0.8141   nearest agreement 0.9421   divergence 0.0064
  raddino_normfix_s2026    solo 0.8155   nearest agreement 0.9421   divergence 0.0064
```

**Six of the eight are the only members in the pool that are both far and not junk.** Every one of the
fourteen strong members sits at 0.9011 or above; these six sit at 0.8335 to 0.8797, and they carry solo
scores of 0.79 to 0.83 rather than `mae_matched`'s 0.4691. The lane's premise — that a different trunk
sees the knee differently — is **confirmed on the distance axis**. What it fails is strength, by 0.024
(vjepa_warm) to 0.065 (cnx_dense) against the 0.8580 bar.

**RAD-DINO bought no distance at all, and that is the sharper half.** Both RAD-DINO members read 0.9421 —
INSIDE the band every strong DINOv2 member occupies, and nearer the pool than three members of the
champion's own family. A medical-report-pretrained ViT-B is as different a pretraining objective as this
project holds, and on this ruler it is a twin. So `different architecture ⇒ different errors` is FALSE as
a general rule; it holds for ConvNeXt, Swin and V-JEPA and fails for RAD-DINO. The two RAD-DINO members
are also each other's exact twin (identical 0.9421 / 0.0064), so `normfix` bought nothing on this axis
either. Whatever RAD-DINO is re-tested for — its own learning rate, frozen features — **diversity is no
longer one of the reasons**, and that retires an argument, not the asset.

**The size of the ask, stated so it can be checked.** vjepa_warm needs +0.024 solo to clear the bar
while holding its distance. That is not a hopeful number: `cnx_dense` → `cnx_own_recipe` bought **+0.0259
from a learning rate alone**, and the ledger records that the inherited rate was the cost. So the gap
between this lane and the bar is exactly the size of a gain the lane has already produced once, on the
axis (recipe, not architecture) that `cnx_warm_anneal` and knee-7ybf are queued on. That is the strongest
mechanical argument this project has for spending card time there, and it now has a number.

**The bar is for a REPLACEMENT, not for seat seven.** Read `strong AND distant` as what knee-nj4l's
question needs — a member that could carry the plateau — never as a seating rule. The champion's own
record refutes it as one: `base_dense` lost its matched control by 0.0066, OUTSIDE the floor, and ticked
the board from a seventh seat; `through_dense` ticked it on pixel distance alone. These six sit 4x to 10x
further under than `base_dense` did, so the gap is real — but a seat is bought on the axis, and the axis
is what they have.

## Where this DISAGREES with the ledger, on purpose

`experiments/README.md:303` records the opposite verdict for the same member: vjepa_warm's **residual
distance 0.8210 against a roster whose most distant seated member is 0.8801** — read there as "the
volumetric trunk disagrees LESS than the 2D members disagree with each other." Two rulers, two answers,
and the disagreement is not resolved here.

What tips it is that the champion pin already retired the losing one in writing: *"stop using residual
distance as a selection rule, which has now failed twice in one day in opposite directions."* Rank
agreement is the ruler that survived that day. It is also measured differently — against 116 members
rather than one roster of six, which is the comparison a census makes and a seat scan does not. Neither
point makes the old reading wrong about what it measured; both are recorded so the next use of either
ruler is a re-test rather than an assumption.

## Side reading, free with the same run

`recal_dense_s2026` (0.8610) and `calib_dense_hold_s2026` (0.8634) report identical nearest agreement
0.9496 and identical nearest divergence 0.0065 bits: they are each other's nearest neighbour. Whatever the
gold ruler says about `recal_dense` under knee-9198, it is not new information for the ensemble.
`convergence_check` reads it converged, so its 0.8610 is a real peak rather than a truncation floor.
