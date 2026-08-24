# The goal, priced against the inventory we actually have — 2026-08-24

Written on a night with no card (it went to the other project) and no submission slots until 00:00Z, which
is the right condition for this: it is arithmetic, not an experiment, and it should have been done before
several of the days behind it.

## The gap grew while we were closing it

`comp/champion.json` records the gap to the leader as **0.008**. Read from the API tonight it is **0.015**.

```
leader (Brandon Low)  0.952   2026-08-24 07:13Z
2nd                   0.948
3rd / 4th             0.947
...
ours (allfit)         0.937   rank ~44 of 2319
```

Our own climb, from `comp/board.jsonl`: 0.927 → the 0.933 chain → `finalfit` 0.934 → `allfit` 0.937, over
roughly eight days. The top of the board moved ~0.945 → 0.952 in about two. **Both are climbing and we are
not closing.** The pin's 0.008 was not wrong when written; it decayed, and nothing in this repo reads the
leaderboard on a schedule, so it decayed silently. That is the same failure shape as `board_transfer` being
quoted from prose at 4.67 / 2.14 / 1.92 while the tool reads **+1.82** — a number that lives in a document
instead of a command goes stale and then gets planned against.

## What first place costs, in the units our experiments produce

Taking first needs 0.952, so **+0.015 board** from where we sit. At the tool's measured compression:

```
+0.015 board  ÷  1.82 board-points-per-holdout-point  =  +0.0082 offline
```

For scale, the **entire offline spread this project has ever produced** — across ~15 shipped entries, every
member in the pool, and every arm in the ledger — is 0.8562 to 0.8731, a span of **0.017**. So first place
asks for roughly **half of our whole historical range, again, on top.**

That is not a knob-sized number, and it is worth saying plainly that it is not a several-knobs-sized number
either. Here is the inventory, with what each family actually returned:

| family | total returned | status |
|---|---|---|
| decode geometry (span, density, crop, depth regions) | ~+0.005 over four axes | closed, ledger's own words |
| pooling / aggregation | four operators span **0.0019** | saturated from three directions |
| resolution at constant field | +0.0002 | refuted, and re-refuted per-label on 2026-08-24 |
| SSL-as-init | −0.09 to −0.15 | five consecutive collapses, family closed |
| loss / objective | seat-positive once, retracted by `dissent_seat` | closed |
| encoder family / trunk | ConvNeXt +0.0259 **from a rate fix**, still 0.8186 | mechanism confirmed, no member |
| all-data dose | +0.003 over five extra seats | real, **sub-linear, and spent** |
| member count / seat swaps | every entry inside 0.930–0.937 | one draw from one distribution |
| multi-scale (tight + wide slots) | ≈+0.0016 estimated, 71.6 GB cache | gated OUT 2026-08-16 |

Summed generously and pretending they compose, which they demonstrably do not — span and depth-regions are
substitutes, crop and band are antagonists — the whole remaining inventory is short of +0.0082 by about 3×.

**Nothing we currently know how to build is sized for the goal.** Every honest path to it starts with
something not on this table.

## Where the deficit actually lives, and why the obvious fix is already dead

The metric is the unweighted mean of twelve per-label AUCs, and ours are very uneven:

```
Effusion            0.9453      Lateral Meniscus    0.6497
...                             MCL                 0.7927
                                ACL                 0.8640
```

The two worst labels are focal 1–3 mm structures. Lateral Meniscus alone, moved 0.65 → 0.80, is
**+0.0125 on the mean** — very nearly the whole gap, from one label. So the deficit is not spread evenly and
the goal does not require a better model everywhere; it requires the focal structures.

The trap is that this looks like it obviously wants more resolution, and **both ways of buying resolution
here are already refuted, in opposite directions:**

- **More pixels at constant field** (`res392`) — +0.0002. Re-tested per-label on 2026-08-24 with a paired
  bootstrap: not one of twelve intervals excludes zero, Lateral Meniscus +0.0002 [−0.0188, +0.0163]. So the
  aggregate null is not an average hiding a split.
- **Same pixels at a tighter field** (`crop80_span`) — −0.0099, and it loses *precisely* on the structures
  it was meant to help: **MCL −0.0265\*, ACL −0.0263\***. The physics is the explanation and it is not
  subtle: the ACL runs obliquely and the MCL sits at the medial margin, so neither is central, and an
  80 mm crop cuts them out of the field.
- **Both at once** (multi-scale: keep wide, add tight) is the scheme those two failures point at, the
  infrastructure exists (`Slot.crop_mm`, slot attention already pools over it), and it was **priced and
  gated out** at ≈+0.0016 in August. It is the right shape and the wrong size.

## The consequence, which is a change of target and not a change of knob

Three separate lines of evidence — the closed inventory, the per-label deficit, and the refuted resolution
axis — converge on the same statement: **the bottleneck is what one slice's feature vector contains, and
every lever we have left operates downstream of that.** The ledger reached this independently at
`attend_true` ("the limit is upstream") and again at `focal_dense` ("the bottleneck moves to supervision or
to the encoder"). This note is the third arrival, from the direction of the scoreboard.

What that licenses, in order of plausible size against the +0.0082 bar:

1. **A genuinely strong member on a different trunk, at a FITTED rate.** The ConvNeXt result is the live
   thread and it is the only confirmed +0.02-class mechanism this project owns: our 4e-5 was fitted over
   ~189 DINOv2-family runs and then handed to every other encoder unexamined, costing ConvNeXt +0.0259.
   The same defect was already sitting corrected-but-unnoticed on DINOv3 (LoRA-over-frozen 0.8268 vs full
   adaptation **0.8455**, +0.0187). Two families, two inherited hyperparameters, both written up as facts
   about architectures.

   **Corrected an hour after this note was first written, by the audit in the ledger's 2026-08-24 entry.**
   The obvious next step looked like `v3base_own_recipe` (written, `arm_check`-clean, 2h16m, a fitted
   1.5e-4 on DINOv3-base). It is a ~0-expected run and should not be armed: `v3base_dense`'s curve peaks
   at e6 and then runs its remaining five epochs without beating that peak, so DINOv3-base is already
   converged at 4e-5 and does not carry the ConvNeXt defect. The real finding is elsewhere and larger —
   `raddino_dense` (0.8137) and `vjepa_member` (0.8343) are **crash readings**, killed mid-epoch at
   healthy throughput and entered in the ledger as facts about their architectures. RadDINO is the most
   pixel-distant trunk in the pool, which is exactly the empty cell the ledger's previous entry names as
   the last open one. A warm start from its e6 checkpoint costs ~1.65 h.

   So the thread is not "fit a rate on the next trunk". It is: **this project has never once measured a
   foreign trunk cleanly**, and three of the four numbers it reasoned from were bounded by the harness
   rather than by the architecture.
2. **Package `v3base_dense` (0.8455).** It is on disk and absent from `comp/members.roster`, so the package
   holds the WEAK end of the v3 family and not the strong end. Card-free; needs an upload window.
3. **Supervision that says WHERE.** `aprior_dense` showed the head already recovers the plane-by-finding
   map from study-level labels, so a warm start buys nothing — the localisation argument has to buy a real
   localisation signal, not an initialisation.

What it forbids: another seat swap, another decode knob, another pooling operator. Those are measured, and
they are measured at a tenth of what the goal costs.

## The honest caveat on this whole note

`board_transfer`'s +1.82 is fitted **entirely on entries at or below 0.933** — our four best (`allfit`,
`finalfit`, `sevenplane`, `accumseat`) are one-sided and enter no pair. The +0.0082 is therefore an
extrapolation at exactly the part of the board we now live on, and the one direct probe we have of the top
of the range says the transfer **saturates** there (0.8712 → 0.933, then 0.8731 → **0.932**). If that
saturation is real, +0.0082 offline is a floor on what first place costs, not an estimate of it.

And by the house bar — no board difference under 0.02 is a result — this project's entire shipped history,
0.930 through 0.937, is one draw from one distribution. That applies to the gap too: 0.015 is *under* the
bar. First place is a real target; "we improved from 0.934 to 0.937" is not a real finding, and this note
should not be read as saying we are 0.015 of good work away. It says the work we know how to do is priced
at a third of the distance, and the rest has to come from somewhere we have not been.
