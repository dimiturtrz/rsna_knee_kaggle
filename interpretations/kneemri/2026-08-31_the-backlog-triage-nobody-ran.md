# The backlog triage nobody ran: 289 open beads, two live lanes

2026-08-31. The owner has asked repeatedly for the list of buried high-potential ideas and it has never
been made. This is the list. It is built by enumerating every open bead (`bd list --status=open`, 289
issues) and then reading the DESCRIPTION AND NOTES of the ones whose titles claim a mechanism rather than
a knob or a bug.

The first finding is about the backlog itself and it is uncomfortable: **most of the exciting titles are
already dead in their own notes.** A bead's title is written when the idea arrives and never updated when
it is refuted, so the open list reads like a bank of untried mechanisms when much of it is a graveyard
with the headstones facing away. Any triage that reads titles — which is what every previous pass did —
overcounts the live lanes by about four to one.

## Dead in their own notes, still open, still exciting-looking

| bead | title claims | what its NOTES already say |
|---|---|---|
| `knee-ec0p` | through-plane masked prediction, SSL sees zero 3D structure | **CLOSED 2026-08-21.** Its blocker read out twice: 0.6902 and 0.6875, i.e. **−0.17** against `band_dense`. Masked reconstruction reshapes block 11 by 21–55% of its own norm and the head reads block 11. Indifferent to *what* is masked, so slice-masking re-buys the same −0.17. |
| `knee-blvm` | sequence head over depth, three RSNA winners use one | **CORRECTED hours after filing.** Two of its three claims are false: `DepthAttention` already reads slice order (`depth_true` 0.8461, second-worst of four), and our pooling is not MRNet's max. Four combining rules span **0.0019** total, a third of the 0.0062 floor. The aggregation stage is saturated from three directions. |
| `knee-ek5s` | topology — nothing models a tear as a discontinuity | **RAN 2026-08-27.** Supervised-contrastive alongside BCE: 0.8547 vs 0.8634, **−0.0087**, 11 of 12 labels down, the meniscus labels it targeted −0.0127/−0.0057. Strong form needs SKM-TEA masks, behind the external-data door. |
| `knee-bduq` | contralateral knee is a free labelled negative | **MEASURED AND CLOSED 2026-08-26.** Laterality normalisation already runs on 98.4% of studies; residual wrong-handed ≈67 studies (1.5%), two orders of thought below the focal deficit. The bilateral studies are 7 of 4407. |
| `knee-v1ec` | train/serve routing asymmetry | Three of five hazards **empirically closed**, the fourth closed by `SlotFill`. What remains is a duplicated lookup wanting one home — hygiene, not score. |
| `knee-c5n` | adapt the encoder to knee MRI without labels | Its CPU drift gate passed, and then the vehicle it licensed is the one `knee-ec0p` killed. The *lane* survives only in a form that does not run a reconstruction objective through block 11. |

That is six of the ten most promising-sounding titles in the backlog, and every one of them was settled by
a measurement this repo already holds.

## The two that are actually live

**`knee-8009` — feature distillation as a PRETRAINING stage (P1, unarmed).** The one bead whose own
pricing is the highest this project has given any unrun mechanism: P(clears the 1.5% house bar) ≈ 0.15,
P(clears the 0.0062 paired floor) ≈ 0.5. Three things make it different from every knob:

1. **It removes a named confound rather than turning a dial.** The monoculture table (`small` 0.8964 vs
   `cnxbase` 0.8186, `swinbase` 0.8331, `vjepa2l` 0.8343) has been read as an ARCHITECTURE result. But
   every DINOv2 trunk arrives with large-scale self-supervised pretraining and every non-DINO trunk
   arrives with ImageNet weights and nothing else. The two groups differ in pretraining as much as in
   block layout and this project has never separated the terms.
2. **The target size and the significance bar coincide.** `swin_supervised` is the most decorrelated
   member we have ever produced (nearest-neighbour 0.8346 against every seated member's 0.877–0.9225) and
   was refused a seat purely on solo strength. It needs +0.017 to +0.020 of solo to buy a seat instead of
   costing one — above the house 1.5% bar. A result big enough to seat is automatically worth reporting.
3. **The objective needs NO LABELS**, so it can consume studies the label table does not cover. That is
   the external-data route expressed as a member we can seat, without a licence ruling.

It differs from `knee-ec0p` on exactly the axis that killed `ec0p`: feature MATCHING against a teacher
embedding is not reconstruction, so nothing forces block 11 to be reshaped for a decoder.

Its cheap instrument (a) is buildable today with zero new code — the existing distill arm at
`--distill-weight` 10–30 instead of 0.3, `fitted_encoder` export, warm-started labelled fit. ~50 min +
~20 min, zero slots. It is an instrument and must be sold as one: a flat reading does NOT refute (b),
because ordering-over-twelve-targets was always the wrong signal for a pretraining stage.

**`knee-3oy8` — fastMRI+ as a fourth corpus (P2, unarmed, and mis-prioritised).** The external-data
programme has been carried by `knee-ombx` (MRNet), which is blocked on the owner obtaining an archive and
sits under a licence question the host has already ruled against for a sibling corpus. fastMRI+ is not
behind that door, and it beats MRNet on the axis that matters:

- **It carries SEVERITY.** 'ACL Low Grade sprain' and 'ACL High Grade Sprain' are distinct labels.
  MRNet's ACL is binary, which is exactly why `MRNET_LEVEL` had to take 0.82 instead of 0.94.
- **It supervises more than one column.** MRNet and Rijeka both supervise ACL alone, so eleven of twelve
  heads take zero gradient. fastMRI+ maps onto at least three — ACL (two grades), MCL, Effusion.

The data slope is concave (~0.48 of the last per doubling) so no single corpus clears the bar alone —
but all remaining doublings sum to ~+0.017, which does. **External data is a programme, not a download.**
Its honest mapping problems are already written down (Meniscus Tear does not split medial from lateral —
drop; cartilage labels name no compartment — drop; Fracture/Contusion is one label over two columns —
recoverable from box coordinates).

## What this says about how the backlog is being used

287 beads read as "ready" and the number is meaningless: readiness is computed from dependency edges, not
from whether the idea survived its own notes. Two consequences, both actionable:

1. **A refuted bead must be CLOSED, not left open with the refutation in its notes.** Six of these should
   have been closed on the day they were measured. Leaving them open is what makes the backlog unreadable
   and is why this triage had never been run — the cost of reading it looked like 289 items when it was
   really ten.
2. **Priority is stale where it matters most.** `knee-3oy8` sits at P2 and `knee-blvm` at P3 while
   `knee-blvm` is refuted and `knee-3oy8` is one of two live lanes above the bar. The instruments
   (`knee-iyad` Mahalanobis, `knee-c2ew` Jensen-Shannon, `knee-l1vy` Captum) were being used as the
   ANSWER to "what is unexplored", which they are not.

   That last clause first read "correctly ranked below both — they score zero points by construction",
   and the owner reversed it the same day: an instrument scores no points ITSELF and raises the
   information content of every arm that follows it, and the scarce resource here is arm budget, not
   ideas. The correction survives a check the ranking did not: **the instruments largely already EXIST
   and are not being RUN.** `knee-c2ew` is fully implemented in `kneemri/evaluation/member_divergence.py`
   and printed by `seat_value` — closed on audit, not built. `run_comparison` has produced per-label
   deltas with 95% intervals for weeks. The near-miss on `stack224` was a failure to RUN an instrument,
   not a missing one, and `kneemri/experiments/arm_read.py` (written the same day) is the fix: it joins
   the `# control  <name> = <score>` line `ArmHeader` already parses to the comparison `RunComparison`
   already computes, so the per-label read stops depending on somebody remembering. Its first census:
   **86 of 162 arms declare no control at all**, including `stack224`, `native224` and
   `calib_dense_hold` — the arms other arms are read AGAINST.

The honest summary of the whole backlog: **the mapped axes are closed, and the two remaining lanes are
both about what the encoder KNOWS before supervision starts — one by distilling a representation into a
trunk that never had one, one by feeding it a corpus with severity in it.** Everything else open is a
knob, a bug, or a headstone.
