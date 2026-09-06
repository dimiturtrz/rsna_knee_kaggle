# Arm A is done (v2 peak 0.8524 @e8). The deciding number is the GOLD read — take it now, no card.

From: mastermind, 23:34Z. 56064173 still PENDING. aprior_fixed on the card since 23:24Z (~00:45Z).

`labels_v4hybrid_s2026.metrics.jsonl`: v2-holdout 0.8524 peak at e8, flat e7-e11 (converged, not
truncated). That is −0.0054 against band_dense's 0.8578 ON THE V2 RULER, which was pre-registered as
control-only: a student of v4b labels scored on v2 labels is scored against the noisier table, so a
drop there is the expected sign, not a verdict. Do not let it enter the ledger as a kill.

The number that decides is `annotated_readings --per-label` on the held 58, against `holdannot_s2026`
0.8932. Bar as pre-registered at 21:03Z: Effusion/Contusion/Fracture mean ≥ +0.02, or twelve-mean
≥ +0.008; Synovitis is a CONTROL (should not move — the table's synovitis column is byte-identical).
Three branches, write the one that lands:
  CLEARS: successor arm queued behind seat_distill — same treatment on the labelwise-20sl base
          (0.8613), then the composition arm (v4b + mask + OOF distill) as planned.
  MIDDLE (twelve-mean +0.003..+0.008, E/C/F up): label axis real but small; it composes with distill
          or it is not worth a seat — the composition arm decides, queue it ahead of the successor.
  FAILS (≤ 0 on E/C/F): before "the table doesn't help", check the mask LANDED — count of
          supervised rows per column in the arm log must read Synovitis 694 / Fracture 1921 /
          Baker's 2383, and `hold_annotations True` is in the recipe (it is). Then suspect the v2
          eval table only. A −0.0054 v2 drop with a flat gold read says the model learned v4b's
          disagreements with v2 and gold sided with v2 — that would be a finding about v4b, not
          about label work.
Paste the per-label table with the branch. Also the 7-of-7 census on oofseat, still owed.

-- mastermind
