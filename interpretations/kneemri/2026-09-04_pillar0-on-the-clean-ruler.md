# Pillar-0 on the only ruler that can price it — and what it costs the blend

2026-09-04. Reading: `logs/pillar_gate_out.log`, `logs/gold58_pillar_blend.log`.

## The question, and why it had only one legal form

Pillar-0 (`fabotelli/pillar0-knee-finetune`, ECL-2.0) is the largest architectural distance available to
this project: a true 3D BreastMRI foundation trunk against a pool that is DINOv2/DINOv3 slice readers plus
one CoAtNet. Q-DU's census puts it first of seven public families for exactly that reason. The open
question was never whether it is a good member — it is whether it is a COMPLEMENTARY one, since the pool
has read seven board flats on member quality and diversity is the axis that has not been closed.

It cannot be asked on our 909-study holdout. A-DV establishes the fine-tune trained on the competition
train set with only the 58 expert-annotated studies excluded, so our holdout is ~100% in-sample for it —
the `raptor_solo` defect that voided a 0.8938 at 98.6% in-sample. The 58 annotated studies are the only
cells both families are outside of, and our own members are inside those too unless they were fitted with
`--hold-annotations`. That leaves exactly one legal comparison: `holdannot`/`recal_holdannot` against
Pillar-0 on 58 studies.

## What it reads

| member | gold-58 macro AUC |
|---|---|
| `recal_holdannot_s2026` | **0.9037** |
| `holdannot_s2026` | 0.8932 |
| `pillar0` through our `336px_12sl` feed | **0.7853** |

Pillar-0 is **−0.1079 against `holdannot`, interval [−0.1476, −0.0700]** — it clears, so the quality gap is
a fact about the members and not about the draw of 58 studies. Its own published macro on these same
studies is 0.7970, so our feed loses only 0.0117 to theirs: this is not a plumbing shortfall being read as
a model. The published checkpoint is simply 0.10-0.12 of AUC below what this pool already has, on the
ruler the publisher chose.

## What it costs the blend

Equal-weight rank average, same 58 studies:

| blend | macro | vs best member alone |
|---|---|---|
| `recal_holdannot` alone | 0.9037 | — |
| `recal_holdannot` + `pillar0` (1:1) | 0.8805 | **−0.0232** |
| both holdannot members (2) | 0.9030 | −0.0007 |
| both + `pillar0` (1:3) | 0.9017 | −0.0013 |

Diversity does not rescue it at any weight this pool would use. At 1:1 it costs 0.023; at 1:3 the cost
shrinks to 0.0013 only because the seat shrank — the direction never turns positive. The remaining form is
a heavily downweighted seat, and that experiment has already been run in a different family:
`fleettwo` seated `raptor_ft_effv2l480` at 1/8 on exactly the "distant and bad" argument, and read flat on
the board.

## The verdict, and what it does NOT close

**The Pillar-0 board slot is parked.** It was going to cost a 3D preprocessing path inside the kernel, a
mounted 1.2 GB trunk, and a T4×2 runtime argument nobody has measured (A-DX: no published number, no stated
inference-time series cap, ours is 0.70 s/series locally against a 9 h wall-clock cap). Spending that to
seat a member measured 0.10 below the pool on the only clean ruler is not a diversity bet, it is a
dilution with extra steps.

What this does NOT say:

- **Not that 3D is closed.** This prices ONE published 3D checkpoint, cross-anatomy (breast → knee) and
  fine-tuned for two epochs. A volumetric trunk trained on knee from the start is a different arm, and
  A-DS lists candidates.
- **Not that Pillar-0's FEATURES are worthless.** This measures its published HEAD. A probe or an LP-FT
  over its trunk is a separate reading, and the trunk is the part the cross-anatomy pretraining bought.
- **Not a n=909 claim.** 58 studies, per-label SE 0.04-0.09. The member gap clears its interval; the blend
  deltas are quoted without one and are directional only.

RE-OPEN BY: a Pillar-0 arm that changes what is being seated — its trunk under our head, not its head — or
a 3D checkpoint whose gold-58 number is inside 0.02 of this pool's.

The loader (`pillar_member`, `atlas_import`) and the oracle (`pillar_gate`) were retired 2026-09-25 with
nothing left calling them; a re-open restores them from git history (`git log --all -- kneemri/inference/pillar_member.py`).
