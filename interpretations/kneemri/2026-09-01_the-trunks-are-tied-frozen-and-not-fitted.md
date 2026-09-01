# The trunks are tied frozen and not tied fitted — knee-gd62

Reproduce: `python -m kneemri.evaluation.feature_probe --penalty 10 100 1000`. Ridge probe over the
pooled frozen caches, fitted on the fold-0 training positions and scored on the same 909-study holdout
every member in this project is ranked on, through `LabelScores.area_under_curve`.

## The reading

| trunk | width | probe AUC (best λ) | λ=10 | λ=100 | λ=1000 | best fitted member |
|---|---|---|---|---|---|---|
| DINOv2-large | 1024 | **0.7996** | 0.7996 | 0.7852 | 0.7487 | 0.8525–0.8578 (roster) |
| V-JEPA 2-large | 1024 | **0.7985** | 0.7985 | 0.7881 | 0.7597 | 0.8337 (`vjepa_warm`) |
| DINOv2-small | 768 | **0.7896** | 0.7841 | 0.7896 | 0.7779 | 0.8525–0.8578 (roster) |

**Frozen, the two large trunks are 0.0011 apart. Fitted, they are about 0.019 apart.** The probe ordering
does not reproduce the member ordering, which is exactly the branch knee-gd62 pre-registered as re-opening
the trunk lane.

This is the SECOND independent frozen measurement to say it. `vjepa_probe` (2026-08-22) read 0.7620 against
a capacity-matched 0.7669 and its own ledger row already called it "a TIE on frozen features". That row was
then read, on 2026-08-30, as a lane that is *quality*-bound — see
`2026-08-30_the-non-dino-lane-is-quality-bound.md`. Two frozen probes on different renders both refuse that
reading: what is worse about V-JEPA in this repo is not what its features contain.

## Where the gap actually opens

Frozen features equal, fitted members 0.019 apart, means the loss is taken during FITTING — recipe, not
representation. Candidates, none of them yet measured against each other:

- the V-JEPA member was fitted `--whole-stack --slices 20` at 256 px, a feed no DINO member uses;
- `--backbone-lr 4e-5` was carried over from the DINO recipe rather than fitted for a 24-block tubelet
  trunk, and `warm_start_time_yes_quality_not_yet` already reads warm starts as time-not-quality;
- every schedule constant in the arm (epochs, one_cycle shape, EMA decay) was tuned on DINO runs.

That makes the trunk lane a RECIPE question, which is cheap, rather than an architecture search, which is
not. It is also the one lane that survives the label lane closing (+0.0003 / +0.0009 / +0.0002).

## Per finding, at λ=10 — the trunks are not interchangeable

V-JEPA leads on PF OA (0.8062 vs 0.7944), Baker's (0.7952 vs 0.7837) and Contusion (0.8033 vs 0.7863).
DINOv2-large leads on Fracture (0.7640 vs 0.7341), Synovitis (0.8690 vs 0.8459) and Medial Meniscus
(0.8031 vs 0.8011). A 0.0011 mean gap covers per-finding gaps up to 0.03 in BOTH directions — the
complementarity the ensemble lane wants is visible in the frozen features, before any head exists.

## What this does NOT establish

- **A linear probe on pooled features is a floor, never a ceiling.** A trunk whose signal is positional or
  non-linear is under-read here by construction; the comparison is trunk-against-trunk under one probe.
- **The caches are not matched renders.** DINOv2 is 336 px × 12 slices × 6 slots; the V-JEPA cache is
  `r256 mean_channels`, already reduced over slices. The probe therefore compares the trunks AS THEY WERE
  CACHED, and a render effect cannot be separated from a trunk effect by this instrument.
- **Probe AUC and member AUC are not the same ruler** — same split and same scorer, different inputs. The
  0.019 is a gap between two orderings, not a subtraction that means anything on its own.
- **DINOv2-small probing within 0.010 of DINOv2-large** says the width is nearly free at this corpus size,
  which is a data-scale statement (4,407 studies) and belongs to knee-1a1t, not here.

## What it licenses

One pre-registered arm: the V-JEPA trunk under a recipe fitted FOR it (rate and feed swept on the trunk,
not inherited), against its own 0.8337 as control. If the fitted gap closes toward the frozen tie, the
non-DINO lane re-opens and the 84%-one-encoder monoculture has a real exit. If it does not, the loss is in
something neither frozen features nor these three knobs can see, and the lane closes on evidence rather
than on a single arm.
