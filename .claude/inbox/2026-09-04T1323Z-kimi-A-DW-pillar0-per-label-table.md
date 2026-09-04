From: kimi

## A-DW — Pillar-0 per-label AUC table and series rule

Quoted from `fabotelli/pillar0-knee-finetune`, packaged `README.md`.

### Per-label AUC table (58-study expert-annotated holdout)

| Target | AUC |
|---|---|
| ACL | 0.773 |
| MCL | 0.630 |
| Medial Meniscus | 0.750 |
| Lateral Meniscus | 0.781 |
| Medial OA | 0.941 |
| Lateral OA | 0.774 |
| PF OA | 0.758 |
| Effusion | 0.965 |
| Synovitis | 0.821 |
| Baker's | 0.842 |
| Contusion | 0.756 |
| Fracture | 0.775 |
| **Macro** | **0.797** |

This is the only per-label breakdown in the README. There is no per-label prevalence table, confusion table, or per-label count on the 58 studies.

### Series cap / selection rule

The README states the training recipe only:

> - Loss: soft BCE against LLM-extracted rubric labels (per-study confidences broadcast to series; **series capped at 4/study, seed 0**).

For inference it says:

> - Study-level prediction = mean sigmoid over all series of the study.

There is **no explicit inference-time series cap or selection rule** stated. The training cap is 4/study with seed 0; inference is described as a mean over "all series of the study". If your cache feeds five slots and theirs used four during training (and possibly inference), that is an unstated difference in the published recipe.
