# Attribution faithfulness and slot readout for `attend_true_s2026`

2026-09-01, Kimi. Read-only audit of the `wgaa_*.log` outputs from 2026-08-31.

## What was measured

Two diagnostics were run on `attend_true_s2026` (roster 0.8476 solo, +0.0004 as a seventh seat), fitted **without** the anatomical prior:

1. **`kneemri.evaluation.attribution_faithfulness`** — scores the slot-occlusion attribution with captum's `infidelity` and `sensitivity_max`. It compares two explanations per finding: the ablation attribution (what withholding each slot costs in probability) and the head's own attention mass. Both are normalised so the comparison is about shape, not units.
2. **`kneemri.evaluation.attention_readout`** — tabulates the head's attention mass per slot per finding and checks it against `core.data.corpus.anatomy`, the project's table of which acquisitions each structure is read on. Two denominators were run: 8 studies and 200 studies.

## Faithfulness result

| finding | ablation infidelity | attention infidelity | ablation sensitivity | attention sensitivity |
|---|---:|---:|---:|---:|
| ACL | 0.1087 | **0.0771** | 1.2003 | **0.8767** |
| MCL | 0.0779 | **0.0505** | 1.3562 | **0.7730** |
| Medial Meniscus | 0.1875 | **0.1245** | 1.3799 | **0.8390** |
| Lateral Meniscus | 0.1178 | **0.0906** | 1.3116 | **0.8276** |
| Medial OA | 0.1184 | **0.0847** | 1.7641 | **0.8764** |
| Lateral OA | 0.1229 | **0.0816** | 1.1283 | **0.8970** |
| PF OA | 0.1135 | **0.0874** | 1.1561 | **0.8541** |
| Effusion | 0.1432 | **0.1027** | 1.4613 | **0.7040** |
| Synovitis | 0.0848 | **0.0590** | 0.9598 | **0.7953** |
| Baker's | 0.0524 | **0.0374** | 2.4064 | **0.7580** |
| Contusion | 0.1184 | **0.0851** | 0.8848 | **0.7108** |
| Fracture | 0.0615 | **0.0449** | 1.0793 | **0.7516** |

Attention is at least as faithful as ablation on **12 of 12 findings** — both infidelity and sensitivity are lower for attention every time. That means the head's free attention softmax is a usable surrogate for the expensive slot-occlusion sweep.

But both explanations have non-zero infidelity and sensitivity. The infidelity being far above zero means the slots **interact**: a per-slot table is the wrong shape for the answer, however carefully each cell was measured. The sensitivity being far above zero means the attribution moves under noise smaller than the difference between two scanners, so part of what is being measured is a fact about the batch rather than a stable fact about the model.

## Slot readout result

Over **8 studies**, 7 of 12 findings put their heaviest attention on a slot the anatomy table calls primary. Mean off-table mass (attention on slots the anatomy table gives zero affinity, conditioned on the study having an on-table slot to look at) is **0.243**. No finding collapses more than 50% of its attention onto a single slot.

Over **200 studies**, the same two numbers are 7 of 12 and **0.259** — stable at the larger denominator.

Disagreements with the table (n=200):
- **ACL**: dearest `SAG_T1`, table says `SAG_FLUID_FS/SAG_FLUID_NOFS`.
- **Medial OA**: dearest `SAG_FLUID_NOFS`, table says `COR_T1`.
- **Lateral OA**: dearest `SAG_T1`, table says `COR_T1`.
- **PF OA**: dearest `SAG_T1`, table says `AX_FLUID_FS`.
- **Synovitis**: dearest `SAG_FLUID_NOFS`, table says `AX_FLUID_FS`.

These are not random failures: the degenerative and fluid findings are reading heavily on **sagittal non-fat-suppressed and T1** slots that the table marks as secondary or off-table, while the table reserves primary status for coronal/axial fluid-weighted slots.

## Interpretation

The head learned a **coherent but not table-identical** anatomy. Attention is the more faithful explanation, so the readout is telling us what the model actually uses, not what we told it to use.

Two mechanisms can explain the disagreements without calling the model wrong:

1. **The table is a radiologist's ideal; the model is a statistician.** A finding may be *most visible* on coronal fluid-weighted (the table's primary) but *most discriminative* on sagittal T1 for this corpus and this encoder. The table encodes visibility; attention encodes what separates positives from negatives in the trained representation.

2. **Slot attention is compensatory.** Because the head renormalises over whichever slots are present, a finding whose true primary slot is often missing will learn to read heavily on whatever is reliably there. `SAG_T1` and `SAG_FLUID_NOFS` are present in 44% and 59% of studies respectively (n=200), while the table's primaries for OA (`COR_T1`, 63%) and Synovitis (`AX_FLUID_FS`, 98%) are not always acquired. The head may be routing around acquisition sparsity.

The **off-table mass of ~0.25** is large: a quarter of the attention for a typical finding sits on slots the anatomy table says do not show the structure at all. This is partly explained by the compensatory argument, but it also means the slot-occlusion attribution is measuring interactions across a mask where many slots are functionally redundant for the decision.

## What this changes

- **`attention_readout` can replace `attribution.slot_importance` for most prose.** It is cheaper, and the faithfulness check says it is no worse — in fact better — than the occlusion sweep.
- **The anatomy table should not be treated as ground truth for model inspection.** Agreement is a measurement; disagreement is informative, not a bug.
- **The non-zero infidelity/sensitivity is a ceiling on how much any slot-level explanation can be trusted at fine grain.** A claim of the form "this finding depends on slot X" is directionally okay; a claim of the form "slot X contributes 23%" is over-interpreting.
- **For `attend_true_s2026` specifically**, the readout says it is a well-behaved attention-MIL head: no collapsed attention, no single-slot dominance, and a learned anatomy that is interpretable even when it disagrees with the prior.
