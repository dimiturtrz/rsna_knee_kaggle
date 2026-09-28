# 2026-09-28 — AXES gap audit and frontier sweep for a solo above 0.946

**Question.** Is there any lever NOT already closed in `docs/AXES.md` that could lift the solo
(`basinsoup_ff_k4_s2026`, board 0.946) past 0.947?

**Method.**
1. Walk the AXES rows looking for gaps: a lever never run, or one closed only by the machine.
2. Web search for published mechanisms, competition write-ups and new MRI foundation models.
3. Grep each find against `research/`, `interpretations/`, the ledger and the beads before recording it.

Tags: **[M]** measured by us · **[L]** literature · **[U]** unverified or secondary.

## Levers that survived, both filed as priced beads

| lever | mechanism | price | P(solo clears 0.947) | bead |
|---|---|---|---|---|
| base trunk at 336/20sl under WSL2 | Capacity is the one trunk axis that paid: +0.0067 matched at 224 [M] (AXES §A). The shipping corner OOMed seven times, all from allocator fragmentation: 28.5 GiB reserved against 23.2 GiB allocated [M]. `expandable_segments` fixes that on Linux and is a no-op on Windows [M]. No record of a Linux or WSL attempt existed | ~2 h CPU + ~1.3 h card for the gate, then ~4 h card for the soup path, + 1 slot | ~0.25 | knee-2xdwi |
| ELR noise-robust loss | Every label-noise attempt so far changed the TARGETS (calibration, confident, relabel, prune, distil, transduce). None changed the LOSS. ELR regularises predictions toward their own temporal average, which is aimed at our runs' peak-then-decline signature | ~1.5 h CPU + ~2.5 h card, then ~4 h card conditional, + 1 slot | ~0.08 | knee-bpuei |

The owner must decide on both before any card time. knee-2xdwi needs the card to itself (23.2 GiB is
above the 22 GB cap), and WSL has wedged on this machine after sleep/resume.

## Priced and not pursued

- **SWA.** There is no `--swa` in our tree.
  - Foreign reads: `_swa` +0.0005 over the best epoch (README:8901); `raptor_v4_swa` +0.0026 on ledger75 (README:9142), but that family's solo read was voided as in-sample.
  - Our stack already pays for the same averaging twice: EMA is closed, and the basin-soup premium is +0.0014.
  - Ceiling ~+0.001–0.003, P ~0.03. Recorded in the AXES §F row.
- **SAM at rho 0.01.** Already ran: +0.0032 ledger75, directional. Composing it onto the shipping recipe is priced at P ~0.05 (knee-k3v5t). Nothing new.
- **MST-style slice transformer [L].** Already answered by our order-aware operators:
  - `depth` pooling: `depth_true` 0.8461, second-worst of four operators.
  - `slice_shift`: +0.0035 at 12 slices, −0.0016 at 24 slices on the shipping line.

  See [slice aggregation summary](../summaries/slice-aggregation-and-head-architecture.md).
- **KneePreM [L]** (arXiv 2609.31461, 2026-09-25): a 3D U-Net MAE on 19k OAI series, fastMRI+ AUC 0.722, APM 0.740.
  - OAI is under a DUA, so it is instrument-only by the 09-15 ruling.
  - It is the same class as OrthoDiffusion, which was closed at −0.095.

  See [trunks summary](../summaries/trunks-pretraining-and-transfer.md).
- **Public write-ups [U].** None is above our rank; see the [landscape summary](../summaries/competition-landscape-and-frontier.md).
  - HF "rank 18" post: 0.903 public LB, EfficientNet-B3 with max-MIL. It names report-extraction accuracy as the ceiling.
  - GitHub repos Alherma7, Amrin24 and niha1905: no stated score above ours.
  - Already covered in earlier sweeps: TranBaDat2607, ShivenKhurana1 (0.935), raahimnawaz.
- **Recent clinical papers [L]:** the eClinicalMedicine 9-abnormality study and the Eur Radiol 23-condition 3D slice-transformer paper. Both were already cited in seven earlier deep dives.

## Verdict

No published or public mechanism clears the bar on its own. The only lever with a real mechanism and
a non-trivial P is **capacity at the shipping corner** (knee-2xdwi). What bars it is a platform
limitation, not a measured null.

Sources:
- https://arxiv.org/abs/2609.31461
- https://huggingface.co/blog/bishnoiyash/rsna-competetion
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/
- https://link.springer.com/article/10.1007/s00330-025-12052-8
- https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection
- https://github.com/ShivenKhurana1/rsna-knee-abnormality
- Liu et al., "Early-Learning Regularization Prevents Memorization of Noisy Labels", NeurIPS 2020
