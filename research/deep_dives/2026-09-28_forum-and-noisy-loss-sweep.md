# Forum since 09-25, the real-noise loss benchmark, and DINOv2 registers — 2026-09-28

Round three of 09-28. Question: does anything read since the [gap audit](2026-09-28_gap-audit-and-frontier-sweep.md)
beat the two survivors (knee-2xdwi, knee-bpuei) on price? **Answer: no. Nothing new is filed.** One loss
(SCE) is recorded as ELR's cheaper sibling. It is gated on ELR's own reading.

Board read 11:47Z: rank 122 / 4499, tied with 30 others at 0.946. Leader 0.960, gap 0.014. The top five move
+0.00029/day and we move +0.00015/day, so the gap is not closing on its own.

## 1. Forum, 18 threads since 09-25 (`discussion_read --live --since`)

| topic | what is in it | against our record |
|---|---|---|
| 743148 "report text from images" | A competitor measured three things. (1) A BGE-M3 report-embedding probe reads 0.8423 on gold-58, below LLM+regex extraction at 0.8843. (2) The choice of LLM is noise: qwen2.5-14b 0.862, qwen3-14b 0.857, qwen3-30b 0.868. (3) Extraction caught 207 of 240 gold positives; 22 of the 33 misses are reports that never name the finding. Also claimed: 24→32 cached slices beat any label work. A second competitor claims EffNet-B0 5-fold+TTA at 0.94+ on 0.88 labels. | (1)–(3) agree with our closed label axes: the ceiling on extraction is silent reports, not the extractor. Slice count is **CLOSED** for us (AXES:90–91, 24→36 −0.0224). Their gain is from a lower base (24 slices). 5-fold+TTA is out of scope (ensemble; TTA null for us). |
| 740610 "Plateau at 0.903" | DINOv2-S at 336, six slots, 12 slices, twelve label queries. Specialist replaces generalist on 4 columns. 2×2 region tokens gave +0.000003. Synovitis is 85% masked. A reply: ResNet34/EffB0 at 224–288 with plain pooling reaches 0.94+, and they have a "low prior" on DINO init. | Specialists: CLOSED, 0 of 12 columns clear (AXES:194). CNN trunk: CLOSED. Nothing above us. |
| 742926 "improving a 0.940 ensemble" | A reproduction of a public DINOv3 blend. Replies are generic ("blend models that disagree"). | Ensemble: out of scope. |
| 743374 "<10 min submissions" | Inference engineering: one model per GPU, threaded DICOM reads, a raw PixelData offset read, fp16 weights. A team is being formed around "a resilient single model > 0.940". | Serving speed is not a score lever for us, since the runtime is not binding. The market signal is useful: 0.940 solo is what teams recruit for, and our solo reads 0.946. |
| 743416 / 743420 external-data rules | 31 votes, and hosts are asked for yes/no on MRNet, SKM-TEA, fastMRI, OAI, KneeMRI and KneeXNet-2.5D. One reply: "only the top 10 get audited". | **No host answer yet.** Our ruling stands (training on these is barred; foreign-derived data is instrument-only). Re-read after a host reply. A host YES would re-open the corpus lever, which belongs to another agent. |
| 743916 autonomous researchers | Meta AIRA took gold in the Nemotron challenge. | Context only. |
| 740164 backbone history | A survey PDF with no measurements. | Nothing. |

**Forum verdict:** no posted mechanism sits above 0.946 solo. The gold-58 extraction numbers are the one reusable
fact: 22/33 misses are silent reports. That independently confirms our label ceiling sits in the reports.

## 2. LNMBench — noise-robust losses on REAL medical label noise

arXiv 2512.09315 (Pattern Recognition 2026). The only benchmark we found that runs the loss family on
real-world medical noise, not on synthetic flips.

- **Result:** "only SCE and CDR achieve slightly better performance than CE" on the real-world sets.
  - SCE vs CE: DRTiD 56.01 vs 46.36; Kaggle DR+ 65.86 vs 65.59; CheXpert 40.46 vs 37.08.
  - CDR vs CE: DR+ 67.93; CheXpert 37.96.
  - Semi-supervised and sample-selection methods fail on real noise.
- **Why it transfers weakly:**
  - The metric "L" is mean test ACCURACY over the LAST epochs. It measures resistance to late memorisation,
    not peak ranking quality.
  - We select the best epoch on holdout and anneal under `one_cycle`, which removes most of what "L" rewards.
  - CheXpert was cut down to single-label (multi-label rows dropped), so none of it is a 12-label AUC.
  - SCE hyperparameters are not reported.
  - ELR, GCE, co-teaching and label smoothing were **not** run on the real-world sets, so the paper cannot rank
    SCE against ELR.
- **Mechanism class:** SCE = CE + a reverse-CE term. With log 0 clipped to A, the reverse term is a scaled MAE,
  a bounded loss that stops rewarding confident fits to wrong labels. That is the same class as ELR:
  anti-memorisation, instance-level. ELR is the stronger member, because its target is the model's own early
  reading rather than a fixed bound, and ELR is already implemented (428aa973).
- **Pricing:** SCE alone, P(clears +0.015 on ledger75) ~0.03. It would cost one ~40-min arm and a small `Term`
  (CPU work, ELR pattern).
- **Decision: not filed.** It is recorded as a conditional in knee-bpuei:
  - If ELR moves gold-58 or ledger75 by any directional amount, SCE is the cheap second probe of the same mechanism.
  - If ELR reads null, the class is answered for this recipe and SCE inherits the null. This is not a kill of
    the idea; it is the same mechanism at a weaker strength.

## 3. DINOv2 with registers (Darcet et al., ICLR 2024)

- **What it is:** register tokens absorb the high-norm "artifact" patches that DINOv2 writes into low-information
  regions. A related search hit (RMLP, arXiv 2511.05509; seen only as a snippet, content unverified) says the
  same artifacts appear in DINOv2-S.
- **Where it touches us:**
  - A head that attends over patch tokens with label queries could have its attention captured by artifact tokens.
  - MRI has large low-information regions (air, background), which is exactly where artifacts form.
  - `dinov2_vits14_reg` itself is untried. **Correction (same day):** registers as a MECHANISM already ran.
    DINOv3 carries four register tokens, and its head strips them from the patch mean
    (`kneemri/models/backbone.py:97-99`). The DINOv3 trunk row is CLOSED in `docs/AXES.md`. That test is
    confounded with patch 16 and a new pretraining, so it bounds registers rather than isolating them. The
    residual prior is lower still.
- **Against:** the trunk is fine-tuned end to end, which lets it re-learn where the artifacts go. And the published
  gains are on dense tasks and linear probes (classification +0.1–0.5 pt on ImageNet), not on fine-tuned
  classifiers.
- **Pricing:** one trunk-init swap arm, ~40 min of card. P(clears +0.015) ~0.02 (lowered after the DINOv3-registers correction above).
- **Decision: not filed as a lever.** It is worth one line as an **instrument** only: if an attention-map read of
  the shipped solo shows high-norm tokens on background patches, the prior rises and it gets priced again.
  That read is CPU-cheap and has not been done.

## 4. Where this leaves the queue

The ranking is unchanged:

1. knee-2xdwi, base trunk at 336/20sl, P ~0.25.
2. knee-bpuei, ELR, P ~0.08. Code landed; the arm is owner-GPU-gated.

Everything read this round is either already closed in `docs/AXES.md`, out of scope, or dominated by knee-bpuei.
The one fact that could change the plan is a host answer on 743416. It is external and not ours to force.

## 5. Round four (13:45Z): every candidate already has a verdict

| candidate | source | record |
|---|---|---|
| label-graph / co-occurrence head | Srivastava label-graph refinement; ML-GCN family | NULL twice: co-occurrence tail +0.017 ± 0.021 (interpretations 09-02), serve-time column mixing null (09-04) |
| hierarchical BCE (parent-child penalty) | CheXpert HBCE, 0.892 on 5 labels | the same inter-label constraint as the label graph; no new mechanism |
| per-label reliability + K-NN relabel | LRC-CXR (ESWA 2026) | relabel / prune is CLOSED; dropping UNK cells (confident224) −0.0071 |
| ASL | — | CLOSED on the record; ASL −0.139 in a competitor's ablation |
| in-domain SSL continuation | OrthoDiffusion (arXiv 2602.20752, no weights, no AUCs in abstract) | CLOSED (AXES:142, :159); multi-plane fusion is already consumed (AXES:95) |
| bone segmentation ROI | TotalSegmentator MRI | non-commercial licence, so not shippable; ROI crop is CLOSED anyway (AXES:87) |

**Nothing new is filed.** The one action this round is readiness: `experiments/arms/elr_hold.sh` is the
knee-bpuei gate (a held control plus ELR at 3.0, `--hold-annotation-studies`, so gold-58 is out of sample).
`arm_check` refuses neither arm. It needs the owner's GPU go: ~3.7 h of card, and it cannot share the card
(knee-l2d7o).

## Sources

- Kaggle discussion topics 743148, 740610, 742926, 743374, 743416, 743420, 743916 and 740164, read through
  `python -m kneemri.kaggle.discussion_read --live`.
- LNMBench: https://arxiv.org/abs/2512.09315
- Darcet et al., "Vision Transformers Need Registers", ICLR 2024: https://arxiv.org/abs/2309.16588
- RMLP (search snippet only, unverified): https://arxiv.org/abs/2511.05509
