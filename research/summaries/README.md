# Research summaries — one per topic

Two layers under `research/`:

- `deep_dives/<date>_<topic>.md` — one piece of research, frozen at the date it was written. It may be superseded.
- `summaries/<topic>.md` — the current synthesis of a topic. It opens with the verdict, and every claim links to the deep dive it came from.

**Start here, not in `deep_dives/`.** Where two deep dives disagree, the summary names the later or measured one. Per-lever verdicts live in [`docs/AXES.md`](../../docs/AXES.md), and the summaries cite its rows.

A new deep dive is not finished until its topic summary is updated, or a new topic is added below.

| Topic | Summary | What it answers |
|---|---|---|
| Competition & frontier | [competition-landscape-and-frontier](competition-landscape-and-frontier.md) | Where the board and public work sit; what the published and rival approaches are |
| Evaluation & transfer | [evaluation-metrics-and-transfer](evaluation-metrics-and-transfer.md) | The metric, rulers (midpoint vs ledger75), noise floor, offline→board transfer |
| Labels & label noise | [labels-and-label-noise](labels-and-label-noise.md) | How wrong report labels are; correction, readers, text supervision |
| External data & licensing | [external-data-and-licensing](external-data-and-licensing.md) | Which corpora exist, what each licence permits, the binding owner ruling |
| Trunks & pretraining | [trunks-pretraining-and-transfer](trunks-pretraining-and-transfer.md) | Encoder family, medical and foundation trunks, SSL continuation, LP-FT, efficiency |
| Slice aggregation & head | [slice-aggregation-and-head-architecture](slice-aggregation-and-head-architecture.md) | Slice/slot pooling, labelwise attention, ROI and localisers, TTA/MIL |
| Input, augmentation, losses | [input-normalisation-augmentation-and-losses](input-normalisation-augmentation-and-losses.md) | Crop/resolution/normalisation, augmentation, loss choices |
| Ensembling | [ensembling-and-decorrelation](ensembling-and-decorrelation.md) | Member diversity, combiners, what an ensemble implies for a solo |

## Known contradictions between deep dives (resolved 2026-09-25)

The later or measured side wins. Each summary explains its own resolution; this table is so the next reader of an older deep dive does not re-import a dead claim.

| Older / weaker claim | Superseded by |
|---|---|
| ROI crop is the #1 lever, +0.056 (`focal-regional`, `single-model-ceiling`) | Measured: appended ROI views lost, roi 0.8984 / roinative 0.9018 vs 0.9051 ledger75 (knee-vgz5b); `input_side` already called it spent |
| mixup, pos_weight, ghosting, AUC-margin, per-label loss weights untried (`knee-mri-normalization-augmentation`, `focal-regional`) | All measured and closed in AXES |
| slice coverage is an unread cell (`input_side`) | AXES: 12→24 slices flat, 24→36 −0.0224 |
| attention pooling untested (`qen`) | `second_moment_pooling`: attention/labelwise pooling already measured |
| label correction is a live lever (`qao`, `qap`, `extractor-repair`) | `state_of_labels`, `qeh` bound it at ≤0.003–0.006; AXES:154 closed |
| forward noise correction ranked #5, +2.44% (`single-model-ceiling`) | `loss-correction-with-measured-noise-matrix`: 80.8 vs BCE 87.0 |
| errors_dense "ran, 0.8420" (`label_noise_confident_learning`) | `qeb`: it had not run |
| Qwen3.8 is the best local reader (`local_vlm_shortlist`) | `qwen38_reader_gate_result`: 122 vs 121 wrong, no gain |
| RadImageNet 0.6716 collapse blamed on LR (`qbm`) | `frozen-batchnorm-drift`: BN running-stat drift voids the number |
| OrthoFoundation is the next experiment (`modern_knee_mri_models`) | Declined, 0.8452 vs 0.8502 |
| MRNet "CC-BY, Kaggle-OK" (`qxd`); OAI "public domain"; legal blocker gone (`scheme_b`, `using_external`) | Non-commercial / DUA, no redistribution; owner ruling 09-15: foreign-derived = instrument only |
| Host forum allows shipping non-commercial corpora (`host_rulings_forum_sweep`) | Owner ruling 09-15 is stricter and binding |
| A hosted LLM is "likely a Data-Security violation" (09-03) | Host post 733965 allows it (09-13) |
| Transfer slope 0.82, or +4.67 / +2.14 / +1.55 / +1.82 (`qts` and others) | Read `board_check` live; on ledger75 ~+1.10, resid sd ~0.0013–0.0017 |
| TTA and threshold calibration help (`past_ensemble_plateau`) | `qbo`: TTA can cost up to −31.6 pp; AXES: calibration is worth 0 for AUC |
| +0.031 F1 from a CNN family (`qi`) | `qel`, `qem`: public three-family blend ≈ +0.001 |
| No public notebook above 0.926 publishes a method (08-31) | 0.943 Hisernberg lineage is published (09-25) |

Still open, where the facts themselves disagree:
- the RadioDINO training corpus (RadImageNet vs chest X-ray)
- OrthoFoundation's init (DINOv3-L vs DINOv2)
- corpus size quoted as ~1,000 vs ~4,400 studies (4,407 is the train count)
- fastMRI+ at 974 on disk vs 1,172 quoted
- gold-58 per-label SE, ≈0.12 vs ≈0.055

Check a primary source before quoting any of these.
