# The report lane is the one unswept axis, and the text is a PARTIAL shortcut

*2026-09-17, 11:20Z. CPU only — one MiniLM encode of 4407 reports (~3 min) and twelve logistic probes
(~10 s). No GPU, no submission slot.
`python -m kneemri.evaluation.label_invertibility --vectors <npz> --labels report_labels_v2.csv
--holdout labelwise_depth24_drop40_long_s2026`.*

## How the axis was found: census the KNOBS, not the results

The seat lane and every capacity axis closed in the previous reading. The question that followed was
which axis this project has never varied AT ALL, and that is answerable mechanically: one pass over the
364 stamped checkpoints, counting distinct values per `options` key.

```
NEVER VARIED (one value across the whole shelf)
  folds 5 (364/364) · split_seed 2026 (364/364) · paths.yaml (364/364)
  report_vectors None (358/364) · report_weight 0.0 (358/364)
  target_corpus canonical (169/364) · tie first (346/364) · sharpness 0.0 (54/364)
WELL SWEPT
  image_size 4 · crop_mm 4 · slices 5 · fold 5 · batch 7 · slice_pooling 7 · scheme 8
  train_size 9 · epochs 10 · teacher_targets 10 · vram_gb 11 · backbone_lr 12
  encoder_weights 14 · variant 18 · labels 26 · out 332
```

**Report supervision is the only lane that is fully BUILT and has never carried a nonzero value on any
member on the shelf.** `kneemri/embeddings/report_embeddings.py`, `kneemri/training/report_supervision.py`,
a `ConfigLeaf` at `config_tree.py:153`, slot-head integration, and a guard at `run_setup.py:383-391` that
refuses a weight with no vectors — all of it, unused.

## The lane was not untouched, and the kill on it is honest

`experiments/arms/report_supervision.sh` carries a real two-weight pre-registration:

```
control  rank24_s2026 = 0.8272
N  report term at 0.3            0.8219   -0.0053
O  report term at 1.0            0.8209   -0.0063
```

Monotone in the weight, which is what makes it a kill rather than a tuning miss. Its stated mechanism:
the report describes the whole study — technique, history, incidental findings — so an auxiliary cosine
pull toward a WHOLE-DOCUMENT embedding drags capacity toward text the metric never scores.

**That kill covers one mechanism and not the other.** It refutes a cosine auxiliary target. The public
baseline's shape is different and has never been run here: broadcast the text vector to all twelve
targets, CONCATENATE with the pooled vision vector, one MLP, `binary_cross_entropy_with_logits` — no
contrastive, no cosine, no auxiliary loss, and the text branch STAGED on rather than live from step zero
(`research/2026-09-02_public_baseline_text_branch_objective.md`). Per the standing rule a refuting
experiment kills that USE of an asset, not the asset.

## The structural fact that made the concat lane look dead before it was measured

Text is **training-only** — `test.csv` has one column and no `Report`, so the text vector is `None` at
inference. The mechanism therefore has to be "text makes the VISION branch learn better", and there is a
reason to think it would do the opposite.

**`train.csv` carries the twelve label columns on 58 of 4407 rows.** The other 4349 are NaN; the 58 are
the radiologist gold set. Every member on the shelf actually trains on `report_labels_v2.csv`, and that
table was built by three extractors — `report_labeler` (lexicon), `llm_labeler` (local weights),
`api_labeler` (hosted, schema-enforced) — all of which READ THE REPORTS. So the input a text branch would
concatenate and the target it would fit are two reads of one document. A head handed a near-invertible
copy of its own target does not enrich the image branch; it starves it, because the text path absorbs the
gradient and then vanishes at the one moment the image branch has to answer alone.

The research note flagged this and never measured it: *"label leakage through the report is a real
concern"*, *"ablate the vision input and see how much the text branch alone predicts."*

## What the probe read, and it refutes the framing above

Report → 384-dim multilingual MiniLM, mean-pooled over the attention mask and L2-normalised, no PCA
(`--dims 384`; the module's default 64 would have understated the text in the direction that flatters
the lane). One logistic probe per finding, fitted on the 3497 training studies, scored on the **exact
909 studies the reference member held out**, taken from its own `.holdout.npz`. Binarised by `Ledger75`'s
own mixed cut — ten findings at 0.75, Medial Meniscus and Baker's at the midpoint.

```
0.8480   153 pos  ACL              0.8311   237 pos  Effusion
0.8251    57 pos  MCL              0.9208    71 pos  Synovitis
0.8920   484 pos  Medial Meniscus  0.8702   218 pos  Baker's
0.8309   138 pos  Lateral Meniscus 0.8037   165 pos  Contusion
0.8765   232 pos  Medial OA        0.8632    42 pos  Fracture
0.8459   130 pos  Lateral OA
0.8559   309 pos  PF OA            mean 0.8553
```

**0.8553, against the 0.9053 `ledger75` the best image member reads on the same 909 rows under the same
ruler.** The prediction that framed this run — that a target extracted from the reports would be
near-invertible from them — is REFUTED at this representation. The image model beats the text on the
text's own label.

The mechanism of the shortfall is not mysterious. The extractors are LEXICAL: they key on the words a
radiologist chose. A 384-wide mean-pooled sentence vector is precisely the transform that destroys word
identity, and a linear read of it cannot re-derive a lexicon rule. So **0.8553 is a FLOOR on
invertibility, never a ceiling** — the extractor itself scores 1.000 by construction, and a stronger
reader lies somewhere in between. This caveat is load-bearing here in a way it would not be at either
extreme.

## The middle band is not "open", and that is the finding

Gradient starvation does not need invertibility of 1.0. It needs the text path to be **cheaper** than the
image path. 0.8553 is available from a FROZEN vector through ONE linear layer; 0.9053 costs a ViT trunk
and an eighteen-epoch schedule. A head given both from step zero takes the cheap one, and the cheap one
is gone at test time.

**This is why the public baseline stages the text branch.** The research note recorded `use_text_branch
= False` first as a fact about their code and gave no reason for it. This number is the reason.

So the verdict is CONTESTED rather than open or closed, and it licenses exactly one shape of arm:

- text branch **staged on** after the vision branch has converged, never live from step zero;
- a **vision-ablated** control, so the text branch's own contribution is measured rather than assumed;
- and it is **incompatible with mixup as currently wired** — `experiments/arms/mixup224.sh:53-62`:
  `ReportSupervision` is handed the batch's POSITIONS rather than a table of rows, so it alone reads
  un-blended while the label, ranking and teacher terms all read blended rows.

**Not armed here.** Priced at ~4 h of card for a mechanism whose best evidence is a middle-band number
and whose nearest relative is a monotone kill. Against the standing 1.5% bar and a +0.012 gap to first
place, that ratio does not beat what else the window could hold. It is written down so the next arm that
wants this lane starts from the number instead of from the argument.

## The second reading, and it is free

The probe's per-finding spread is a label-noise instrument at no extra cost. Synovitis reads 0.9208 —
the highest — on only 71 positives, while Contusion reads 0.8037 on 165. A finding whose extraction is
HIGHLY invertible from the text is one whose label is close to a pure restatement of the report; one that
is not is either genuinely lexically diffuse or noisily extracted. That bears on knee-312m's measured
714-of-2604 silent OA cells (27.4%) and knee-5oxzo's 21.1% cell disagreement against the radiologist, and
the three OA columns here (0.8765 / 0.8459 / 0.8559) sit in the middle of the range rather than at either
end. Not pursued in this reading; recorded so it is not re-derived.

## The denominator, and the cut

909 held-out studies of 4406 in the label table, 4407 reports encoded. Training rows 3497. Positives per
finding range 42 (Fracture) to 484 (Medial Meniscus) and are printed beside every AUC, so a rare column
cannot hide behind a ratio. `ledger75` mixed cut, not the midpoint — a number quoted from this page
carries that cut. AUC is the rank statistic computed in-module, owing nothing to a metric import.

## A near-miss worth recording

The gate that blocks on a misquoted board number (`board_claims --assert`) reads an EMPTY description as
CLEAN. Correcting knee-4zw4n's `0.931` this morning, a `/tmp` that means MSYS to bash and `D:\tmp` to
Windows python fed `bd update --description ""`, and the checker went from one misquoted line to zero —
by deletion, not by correction. It was restored from the file within a minute, but a checker that cannot
distinguish "fixed" from "deleted" scores the destruction of its own evidence as a pass. Filed.
