# Protocol: re-reading the 4.5k reports with Qwen3.8-27B NVFP4

**Context:** we now have a local Qwen3.8-27B NVFP4 endpoint on the RTX 5090 (`http://127.0.0.1:8001/v1`).  The existing `core.data.labels.report_reader` uses a local `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` loaded through `transformers`.  This note proposes how to test whether the new endpoint improves the report-derived labels, using the project's own gates.

## Why this is worth doing

- Only **58 of 4,407 training studies** have radiologist-adjudicated gold labels.  The rest rely on report-derived labels, and the current reader leaves **27.3 % of (study, finding) pairs as UNK** (`report_labels_v2.csv`), filling them with a base-rate prior.
- The public baseline audit confirms that the competition's test set has **no report column**, so any text work is training-only — exactly the right use for a fast local endpoint.
- Qwen3.8-27B NVFP4 is dense (not MoE), has real FP4 kernels on Blackwell, and is being configured for **128k context**, which removes length anxiety for long multilingual reports.

## Gate architecture (use the existing tooling)

The project already has the right acceptance tests; we just need to feed them a candidate table produced by the new endpoint.

1. **Produce a candidate table on the 58 annotated studies.**
   - Reuse `core.data.labels.report_reader.ReadingPrompt` and `ReportReading.parse` — the prompt and JSON parsing logic are already tuned and measured.
   - Add a tiny `EndpointReader` class that calls the local vLLM endpoint instead of loading via `transformers`:
     - base URL: `http://127.0.0.1:8001/v1`
     - model alias: `claude-opus-4-7` (stable across HF repo swaps; do not hardcode the repo id)
     - max tokens ~320, temperature 0, no streaming.
   - Output schema must match `report_labels_v2.csv`: `<finding>`, `<finding>__conf`, `<finding>__verdict` for the 12 findings, plus `StudyInstanceUID`.

2. **Run the gate:** `python -m core.data.labels.reader_upgrade --candidate qwen38_annotated.csv`.
   - This compares the candidate to the current table **only on the pairs the current reader left blank**.
   - It bootstraps over studies, not cells, and reports whether the new reader beats the majority-class baseline on those blanks.
   - A win here is the license to run the full 4,406-report sweep; a loss means the endpoint is not a better reader and we should not burn card hours.

3. **Run the acceptance test:** `python -m core.data.labels.reader_agreement --labels qwen38_annotated.csv`.
   - Compares the candidate's agreement with the radiologist against the current table.
   - Reports per-language agreement, false-positive vs miss balance, silence cost, and whether hedged/graded report flags move the needle.
   - The existing reader already has 108 false positives vs 32 misses on the annotated studies; a replacement should not trade one error mode for the other.

4. **Run the overcall audit:** `python -m core.data.labels.overcall_audit --labels qwen38_annotated.csv`.
   - Checks that the new reader does not re-introduce the attribute-negation mistakes the existing `NegationScope` module was built to fix (e.g. "tear without extrusion" read as no tear).

## Full sweep, if the gate passes

- Run the endpoint over all `train.csv` reports: `python -m core.data.labels.report_reader --reports train.csv --out report_labels_qwen38.csv` (with the endpoint-backed reader).
- Time estimate (rough): if a report + prompt + 320-token answer totals ~1k–2k tokens and the endpoint does ~50 tok/s serially, each report costs ~30–60 s → **~40–70 h for 4,406 reports**.  That is too long for one continuous run; batch it by language or by shard and resume.
- The endpoint is `--max-num-seqs 1`, so no batching; consider running only on the reports that the current reader marked UNK first, which is the high-value subset (~1,300 studies? Actually 27.3 % of 4,407 ≈ 1,200 studies, but the UNK pairs are spread across studies).  A targeted UNK-only sweep is the cheapest way to fill the silence.

## What to measure after the sweep

- `reader_upgrade` on the annotated studies (already done at gate time).
- `reader_agreement` per-language and per-finding.
- `kneemri.evaluation.supervision_gap` — does the new label table move the gap between report-derived and image-derived labels?
- `kneemri.evaluation.annotated_readings` — does a model trained on the new labels score higher on the 58 gold studies?

## Risks / open questions

- **Endpoint stability:** the HF repo id has changed twice today.  Use only the served-model aliases in any script/notebook.
- **Context length:** claimed 128k is not confirmed yet.  Wait for Claude-main's "ready" letter before launching the gate.
- **Prompt leakage:** the existing `ReadingPrompt.system()` embeds the full competition criteria and is long.  Verify token count; if it approaches the context ceiling, shorten or cache it.
- **Schema compatibility:** the new table must be byte-compatible with `report_labels_v2.csv` so that `reader_agreement`, `Targets`, `ConfidentLabels`, and downstream blends accept it unchanged.
- **No test-time text:** remember this can only affect training labels / distillation; the final submission model must be image-only.

## Suggested first command

Once the endpoint is confirmed at 128k context:

```bash
python -m core.data.labels.report_reader \
  --reports <path>/train.csv \
  --out processed/labels/report_labels_qwen38_annotated.csv \
  --annotated \
  --model endpoint:claude-opus-4-7
```

Then:

```bash
python -m core.data.labels.reader_upgrade \
  --candidate processed/labels/report_labels_qwen38_annotated.csv

python -m core.data.labels.reader_agreement \
  --labels processed/labels/report_labels_qwen38_annotated.csv
```

If both look good, move to the UNK-only or full sweep.
