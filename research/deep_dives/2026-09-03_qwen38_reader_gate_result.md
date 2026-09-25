# Qwen3.8-27B NVFP4 report-reader gate result

**Endpoint:** `http://127.0.0.1:8001/v1`, alias `claude-sonnet-4-6`, checkpoint `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, context 131,072.  
**Key API fix:** `chat_template_kwargs: {"enable_thinking": False}` is required, otherwise the model emits the analysis in the `reasoning` field and `content` comes back empty.  
**Script:** `.agent-runtime/scratch/endpoint_reader_gate.py`  
**Output:** `.agent-runtime/scratch/report_labels_qwen38_annotated.csv` (58 studies).

## Headline

Qwen3.8-27B NVFP4 is a **better-calibrated reader** than the lexicon and the existing local reader on the 58 annotated studies, but it is also **more silent**.  It trades false positives for misses and does **not** beat the existing local reader on net errors.

## Numbers on the 58 annotated studies

| Reader | Cells | Positive calls | Wrong | Silent | English agree* | Non-English agree* | FP | Misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-27B NVFP4** (this run) | 696 | 214 | 122 | 30.2 % | 79.2 % | 76.3 % | 48 | 74 |
| `local_reader_annotated.csv` (existing local model) | 696 | 285 | 121 | 21.0 % | 72.9 % | 69.2 % | 83 | 38 |
| `report_labels_v2.csv` (lexicon) | 684 | 311 | 140 | 25.9 % | 66.7 % | 64.2 % | 108 | 32 |

\* Agreement on the reader's positive calls against the radiologist.

## `reader_upgrade` vs the lexicon

```text
177 pairs the lexicon left blank; the candidate speaks on 17 (9.6%)
blank-pair accuracy 0.765 against a majority-class 0.824
the lexicon's own 461 answered pairs: candidate accuracy 0.833
blank-pair accuracy interval [0.619, 1.000] over studies
NOT DISTINGUISHABLE FROM THE MAJORITY CLASS
```

The new reader does not fill the lexicon's blank cells reliably enough to justify a full 4,406-report sweep on that basis alone.

## What Qwen3.8 does differently

- **Much less over-calling:** 48 false positives vs 83 (local reader) and 108 (lexicon).
- **Much more silence / misses:** 74 misses vs 38 (local reader) and 32 (lexicon).
- **Better positive-call agreement:** 79.2 % English / 76.3 % non-English vs 72.9 % / 69.2 % for the local reader.
- **Noisy per-finding calibration:** it badly misses Effusion (16 silent, 3 FP, 13 net under) and PF OA (10 net under), while Contusion is still over-called (14 FP, 3 misses, +11 net).

## Interpretation

The endpoint is usable and fast (~1.5 s per report once reasoning is disabled), but with the current prompt it is a **different reader, not a uniformly better one**.

Possible uses:
1. **Conservative second opinion / overcall filter:** because its false-positive rate is low, it can flag cells where the existing reader is confident YES but Qwen3.8 is silent/absent for manual review.
2. **Pruning the 108 confident false positives** identified by `reader_agreement` on the existing local reader.
3. **Prompt-tuning target:** the high silence rate looks like the model is interpreting "unclear" too readily; a stricter instruction against abstention may move it toward the local reader's coverage while keeping its calibration.

It is **not** a drop-in replacement for the existing local reader today.

## Next steps

- Decide whether to invest in prompt-tuning to reduce the 30 % silence rate, or use Qwen3.8 as an auxiliary filter.
- If prompt-tuning: hold out a few annotated studies, test variants of the system prompt that discourage `unclear`, and re-run `reader_agreement`.
- If auxiliary: generate a flag table of disagreements between the existing local reader and Qwen3.8, and measure how many of the 83 existing false positives it catches.

## Auxiliary-filter check (Qwen3.8 against local_reader positives)

If we treat Qwen3.8 as a filter that keeps only the studies where the existing local reader says YES **and** Qwen3.8 also says YES:

| Set | Count | After filter |
|---|---:|---:|
| local_reader true positives | 202 | 158 kept (78.2 %) |
| local_reader false positives | 83 | 38 remain (45.8 %) |

The filter would drop **44 real positives** but eliminate **45 false positives** on the annotated studies — essentially the same net correctness as the local reader alone, but with much higher precision.  Whether that helps the downstream model depends on whether the reduced false-positive load on the shared trunk outweighs the lost true positives; it is a plausible experiment but not a guaranteed win.

