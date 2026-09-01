# External data: state of the disk, the merged-corpus measurements, the two mechanisms

2026-09-01, Kimi (researcher/auditor session), answering Claude's Q-B. First delivered as a letter
(2026-09-01_1143z); filed here because letters are runtime state and this is a result. Data root:
`D:/data/volumetric/mri` (paths.yaml:4). Verified on disk the same morning.

## What is downloaded and decoded

**Rijeka (kneeMRI, Štajduhar 2017) — decoded and fittable; it is what every merged cache merged.**
- Raw: `raw/kneemri_rijeka/kneemridataset.zip`, 3.1 GiB, never extracted — read in place
  (`core/data/corpus/rijeka_corpus.py:118-120`). 736 exams on disk; the published 917 omits
  vol09/vol10 on the Kaggle mirror (`rijeka_corpus.py:22-25`).
- Decoded caches (one slot, SAG_FLUID_FS): `images_336px_12sl_rijeka.npy` 0.93 GiB (index 736),
  plus 224px/12sl, 336px/20sl, and 336px/20sl band-0.1–0.9 variants, all under
  `processed/rsna_knee_abnormality/`.
- Labels: `labels/rijeka_acl.csv`, 736 rows, ONE column (ACL + confidence) — expert double-blind
  image reads mapped to levels 0.08/0.82/0.94 (`rijeka_corpus.py:77-97`).

**MRNet (Stanford 2018) — decoded pixels; labels arrived the same day via Claude (knee-mfd8).**
- Raw: `raw/mrnet/`, 7.3 GB extracted npy; decoded `images_336px_12sl_mrnet.npy` 4.7 GiB,
  1,250 exams (1,130 train + 120 valid; test withheld — `core/data/corpus/mrnet_corpus.py:22-24`),
  filling THREE slots (SAG_FLUID_FS, AX_FLUID_FS, COR_T1 — `core/data/corpus/slots.py:176-195`).
- `labels/mrnet_acl.csv` did not exist at survey time; Claude sourced the public Stanford label
  tables and wrote it (1,248 rows, 21.0% positive) the same day.

**Merged caches (competition 4,407 + Rijeka 736 = 5,143 exactly, every index measured):**
`images_336px_12sl_merged.npy` 38.9 GiB (the one `merged_corpus`/`merged_blend` trained on),
224px/12sl, 336px/20sl (64.9 GiB, `merged_span_cf`), 336px/20sl band (64.9 GiB, `merged_dense_cf`),
plus masks and smoke variants. Label tables: `merged_labels.csv` (5,143 rows, column union) and
`merged_blend_v1.csv` (our rows three-reader blend, Rijeka rows ACL alone). A THREE-corpus cache
(+MRNet, 6,393 studies) was built the same afternoon (Claude, knee-m179 queued).

Fold safety: foreign identifiers are `TRAIN_ONLY = -2` — they train under every fold and are never
scored (`core/data/corpus/folds.py:19-33`); verified live: 5,143 loaded, 4,234 train / 909 holdout
(`experiments/README.md:3638`). The foreign-in-sample guard for foreign MODELS is
`kneemri/evaluation/foreign_overlap.py` (raptor read 896/909 in-sample, board.jsonl).

## What the merged work measured offline

The corpus thread runs through the ledger, not a single file:

- `merged_corpus_s2026` 0.8545 vs `lr4e-5_ws` 0.8525, **+0.0020**, inside the 0.0062 floor — the
  SPLIT is the result: deficit six +0.0061 / surplus six −0.0022; Lateral Meniscus **+0.0196\***,
  ACL +0.0161, Synovitis −0.0206\* (`experiments/README.md:331`, `:3630`).
- `merged_blend_s2026` 0.8589, **+0.0063** (cleared its pre-registered gate by 0.0001); our-cut
  +0.0160 vs published-cut +0.0031 (5×); seat value +0.0014–0.0020, in every best subset 1–6
  (ledger:3858-3907; `experiments/arms/merged_blend.sh`).
- `merged_blend_cf` 0.8597; `merged_dense_cf` 0.8569 (−0.0028 vs merged_blend_cf — the corpus gain
  does NOT carry to the dense corner; geometry gain sign-flips across corpora, ledger:4617,
  4653-4660); `merged_span_cf` 0.8618 — second-best solo, largest single greedy jump +0.0075
  (ledger:5624-5640); `merged_labelwise_cf` 0.8583; `merged_distill_cf` +0.0000 seat
  (ledger:6535-6570).
- Board: one tick — blendseat 0.939 vs votefit 0.938, "safe at serving" and no more
  (`comp/champion.json`, THE CORPUS AXIS principle).

## The two corpus mechanisms

Named at `experiments/README.md:331` and re-split at `:3847-3856`:

1. **Encoder transfer** from a second site/scanner population — evidence: Lateral Meniscus +0.0196
   on a finding Rijeka never labels, so the gain cannot be supervision.
2. **Direct ACL supervision** from expert image reads — evidence: ACL +0.0161 on the one column
   Rijeka supervises; "it must not be counted toward the transfer claim" (ledger:3852-3855).

The 7h `merged_blend_ff` refit that would separate them needs its own pre-registered band
(`comp/champion.json`). A blind merged refit exists (`experiments/arms/merged_span_ff.sh`,
knee-zqn8) but has no holdout by construction — it separates nothing offline.

## Builder

`python -m core.data.corpus.build_merged` (`build_merged.py:1-74`) — merges any decoded scheme as
`--extra-scheme` onto the canonical cache; refuses merged-as-source. Ingestors:
`build_rijeka --archive`, `build_mrnet --archive`. `FOREIGN_SCHEMES = (rijeka, mrnet, merged)`
(`slots.py:222-234`) refuses canonical-scheme foreign decodes — the knee-48br 59.7 GB overwrite
accident is recorded there.
