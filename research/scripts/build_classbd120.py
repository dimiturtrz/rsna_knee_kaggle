"""Build audit400/classbd120: the class-B DISAGREEMENT-stratum adjudication
batch, successor to classb120 (letter M392 §4, task 1).

Identical construction to research/scripts/build_classb120.py except:
  * stratum is the COMPLEMENT: the five tables (v2 + flight + lixin +
    steven_blend + steven_full, binarised >= 0.5) do NOT unanimously agree
  * the sealed key carries ALL FIVE tables' soft values per cell
    (soft_flight, soft_lixin, soft_steven_blend, soft_steven_full beside
    v2_soft) so the post-adjudication error can be scored against the majority
    sign and against each table separately — the per-table-carrier analysis
  * exclusions cover SEVEN prior packs (classb120 added), asserted 0 overlap
  * row_id prefix "g" (c/d/b/h/i/r taken; d already collides between denom60
    and dstratum120 — recorded in classb120's builder)

The key is written sealed. Nothing here scores anything against it.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/build_classbd120.py
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
DRAW = [("MCL", 40), ("Lateral OA", 40), ("Fracture", 40)]
SEED = 20260911
A = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/audit400"
LBL = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/"
NAME = "classbd120"
CHUNKS = 6
TABLES = ["ours_v2", "flight", "lixin", "steven_blend", "steven_full"]

reports = pd.read_csv(
    "D:/data/volumetric/mri/raw/rsna_knee_abnormality/train.csv"
).set_index("StudyInstanceUID")["Report"]

PATHS = {
    "ours_v2": LBL + "report_labels_v2.csv",
    "flight": "D:/data/volumetric/mri/raw/knee_label_tables/flight0234_rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv",
    "lixin": "D:/data/volumetric/mri/raw/knee_label_tables/lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv",
    "steven_blend": "D:/data/volumetric/mri/raw/knee_label_tables/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    "steven_full": "D:/data/volumetric/mri/raw/knee_label_tables/stevenleehans_rsna-knee-llm-report-labels/llm_labels_full.csv",
}
raw = {k: pd.read_csv(p).set_index("StudyInstanceUID")[COLS] for k, p in PATHS.items()}
uids = sorted(set.intersection(*[set(f.index) for f in raw.values()]))
R = np.stack([f.loc[uids].to_numpy() for f in raw.values()])
B = (R >= 0.5).astype(int)
disc = ~((B == B[0]).all(axis=0))
majority_pos = (B.sum(axis=0) >= 3)

prior = set()
PRIOR_PACKS = ("adjudication_key.csv", "denom60/adjudication_key.csv",
               "balance56/adjudication_key.csv", "dstratum120/adjudication_key.csv",
               "holdout469/adjudication_key.csv", "imagesel56/adjudication_key.csv",
               "classb120/adjudication_key.csv")
for pack in PRIOR_PACKS:
    k = pd.read_csv(f"{A}/{pack}")
    col = "uid" if "uid" in k.columns else "study_uid"
    prior |= set(zip(k[col], k["column"]))

print(f"population: {len(uids)} studies in the five-way table intersection "
      f"(corpus-wide — same as classb120/denom60)")

rng = np.random.default_rng(SEED)
drawn = []
strata = {}
for col, n in DRAW:
    j = COLS.index(col)
    cells = np.argwhere(disc[:, j])
    cells = [int(i[0]) for i in cells if (uids[int(i[0])], col) not in prior]
    n_pos = sum(1 for i in cells if majority_pos[i, j])
    strata[col] = (len(cells), n_pos, len(cells) - n_pos)
    print(f"{col}: disagreement stratum {len(cells)} cells "
          f"(majority-positive {n_pos}, majority-negative {len(cells) - n_pos})")
    assert len(cells) >= n, f"{col} stratum too thin: {len(cells)} < {n}"
    take = rng.choice(len(cells), size=n, replace=False)
    got_pos = sum(1 for t in take if majority_pos[cells[int(t)], j])
    print(f"  drew {n}: {got_pos} majority-positive, {n - got_pos} majority-negative")
    for t in take:
        i = cells[int(t)]
        drawn.append((i, j))

key = pd.DataFrame(drawn, columns=["si", "cj"])
key["stratum"] = "D_random"
key["uid"] = [uids[int(i)] for i, _ in drawn]
key["column"] = [COLS[int(j)] for _, j in drawn]
key["v2_soft"] = [float(R[0][int(i), int(j)]) for i, j in drawn]
key["soft_flight"] = [float(R[1][int(i), int(j)]) for i, j in drawn]
key["soft_lixin"] = [float(R[2][int(i), int(j)]) for i, j in drawn]
key["soft_steven_blend"] = [float(R[3][int(i), int(j)]) for i, j in drawn]
key["soft_steven_full"] = [float(R[4][int(i), int(j)]) for i, j in drawn]
key["majority_sign"] = [int(majority_pos[int(i), int(j)]) for i, j in drawn]

overlap = [(u, c) for u, c in zip(key.uid, key.column) if (u, c) in prior]
print(f"overlap vs the seven prior packs: {len(overlap)} (must be 0)")
assert not overlap
assert len(set(zip(key.uid, key.column))) == len(key)

key["report"] = key["uid"].map(reports)
assert key["report"].notna().all()
key["row_id"] = ["g" + "%04d" % k for k in range(len(key))]
pack = key[["row_id", "column", "report"]].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
pack.columns = ["row_id", "finding", "report"]
assert not pack["report"].str.contains(r"1\.2\.826", regex=True).any()

out = f"{A}/{NAME}"
os.makedirs(out, exist_ok=True)
pack.to_csv(f"{out}/adjudication_pack.csv", index=False)
for k, ix in enumerate(np.array_split(np.arange(len(pack)), CHUNKS), 1):
    pack.iloc[ix].to_csv(f"{out}/adjudication_pack_{k}of{CHUNKS}.csv", index=False)
key.drop(columns=["report"]).to_csv(f"{out}/adjudication_key.csv", index=False)

counts = pack.finding.value_counts().to_dict()
chunk_cols = [pack.iloc[ix].finding.value_counts().to_dict()
              for ix in np.array_split(np.arange(len(pack)), CHUNKS)]
print(f"pack composition: {counts}")
for k, cc in enumerate(chunk_cols, 1):
    print(f"  chunk {k}: {cc}")

manifest = dict(
    seed=SEED,
    n_total=len(pack),
    preregistered_expectation=(
        "v2-wrong on the disagreement stratum ~0.138-0.156 (the audit400 root "
        "D-stratum comparator on record), against classb120's unanimous-stratum "
        "0.033; per-table-carrier: if ONE table's sign is wrong on most "
        "contradicted cells, that names the table to drop"
    ),
    pack_sha256=hashlib.sha256(open(f"{out}/adjudication_pack.csv", "rb").read()).hexdigest()[:16],
    key_sha256=hashlib.sha256(open(f"{out}/adjudication_key.csv", "rb").read()).hexdigest()[:16],
    preregistration=dict(
        fork=("SUPERVISION: the class-B hole concentrates where the target is "
              "noisy — v2 (and the majority) wrong at D-stratum rates far above "
              "the unanimous 0.033 — vs TASK DIFFICULTY: labels in the "
              "disagreement stratum are as good as at unanimity and the finding "
              "is genuinely hard. The verdict-vs-v2 and verdict-vs-majority "
              "rates on these 120 cells price the fork for MCL, Lateral OA and "
              "Fracture at n=40 each"),
        stratum=("five-table DISAGREEMENT: NOT unanimously agreed (v2 + flight + "
                 "lixin + steven_blend + steven_full, binarised >= 0.5); same "
                 "population and tables as classb120"),
        population=f"{len(uids)} studies, five-table UID intersection, corpus-wide",
        per_column_stratum_sizes={
            c: {"total": t, "majority_positive": p, "majority_negative": g}
            for c, (t, p, g) in strata.items()},
        draw=("uniform within each column's disagreement stratum, one rng stream "
              "in MCL -> Lateral OA -> Fracture order; no member signal; 0 "
              "overlap asserted against " + ", ".join(PRIOR_PACKS)),
        per_table_carrier_plan=(
            "the sealed key carries all five tables' soft values and the "
            "majority sign per cell. After adjudication: score each table's "
            "binarised sign against the verdict per cell; a table whose sign is "
            "wrong on most contradicted cells is the carrier and is named for "
            "dropping from blend tables; also score majority-sign-vs-verdict "
            "and v2-vs-verdict to separate 'v2 wrong, majority right' from "
            "'the whole five-way split is noise'"),
        chunking="one full-pack shuffle (random_state=SEED) then sequential np.array_split into 6 chunks of 20, as classb120/denom60",
    ),
)
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(f"{NAME} {len(pack)} -> {out}")
