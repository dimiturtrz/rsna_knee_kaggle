"""Build audit400/classb120: the commissioned successor unanimous-stratum gold
batch, column-weighted to MCL / Lateral OA / Fracture (letter M391 §2).

Construction replicates build_denom60_balance56.py EXACTLY where the commission
does not say otherwise:
  * same five-way unanimous stratum: v2 + flight + lixin + steven_blend +
    steven_full, all binarised >= 0.5, all five agreeing (B == B[0]).all(axis=0)
  * same population: the sorted UID intersection of the five tables (corpus-wide,
    NOT the train split — stated in the manifest)
  * same key schema (si, cj, stratum, uid, column, v2_soft, row_id), same pack
    schema (row_id, finding, report), same report source (raw train.csv),
    same UID-leak assert, same one-shuffle-then-sequential-split chunking
    (denom60: sample(frac=1.0, random_state=SEED) then np.array_split), same
    manifest fields (seed, n_total, preregistered_expectation, pack/key sha256)
  * same SEED = 20260911 for lineage
Commission deltas: 120 cells = 40 MCL + 40 Lateral OA + 40 Fracture, uniform
WITHIN each column's stratum (no member signal anywhere in the draw); no
(uid, column) overlap with SIX prior packs (root, denom60, balance56,
dstratum120, holdout469, imagesel56 — denom60's builder excluded the first
three; the commission names the other three; all six are checked and the count
is asserted 0); 6 chunks of 20; row_id prefix "c" (r/d/b/h/i taken — d already
collides between denom60 and dstratum120, recorded here).

The key is written sealed. Nothing here scores anything against it.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/build_classb120.py
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
NAME = "classb120"
CHUNKS = 6

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
unan = (B == B[0]).all(axis=0)
v2b = B[0]

prior = set()
PRIOR_PACKS = ("adjudication_key.csv", "denom60/adjudication_key.csv",
               "balance56/adjudication_key.csv", "dstratum120/adjudication_key.csv",
               "holdout469/adjudication_key.csv", "imagesel56/adjudication_key.csv")
for pack in PRIOR_PACKS:
    k = pd.read_csv(f"{A}/{pack}")
    col = "uid" if "uid" in k.columns else "study_uid"
    prior |= set(zip(k[col], k["column"]))

print(f"population: {len(uids)} studies in the five-way table intersection "
      f"(corpus-wide, NOT the train split — same population denom60 used)")

rng = np.random.default_rng(SEED)
drawn = []
strata = {}
for col, n in DRAW:
    j = COLS.index(col)
    cells = np.argwhere(unan[:, j])
    cells = [int(i[0]) for i in cells if (uids[int(i[0])], col) not in prior]
    n_pos = sum(1 for i in cells if v2b[i, j] == 1)
    strata[col] = (len(cells), n_pos, len(cells) - n_pos)
    print(f"{col}: unanimous stratum {len(cells)} cells "
          f"(agreement-positive {n_pos}, agreement-negative {len(cells) - n_pos})")
    assert len(cells) >= n, f"{col} stratum too thin: {len(cells)} < {n}"
    take = rng.choice(len(cells), size=n, replace=False)
    got_pos = sum(1 for t in take if v2b[cells[int(t)], j] == 1)
    print(f"  drew {n}: {got_pos} agreement-positive, {n - got_pos} agreement-negative")
    for t in take:
        i = cells[int(t)]
        drawn.append((i, j))

key = pd.DataFrame(drawn, columns=["si", "cj"])
key["stratum"] = "U_random"
key["uid"] = [uids[int(i)] for i, _ in drawn]
key["column"] = [COLS[int(j)] for _, j in drawn]
key["v2_soft"] = [float(R[0][int(i), int(j)]) for i, j in drawn]

overlap = [(u, c) for u, c in zip(key.uid, key.column) if (u, c) in prior]
print(f"overlap vs the six prior packs: {len(overlap)} (must be 0)")
assert not overlap
assert len(set(zip(key.uid, key.column))) == len(key), "duplicate cells in the draw"

key["report"] = key["uid"].map(reports)
assert key["report"].notna().all()
key["row_id"] = ["c" + "%04d" % k for k in range(len(key))]
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


def wilson(k_, n_, z=1.96):
    p = k_ / n_
    d = 1 + z * z / n_
    c = (p + z * z / (2 * n_)) / d
    h = z * np.sqrt(p * (1 - p) / n_ + z * z / (4 * n_ * n_)) / d
    return round(float(c - h), 3), round(float(c + h), 3)


exp_k = round(0.033 * 180)
manifest = dict(
    seed=SEED,
    n_total=len(pack),
    preregistered_expectation=(
        "pooled eps_u = denom60's 0.033 (2/60 [0.009, 0.114]); pooled at n=180 "
        "the null expects ~6/180, Wilson " + str(wilson(exp_k, 180))
    ),
    pack_sha256=hashlib.sha256(open(f"{out}/adjudication_pack.csv", "rb").read()).hexdigest()[:16],
    key_sha256=hashlib.sha256(open(f"{out}/adjudication_key.csv", "rb").read()).hexdigest()[:16],
    preregistration=dict(
        purpose=("widen denom60's unanimous-stratum denominator on the class-B "
                 "columns; test whether MCL's unanimous error is materially "
                 "above 0.033 (the label wall wearing a small numerator)"),
        stratum=("five-way unanimous: v2 + flight + lixin + steven_blend + "
                 "steven_full, binarised >= 0.5, all agree; identical definition "
                 "and tables to build_denom60_balance56.py so the batches pool"),
        population=f"{len(uids)} studies, five-table UID intersection, corpus-wide",
        per_column_stratum_sizes={
            c: {"total": t, "agreement_positive": p, "agreement_negative": g}
            for c, (t, p, g) in strata.items()},
        draw=("uniform within each column's stratum, one rng stream in "
              "MCL -> Lateral OA -> Fracture order; no member signal; exclusions "
              "verified 0 overlap against " + ", ".join(PRIOR_PACKS)),
        per_column_hypothesis=(
            "MCL unanimous-stratum v2-error materially ABOVE 0.033 would say part "
            "of MCL's 0.121 best-to-ceiling hole is the class-C label wall at a "
            "small numerator; Lateral OA and Fracture are the other two "
            "class-B-headline columns and are measured alongside at n=40 each"),
        decision_supported=(
            "if MCL's rate is consistent with 0.033, MCL's 0.121 hole stands as "
            "modelling; if materially above, the ceiling for MCL drops and the "
            "hole is part wall. Pooled with denom60 at n=180 the eps_u interval "
            "tightens from [0.009, 0.114] to the manifest's Wilson bound, which "
            "every class-B ceiling in research/scripts/column_headroom.py "
            "consumes as the unanimous-stratum term"),
        chunking="one full-pack shuffle (random_state=SEED) then sequential np.array_split into 6 chunks of 20, replicating denom60's procedure",
    ),
)
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(f"{NAME} {len(pack)} -> {out}")
