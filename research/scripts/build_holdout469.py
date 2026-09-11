import hashlib
import json
import os

import numpy as np
import pandas as pd

SEED = 20260911
OUT = "C:/Users/User/AppData/Local/Temp/claude/D--personal-projects-rsna-knee-kaggle/3dc4cc46-abdd-4d38-b0dc-f1461a5b836a/scratchpad/audit400/holdout469"

sel = pd.read_csv("D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/v2_lone_dissent.csv")
ref = np.load("D:/models/rsna_knee_abnormality/band_span_s2026.holdout.npz")
studies = set(ref["studies"].astype(str))
key = sel[sel.study_uid.isin(studies)].reset_index(drop=True)
assert len(key) == 469, len(key)

reports = pd.read_csv("D:/data/volumetric/mri/raw/rsna_knee_abnormality/train.csv").set_index("StudyInstanceUID")["Report"]
key["report"] = key["study_uid"].map(reports)
assert key["report"].notna().all()

key["row_id"] = ["h%04d" % k for k in range(len(key))]
pack = key[["row_id", "column", "report"]].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
pack.columns = ["row_id", "finding", "report"]

assert pack["row_id"].is_unique
assert not pack["report"].str.contains(r"1\.2\.826", regex=True).any()
assert set(pack.columns) == {"row_id", "finding", "report"}

os.makedirs(OUT, exist_ok=True)
pack.to_csv(f"{OUT}/adjudication_pack.csv", index=False)
for k, ix in enumerate(np.array_split(np.arange(len(pack)), 10), 1):
    pack.iloc[ix].to_csv(f"{OUT}/adjudication_pack_{k}of10.csv", index=False)
key.drop(columns=["report"]).to_csv(f"{OUT}/adjudication_key.csv", index=False)

manifest = dict(
    seed=SEED, n_total=len(pack),
    selector="v2_lone_dissent.csv cells intersected with the 909-study fold-0 holdout",
    preregistered_expectation="v2 is RIGHT in ~25% of cells (~117 of 469) and WRONG in ~75% (~352), "
    "from the two independent purity readings (9/12 pack, 24/32 gold-58)",
    rubric="does the report AFFIRM the named finding: 1 = affirmed at any severity; 0 = explicitly "
    "negated OR not mentioned",
    pack_sha256=hashlib.sha256(open(f"{OUT}/adjudication_pack.csv", "rb").read()).hexdigest()[:16],
    key_sha256=hashlib.sha256(open(f"{OUT}/adjudication_key.csv", "rb").read()).hexdigest()[:16],
)
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
