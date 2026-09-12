import hashlib
import json
import os

import numpy as np
import pandas as pd

COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
        "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = 20260911
A = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/audit400"
LBL = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/"
reports = pd.read_csv("D:/data/volumetric/mri/raw/rsna_knee_abnormality/train.csv").set_index("StudyInstanceUID")["Report"]

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

prior = set()
for pack in ("adjudication_key.csv", "holdout469/adjudication_key.csv", "imagesel56/adjudication_key.csv"):
    k = pd.read_csv(f"{A}/{pack}")
    col = "uid" if "uid" in k.columns else "study_uid"
    prior |= set(zip(k[col], k["column"]))

rng = np.random.default_rng(SEED)
idx = np.argwhere(unan)
idx = np.array([c for c in idx if (uids[c[0]], COLS[c[1]]) not in prior])
denom = idx[rng.choice(len(idx), size=60, replace=False)]

key1 = pd.DataFrame(denom, columns=["si", "cj"])
key1["stratum"] = "U_random"
key1["uid"] = [uids[int(i)] for i, _ in denom]
key1["column"] = [COLS[int(j)] for _, j in denom]
key1["v2_soft"] = [float(R[0][int(i), int(j)]) for i, j in denom]

sel = pd.read_csv(LBL + "v2_lone_dissent.csv")
neg = sel[sel.v2_value == 1]
pos = sel[sel.v2_value == 0]
neg = neg.iloc[rng.choice(len(neg), size=28, replace=False)]
pos = pos.iloc[rng.choice(len(pos), size=28, replace=False)]
key2 = pd.concat([neg.assign(stratum="neg_direction"), pos.assign(stratum="pos_direction")]).reset_index(drop=True)

for name, key, chunks in (("denom60", key1, 4), ("balance56", key2, 4)):
    key = key.copy()
    if "uid" not in key.columns:
        key = key.rename(columns={"study_uid": "uid"})
    key["report"] = key["uid"].map(reports)
    assert key["report"].notna().all()
    key["row_id"] = [name[0] + "%04d" % k for k in range(len(key))]
    pack = key[["row_id", "column", "report"]].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pack.columns = ["row_id", "finding", "report"]
    assert not pack["report"].str.contains(r"1\.2\.826", regex=True).any()
    out = f"{A}/{name}"
    os.makedirs(out, exist_ok=True)
    pack.to_csv(f"{out}/adjudication_pack.csv", index=False)
    for k, ix in enumerate(np.array_split(np.arange(len(pack)), chunks), 1):
        pack.iloc[ix].to_csv(f"{out}/adjudication_pack_{k}of{chunks}.csv", index=False)
    key.drop(columns=["report"]).to_csv(f"{out}/adjudication_key.csv", index=False)
    exp = ("U-stratum v2-wrong ~1.5% (18% figure) vs ~7% (unflagU pilot)" if name == "denom60"
           else "negative-direction substitutions correct at high rate (over-call repair); "
                "positive-direction is the over-correction suspect (D345's 31, D360's 28-of-28)")
    json.dump(dict(seed=SEED, n_total=len(pack), preregistered_expectation=exp,
                   pack_sha256=hashlib.sha256(open(f"{out}/adjudication_pack.csv", "rb").read()).hexdigest()[:16],
                   key_sha256=hashlib.sha256(open(f"{out}/adjudication_key.csv", "rb").read()).hexdigest()[:16]),
              open(f"{out}/manifest.json", "w"), indent=1)
    print(name, len(pack), "->", out)
