import hashlib
import json
import os

import numpy as np
import pandas as pd

COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
        "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = 20260911
OUT = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/audit400/imagesel56"
LBL = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/"
MEMBERS = ["band_span_s2026", "chainfour_s2026", "foreign_labels_dense_s2026", "recal_dense_s2026"]

ref = np.load("D:/data/volumetric/mri/models/rsna_knee_abnormality/band_span_s2026.holdout.npz")
studies = ref["studies"].astype(str)
P = np.stack([np.load(f"D:/data/volumetric/mri/models/rsna_knee_abnormality/{s}.holdout.npz")["predictions"] for s in MEMBERS])
npos = (P >= 0.5).astype(int).sum(axis=0)
v2 = pd.read_csv(LBL + "report_labels_v2.csv").set_index("StudyInstanceUID")[COLS].loc[studies]
v2b = (v2.to_numpy() >= 0.5).astype(int)
flag = (((npos >= 3) & (v2b == 0)) | ((npos <= 1) & (v2b == 1)))

FT = {"flight": "D:/data/volumetric/mri/raw/knee_label_tables/flight0234_rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv",
      "lixin": "D:/data/volumetric/mri/raw/knee_label_tables/lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv",
      "steven_blend": "D:/data/volumetric/mri/raw/knee_label_tables/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
      "steven_full": "D:/data/volumetric/mri/raw/knee_label_tables/stevenleehans_rsna-knee-llm-report-labels/llm_labels_full.csv"}
B = np.stack([(pd.read_csv(p).set_index("StudyInstanceUID")[COLS].loc[studies] >= 0.5).astype(int).to_numpy()
              for p in [LBL + "report_labels_v2.csv", *FT.values()]])
unan = (B == B[0]).all(axis=0)
margin = np.abs(P.mean(axis=0) - v2.to_numpy())

rng = np.random.default_rng(SEED)
rows = []
U, D = flag & unan, flag & ~unan
iu = np.argwhere(U)
iu = iu[np.argsort(-margin[U])]
third = len(iu) // 3
for stratum, idx in (("flagU_top", iu[:third]), ("flagU_mid", iu[third:2 * third])):
    pick = idx[rng.choice(len(idx), size=min(14, len(idx)), replace=False)]
    rows += [("flagU_top" if stratum == "flagU_top" else "flagU_mid", i, j) for i, j in pick]
idr = np.argwhere(D)
rows += [("flagD", i, j) for i, j in idr[rng.choice(len(idr), size=14, replace=False)]]
unflagU = np.argwhere(~flag & unan)
rows += [("unflagU", i, j) for i, j in unflagU[rng.choice(len(unflagU), size=14, replace=False)]]

key = pd.DataFrame(rows, columns=["stratum", "si", "cj"])
key["uid"] = [studies[int(i)] for _, i, _ in rows]
key["column"] = [COLS[int(j)] for _, _, j in rows]
key["v2_soft"] = [float(v2.to_numpy()[int(i), int(j)]) for _, i, j in rows]
key["member_mean_soft"] = [float(P[:, int(i), int(j)].mean()) for _, i, j in rows]
key["margin"] = [float(margin[int(i), int(j)]) for _, i, j in rows]

reports = pd.read_csv("D:/data/volumetric/mri/raw/rsna_knee_abnormality/train.csv").set_index("StudyInstanceUID")["Report"]
key["report"] = key["uid"].map(reports)
assert key["report"].notna().all()
key["row_id"] = ["i%04d" % k for k in range(len(key))]
pack = key[["row_id", "column", "report"]].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
pack.columns = ["row_id", "finding", "report"]
assert not pack["report"].str.contains(r"1\.2\.826", regex=True).any()

os.makedirs(OUT, exist_ok=True)
pack.to_csv(f"{OUT}/adjudication_pack.csv", index=False)
for k, ix in enumerate(np.array_split(np.arange(len(pack)), 4), 1):
    pack.iloc[ix].to_csv(f"{OUT}/adjudication_pack_{k}of4.csv", index=False)
key.drop(columns=["report"]).to_csv(f"{OUT}/adjudication_key.csv", index=False)
manifest = dict(seed=SEED, n_total=len(pack), strata=key.stratum.value_counts().to_dict(),
                selector=">=3-of-4 member consensus vs v2, members: band_span, chainfour, foreign_labels_dense, recal_dense",
                preregistered_expectation="flagU_top precision ~0.3-0.5 (n=3 prior), flagD high (dissent stratum), "
                "unflagU v2-wrong ~1.5% (the U-stratum base, the recall denominator)",
                pack_sha256=hashlib.sha256(open(f"{OUT}/adjudication_pack.csv", "rb").read()).hexdigest()[:16],
                key_sha256=hashlib.sha256(open(f"{OUT}/adjudication_key.csv", "rb").read()).hexdigest()[:16])
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
