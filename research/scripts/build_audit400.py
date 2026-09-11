import hashlib
import json

import numpy as np
import pandas as pd

COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
        "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PATHS = {
    "ours_v2": "D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/report_labels_v2.csv",
    "flight": "D:/data/volumetric/mri/raw/knee_label_tables/flight0234_rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv",
    "lixin": "D:/data/volumetric/mri/raw/knee_label_tables/lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv",
    "steven_blend": "D:/data/volumetric/mri/raw/knee_label_tables/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    "steven_full": "D:/data/volumetric/mri/raw/knee_label_tables/stevenleehans_rsna-knee-llm-report-labels/llm_labels_full.csv",
}
SEED = 20260911
N_UNAN, N_DISC = 336, 64
OUT = "C:/Users/User/AppData/Local/Temp/claude/D--personal-projects-rsna-knee-kaggle/3dc4cc46-abdd-4d38-b0dc-f1461a5b836a/scratchpad/audit400"

raw = {k: pd.read_csv(p).set_index("StudyInstanceUID")[COLS] for k, p in PATHS.items()}
uids = sorted(set.intersection(*[set(f.index) for f in raw.values()]))
R = np.stack([f.loc[uids].to_numpy() for f in raw.values()])
B = (R >= 0.5).astype(int)
unan = (B == B[0]).all(axis=0)
uid_arr = np.array(uids)

rng = np.random.default_rng(SEED)
rows = []

# unanimous strata: column x agreed-sign, self-weighting
strata = []
for j, col in enumerate(COLS):
    for sign in (1, 0):
        mask = unan[:, j] & (B[0][:, j] == sign)
        idx = np.where(mask)[0]
        strata.append((col, sign, idx))
total = sum(len(s[2]) for s in strata)
for col, sign, idx in strata:
    n = max(1, int(round(N_UNAN * len(idx) / total)))
    n = min(n, len(idx))
    for i in rng.choice(idx, size=n, replace=False):
        rows.append(dict(stratum="U", column=col, sign=sign, study_idx=int(i)))

# discordant sentinel: per column proportional, sign = ours_v2's as-trained call
disc_counts = [(col, int((~unan[:, j]).sum())) for j, col in enumerate(COLS)]
disc_total = sum(c for _, c in disc_counts)
for j, col in enumerate(COLS):
    n = max(1, int(round(N_DISC * disc_counts[j][1] / disc_total)))
    idx = np.where(~unan[:, j])[0]
    n = min(n, len(idx))
    for i in rng.choice(idx, size=n, replace=False):
        rows.append(dict(stratum="D", column=col, sign=int(B[0][i, j]), study_idx=int(i)))

key = pd.DataFrame(rows)
key["uid"] = uid_arr[key["study_idx"].to_numpy()]
for ti, tname in enumerate(PATHS):
    key[f"soft_{tname}"] = R[ti][key["study_idx"].to_numpy(), key["column"].map({c: k for k, c in enumerate(COLS)}).to_numpy()]
key["row_id"] = ["r%04d" % k for k in range(len(key))]

reports = pd.read_csv("D:/data/volumetric/mri/raw/rsna_knee_abnormality/train.csv").set_index("StudyInstanceUID")["Report"]
key["report"] = key["uid"].map(reports)
assert key["report"].notna().all()

pack = key[["row_id", "column", "report"]].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
pack.columns = ["row_id", "finding", "report"]

# leak tests
assert pack["row_id"].is_unique and len(pack) == len(key)
assert not pack["report"].str.contains(r"1\.2\.826", regex=True).any()
assert set(pack.columns) == {"row_id", "finding", "report"}

import os
os.makedirs(OUT, exist_ok=True)
pack.to_csv(f"{OUT}/adjudication_pack.csv", index=False)
chunks = np.array_split(np.arange(len(pack)), 8)
for k, ix in enumerate(chunks, 1):
    pack.iloc[ix].to_csv(f"{OUT}/adjudication_pack_{k}of8.csv", index=False)
key.drop(columns=["report"]).to_csv(f"{OUT}/adjudication_key.csv", index=False)

manifest = dict(seed=SEED, n_total=len(pack),
                n_unanimous=int((key.stratum == "U").sum()), n_discordant=int((key.stratum == "D").sum()),
                unanimity_definition="all five tables binarised >=0.5 agree",
                corpus_unanimous_cells=int(unan.sum()), corpus_cells=int(unan.size),
                corpus_unanimity=round(float(unan.mean()), 4),
                pack_sha256=hashlib.sha256(open(f"{OUT}/adjudication_pack.csv", "rb").read()).hexdigest()[:16],
                key_sha256=hashlib.sha256(open(f"{OUT}/adjudication_key.csv", "rb").read()).hexdigest()[:16])
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
print(key.groupby(["stratum", "column"]).size().to_string())
