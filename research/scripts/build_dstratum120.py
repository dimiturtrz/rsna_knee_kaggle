import hashlib
import json
import os

import numpy as np
import pandas as pd

COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
        "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ["Medial OA", "Lateral OA", "PF OA"]
SEED = 20260911
A = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/audit400"
LBL = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/"
N_TOTAL, OA_FRAC = 120, 0.5

PATHS = {
    "ours_v2": LBL + "report_labels_v2.csv",
    "flight": "D:/data/volumetric/mri/raw/knee_label_tables/flight0234_rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv",
    "lixin": "D:/data/volumetric/mri/raw/knee_label_tables/lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv",
    "steven_blend": "D:/data/volumetric/mri/raw/knee_label_tables/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    "steven_full": "D:/data/volumetric/mri/raw/knee_label_tables/stevenleehans_rsna-knee-llm-report-labels/llm_labels_full.csv",
}
raw = {k: pd.read_csv(p).set_index("StudyInstanceUID")[COLS] for k, p in PATHS.items()}
uids = sorted(set.intersection(*[set(f.index) for f in raw.values()]))
B = np.stack([(f.loc[uids] >= 0.5).astype(int).to_numpy() for f in raw.values()])
disc = ~((B == B[0]).all(axis=0))
v2b = B[0]

prior = set()
for pack in ("adjudication_key.csv", "holdout469/adjudication_key.csv",
             "imagesel56/adjudication_key.csv", "denom60/adjudication_key.csv",
             "balance56/adjudication_key.csv"):
    k = pd.read_csv(f"{A}/{pack}")
    col = "uid" if "uid" in k.columns else "study_uid"
    prior |= set(zip(k[col], k["column"]))

cells = [(i, j) for i, j in np.argwhere(disc) if (uids[i], COLS[j]) not in prior]
df = pd.DataFrame(cells, columns=["si", "cj"])
df["column"] = [COLS[j] for _, j in cells]
df["uid"] = [uids[i] for i, _ in cells]
df["direction"] = ["neg" if v2b[i, j] == 1 else "pos" for i, j in cells]
df["is_oa"] = df.column.isin(OA)

rng = np.random.default_rng(SEED)
parts = []
for direction in ("neg", "pos"):
    pool = df[df.direction == direction]
    n_oa = int(N_TOTAL / 2 * OA_FRAC)
    oa_pick = pool[pool.is_oa].iloc[rng.choice(pool.is_oa.sum(), size=min(n_oa, int(pool.is_oa.sum())), replace=False)]
    rest = pool[~pool.is_oa].iloc[rng.choice((~pool.is_oa).sum(), size=N_TOTAL // 2 - len(oa_pick), replace=False)]
    parts += [oa_pick, rest]
key = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

reports = pd.read_csv("D:/data/volumetric/mri/raw/rsna_knee_abnormality/train.csv").set_index("StudyInstanceUID")["Report"]
key["report"] = key["uid"].map(reports)
assert key["report"].notna().all()
key["row_id"] = ["d%04d" % k for k in range(len(key))]
pack = key[["row_id", "column", "report"]]
assert not pack["report"].str.contains(r"1\.2\.826", regex=True).any()
out = f"{A}/dstratum120"
os.makedirs(out, exist_ok=True)
pack.to_csv(f"{out}/adjudication_pack.csv", index=False)
for k, ix in enumerate(np.array_split(np.arange(len(pack)), 6), 1):
    pack.iloc[ix].to_csv(f"{out}/adjudication_pack_{k}of6.csv", index=False)
key.drop(columns=["report"]).to_csv(f"{out}/adjudication_key.csv", index=False)

natural = df.column.value_counts(normalize=True).round(3).to_dict()
manifest = dict(
    seed=SEED, n_total=len(pack),
    design="120 D-stratum cells, direction-balanced 60/60, OA columns DELIBERATELY over-sampled to 50% "
           "of each direction half (natural D-stratum OA share below, so print both the OA-weighted "
           "rate and the reweighted corpus estimate)",
    natural_oa_share=natural,
    strata={f"{d}|oa={o}": int(n) for (d, o), n in key.groupby(["direction", "is_oa"]).size().items()},
    preregistered_branches="D v2-wrong <=0.10 -> share moves toward half, mask's bigger-half argument weakens; "
                           "0.10-0.25 -> current reading holds, quote share with interval; >0.25 -> dissent "
                           "dominance grows",
    pack_sha256=hashlib.sha256(open(f"{out}/adjudication_pack.csv", "rb").read()).hexdigest()[:16],
    key_sha256=hashlib.sha256(open(f"{out}/adjudication_key.csv", "rb").read()).hexdigest()[:16])
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1, default=str))
