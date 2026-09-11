import json
import sys

import pandas as pd

OUT = "C:/Users/User/AppData/Local/Temp/claude/D--personal-projects-rsna-knee-kaggle/3dc4cc46-abdd-4d38-b0dc-f1461a5b836a/scratchpad/audit400"
W_U, W_D = 0.6934, 0.3066  # measured corpus shares, five-way unanimity at >=0.5

verdicts = pd.read_csv(sys.argv[1])
key = pd.read_csv(f"{OUT}/adjudication_key.csv")
df = verdicts.merge(key, on="row_id", how="left", validate="one_to_one")
assert df["answer"].isin([0, 1]).all() and df["column"].notna().all()
df["err"] = (df["answer"] != df["sign"]).astype(int)

res = {}
for s, w in (("U", W_U), ("D", W_D)):
    sub = df[df["stratum"] == s]
    k, n = int(sub["err"].sum()), len(sub)
    res[s] = dict(k=k, n=n, rate=round(k / n, 4), upper95=round((k + 3) / n, 4))
B = W_U * res["U"]["upper95"] + W_D * res["D"]["upper95"]
res["corpus_upper95"] = round(B, 4)
res["verdict"] = "LICENSED" if B < 0.03 else ("CONDITIONAL" if B <= 0.06 else "REFUTED")
res["per_column"] = df.groupby(["stratum", "column"])["err"].agg(["sum", "count"]).to_dict()
print(json.dumps(res, indent=1, default=str))
