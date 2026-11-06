"""
ext_seed_check.py: Monte Carlo noise of the paired bootstrap limits behind the pre-stated criterion.
Re-draws the moving-block bootstrap of DASC (projected) minus ACI (projected) with seeds 1 to 8 (the reported
results use seed 0) from the saved per-step diagnostics, for the primary and the stage-0 design in every real cell.
Reports how often an interval would exclude zero in DASC's favour.  Does not change the verdict.
Writes results/ext_seed_check.csv.
"""
import numpy as np
import pandas as pd

import datasets as ds
from metrics import paired_block_bootstrap

import ext_lib as E

R = E.RESULTS
cells = pd.read_csv(R / "ext_real_summary.csv")[["dataset", "forecaster"]].drop_duplicates().values.tolist()


def load(name, fname, lab):
    d = pd.read_parquet(R / "diagnostics" / f"ext_{name}_{fname}_{lab}.parquet")
    out = {k: d[k].to_numpy() for k in ["t", "width", "score", "miss"]}
    return out, d["in_eval"].to_numpy()


rows = []
for name, fname in cells:
    bl = ds.REAL_DATASETS[name]["block_len"]
    aci, ev = load(name, fname, "aci_clip_ext")
    for design, lab in [("primary", "dasc_clip_ext"), ("wide", "dasc_clip_wide")]:
        da, ev2 = load(name, fname, lab)
        assert (ev == ev2).all()
        for seed in range(0, 9):
            r = paired_block_bootstrap(da, aci, bl, n_boot=2000, seed=seed, mask=ev)
            rows.append({"dataset": name, "forecaster": fname, "design": design, "seed": seed,
                         "rel_width_pct": 100 * r["rel_width_diff"], "rel_width_hi": 100 * r["rel_width_diff_hi"],
                         "score_diff": r["score_diff"], "score_hi": r["score_diff_hi"],
                         "width_adv_excl0": r["rel_width_diff_hi"] < 0, "score_adv_excl0": r["score_diff_hi"] < 0})
df = pd.DataFrame(rows)
df.to_csv(R / "ext_seed_check.csv", index=False)
g = df.groupby(["dataset", "forecaster", "design"]).agg(score_hi_min=("score_hi", "min"), score_hi_max=("score_hi", "max"),
                                                      width_hi_min=("rel_width_hi", "min"),
                                                      n_score_adv=("score_adv_excl0", "sum"), n_width_adv=("width_adv_excl0", "sum"),
                                                      n_seeds=("seed", "size")).reset_index()
print(g.round(4).to_string(index=False))
print("total draws", len(df), "with score advantage excl 0:", int(df.score_adv_excl0.sum()), "with width advantage excl 0:", int(df.width_adv_excl0.sum()))
