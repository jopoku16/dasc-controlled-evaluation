"""
blockseed_summarise.py: tallies of the block-length and seed sensitivity grid (reads results/blockseed_cells.csv,
writes results/blockseed_summary.csv).  New file only; no existing file is changed.

Verdict labels (per interval, DASC minus comparator, percentile 95 per cent):
  favours_DASC   interval wholly below zero (narrower for width, lower score for the interval score)
  against_DASC   interval wholly above zero
  includes_zero  otherwise
'Reported' is the paper's configuration: block 168 hours (21 days for finance), bootstrap seed 0.

summary 'level' values
  overall        design x comparator x metric over all cells, block lengths and seeds
  by_variation   same, split by the source of change (seed_only, block_only, block_and_seed)
  by_block       design x comparator x metric x block rank (1 reported, 2, 3): counts over cells and seeds
  cell           design x cell x comparator x metric: reported verdict and counts over the 27 (block, seed) draws
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
df = pd.read_csv(RESULTS / "blockseed_cells.csv")

LONG = []
for m, lo, hi, pt in [("width", "rel_width_lo_pct", "rel_width_hi_pct", "rel_width_pct"),
                      ("score", "score_diff_lo", "score_diff_hi", "score_diff")]:
    t = df.copy()
    t["metric"] = m
    t["verdict"] = t[f"{m}_verdict"]
    t["verdict_reported"] = t[f"{m}_verdict_reported"]
    t["differs"] = t[f"{m}_differs_from_reported"]
    t["pt"], t["lo"], t["hi"] = t[pt], t[lo], t[hi]
    LONG.append(t)
L = pd.concat(LONG, ignore_index=True)


def counts(g):
    v = g["verdict"]
    return pd.Series({
        "n_draws": len(g),
        "n_favours_DASC": int((v == "favours_DASC").sum()),
        "n_includes_zero": int((v == "includes_zero").sum()),
        "n_against_DASC": int((v == "against_DASC").sum()),
        "n_differs_from_reported": int(g["differs"].sum()),
        "n_new_favours_DASC": int(((v == "favours_DASC") & (g["verdict_reported"] != "favours_DASC")).sum()),
        "n_new_against_DASC": int(((v == "against_DASC") & (g["verdict_reported"] != "against_DASC")).sum()),
        "n_lost_favours_DASC": int(((v != "favours_DASC") & (g["verdict_reported"] == "favours_DASC")).sum()),
        "n_lost_against_DASC": int(((v != "against_DASC") & (g["verdict_reported"] == "against_DASC")).sum()),
        "n_cells_with_any_favours": int(g.groupby(["dataset", "forecaster"])["verdict"].apply(lambda s: (s == "favours_DASC").any()).sum()),
    })


rows = []
k = ["design", "comparator", "metric"]
o = L.groupby(k).apply(counts, include_groups=False).reset_index()
o.insert(0, "level", "overall")
rows.append(o)
nr = L[L.variation != "reported"]
bv = nr.groupby(k + ["variation"]).apply(counts, include_groups=False).reset_index()
bv.insert(0, "level", "by_variation")
rows.append(bv)
bb = L.groupby(k + ["block_rank"]).apply(counts, include_groups=False).reset_index()
bb.insert(0, "level", "by_block")
rows.append(bb)


def cell_row(g):
    v = g["verdict"]
    r = {"reported_verdict": g["verdict_reported"].iloc[0],
         "reported_point": g.loc[g.reported, "pt"].iloc[0],
         "reported_lo": g.loc[g.reported, "lo"].iloc[0], "reported_hi": g.loc[g.reported, "hi"].iloc[0],
         "n_draws": len(g), "n_favours_DASC": int((v == "favours_DASC").sum()),
         "n_includes_zero": int((v == "includes_zero").sum()), "n_against_DASC": int((v == "against_DASC").sum()),
         "n_differs_from_reported": int(g["differs"].sum()),
         "min_hi": g["hi"].min(), "max_lo": g["lo"].max()}
    for b in (1, 2, 3):
        gb = g[g.block_rank == b]
        r[f"block{b}_len"] = int(gb["block_len"].iloc[0])
        r[f"block{b}_favours"] = int((gb["verdict"] == "favours_DASC").sum())
        r[f"block{b}_against"] = int((gb["verdict"] == "against_DASC").sum())
        r[f"block{b}_includes_zero"] = int((gb["verdict"] == "includes_zero").sum())
    return pd.Series(r)


c = L.groupby(["design", "dataset", "forecaster", "comparator", "metric"]).apply(cell_row, include_groups=False).reset_index()
c.insert(0, "level", "cell")
rows.append(c)
S = pd.concat(rows, ignore_index=True)
S.to_csv(RESULTS / "blockseed_summary.csv", index=False)
print("summary rows", len(S))
