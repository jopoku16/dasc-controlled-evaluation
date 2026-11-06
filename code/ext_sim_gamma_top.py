"""
ext_sim_gamma_top.py: SUPPLEMENT to the extended-grid experiment (outside the pre-stated design).

In the recurring frequency-and-noise scenario the selected step size of ACI, spectral ACI and DASC is the top value
of the extended set (0.1) for the lag-1 forecaster.  This script extends the matched-step sweep to larger steps
(0.15, 0.2, 0.3, 0.5) on the evaluation seeds, for each method's selected non-step configuration of the extended
tuning (ACI: pool; spectral ACI and DASC projected: all other selected parameters), and reports coverage, width and
interval score with standard errors over seeds and the paired differences against ACI.  No re-tuning.

Usage: python -X utf8 ext_sim_gamma_top.py [--seeds 50]       (2 worker processes)
Writes results/ext_sim_gamma_top.csv
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import datasets as ds
import ext_lib as E
from metrics import summarize_run
from run_simulations import build_stream
from tuning import run_chosen

R = E.RESULTS
CFG = ds.SIM_CFG
SCEN = "recurring_freq_noise"
STEPS = [0.1, 0.15, 0.2, 0.3, 0.5]


def _cfg(row):
    c = {k: row[k] for k in E.CFG_KEYS if k in row.index and not pd.isna(row[k])}
    for k in ("window", "fw"):
        if k in c:
            c[k] = int(c[k])
    return c


def job(args):
    seed, fams = args
    y, preds, feats, regime = build_stream(SCEN, seed)
    rows = []
    for fname, fam in fams.items():
        for step in STEPS:
            vals = {}
            for family, (m, c) in fam.items():
                out = run_chosen(y, preds[fname], feats, m, {**c, "gamma": step}, CFG["start"], CFG["clip_zero"], CFG["default_fw"])
                s = summarize_run(out, None, CFG["worst_block_len"])
                vals[family] = (s["coverage"], s["width_mean"], s["score_mean"])
            for family, (cv, w, sc) in vals.items():
                rows.append({"seed": seed, "forecaster": fname, "family": family, "step": step, "coverage": cv,
                             "width_mean": w, "score_mean": sc,
                             "score_diff_vs_aci": sc - vals["aci"][2], "width_diff_vs_aci": w - vals["aci"][1],
                             "coverage_diff_vs_aci": cv - vals["aci"][0]})
    return rows


def main(argv):
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 50
    fams = {}
    for fname in ("naive1", "ridge"):
        p = R / "tuning" / f"ext_sim_{SCEN}_{fname}_chosen.csv"
        if not p.exists():
            continue
        ch = pd.read_csv(p).set_index("variant")
        fams[fname] = {"aci": ("aci_clip", _cfg(ch.loc["aci_clip_ext"])),
                       "spectral_aci": ("spectral_aci", _cfg(ch.loc["spectral_aci_ext"])),
                       "dasc": ("dasc_clip", _cfg(ch.loc["dasc_clip_ext"]))}
    print("forecasters:", list(fams), flush=True)
    with Pool(E.N_PROC) as pool:
        res = pool.map(job, [(s, fams) for s in ds.EVAL_SEEDS[:n_seeds]], chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    g = df.groupby(["forecaster", "family", "step"])
    cols = ["coverage", "width_mean", "score_mean", "score_diff_vs_aci", "width_diff_vs_aci", "coverage_diff_vs_aci"]
    out = pd.concat([g[cols].mean(), g[cols].sem().add_suffix("_se"), g.size().rename("n_reps")], axis=1).reset_index()
    out.to_csv(R / "ext_sim_gamma_top.csv", index=False)
    print(out[["forecaster", "family", "step", "coverage", "width_mean", "score_mean", "score_diff_vs_aci", "score_diff_vs_aci_se"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
