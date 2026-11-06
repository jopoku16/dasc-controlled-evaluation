"""
run_sim_gamma_sensitivity.py: the simulation comparison at equal step size.

In the main simulation tables ACI and spectral ACI use the step size gamma
selected on the development seeds while DASC's gamma is fixed at 0.015.
This script re-evaluates ACI (projected and free) and spectral ACI at
gamma = 0.015 with DASC's selected pool (other selected parameters kept) on
the 100 evaluation seeds of two scenarios, next to DASC projected with its
selected configuration, and reports coverage, width, interval score and
regime-wise coverage with standard errors over seeds.

Writes ../results/sim_gamma_sensitivity.csv.
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
from dasc_lib import spectral_feature_matrix
from metrics import summarize_run
from run_simulations import build_stream
from tuning import run_chosen

RESULTS = Path(__file__).resolve().parent.parent / "results"
SCENARIOS = ["recurring_freq_noise", "original_three_phase"]
GAMMA = 0.015
CFG = ds.SIM_CFG


def _configs(scenario, fname):
    ch = pd.read_csv(RESULTS / "tuning" / f"sim_{scenario}_{fname}_chosen.csv").set_index("method")
    pool = int(ch.loc["dasc_clip", "window"])
    out = {}
    for m in ["aci_clip", "aci_noclip", "spectral_aci"]:
        cfg = {k: ch.loc[m, k] for k in ["window", "gamma", "h", "fw"] if k in ch.columns and not pd.isna(ch.loc[m, k])}
        cfg["window"] = pool
        cfg["gamma"] = GAMMA
        if "fw" in cfg:
            cfg["fw"] = int(cfg["fw"])
        out[m + "_gamma015"] = (m, cfg)
    dcfg = {k: ch.loc["dasc_clip", k] for k in ["window", "h", "fw", "drift_lambda", "m_min_frac"] if not pd.isna(ch.loc["dasc_clip", k])}
    dcfg["window"] = int(dcfg["window"]); dcfg["fw"] = int(dcfg["fw"])
    out["dasc_clip_selected"] = ("dasc_clip", dcfg)
    return out


def _job(args):
    scenario, seed, cfgs = args
    y, preds, feats, regime = build_stream(scenario, seed)
    rows = []
    for fname, pred in preds.items():
        for label, (m, cfg) in cfgs[fname].items():
            out = run_chosen(y, pred, feats, m, cfg, CFG["start"], CFG["clip_zero"], CFG["default_fw"])
            s = summarize_run(out, None, CFG["worst_block_len"])
            row = {"scenario": scenario, "seed": seed, "forecaster": fname, "variant": label,
                   "coverage": s["coverage"], "width_mean": s["width_mean"], "score_mean": s["score_mean"]}
            reg = regime[out["t"]]
            for g in np.unique(reg):
                sel = reg == g
                row[f"cov_{g}"] = 1 - out["miss"][sel].mean()
                row[f"width_{g}"] = float(np.nanmean(np.where(np.isfinite(out["width"][sel]), out["width"][sel], np.nan)))
            rows.append(row)
    return rows


def main(argv):
    t0 = time.time()
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 100
    frames = []
    for scenario in SCENARIOS:
        cfgs = {fname: _configs(scenario, fname) for fname in ["naive1", "ridge"]}
        with Pool(8) as pool:
            res = pool.map(_job, [(scenario, s, cfgs) for s in ds.EVAL_SEEDS[:n_seeds]], chunksize=4)
        df = pd.DataFrame([r for rows in res for r in rows])
        frames.append(df)
        print(f"[{scenario}] done ({time.time() - t0:.0f} s)", flush=True)
    per_seed = pd.concat(frames, ignore_index=True)
    per_seed.to_csv(RESULTS / "sim_gamma_sensitivity_per_seed.csv", index=False)
    cols = [c for c in per_seed.columns if c not in ("scenario", "seed", "forecaster", "variant")]
    g = per_seed.groupby(["scenario", "forecaster", "variant"])
    summ = pd.concat([g[cols].mean(), g[cols].sem().add_suffix("_se"), g.size().rename("n_reps")], axis=1).reset_index()
    summ.to_csv(RESULTS / "sim_gamma_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
