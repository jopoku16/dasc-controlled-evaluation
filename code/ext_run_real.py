"""
ext_run_real.py: the extended-grid experiment on the real streams (extended step-size set, scaled bandwidths).

Usage: python -X utf8 ext_run_real.py finance [electricity weather] [--fc naive,ridge,xgb] [--no-wide] [--force]

Per cell (dataset x forecaster) it writes, without touching any existing file:
  results/tuning/ext_<dataset>_<fc>_grid.csv       every configuration with development metrics
  results/tuning/ext_<dataset>_<fc>_chosen.csv     chosen configurations and budgets
  results/ext_parts/summary_<dataset>_<fc>.csv     evaluation summaries (ext, wide, main-run and previous-study runs)
  results/ext_parts/paired_<dataset>_<fc>.csv      paired moving-block bootstrap
  results/ext_parts/spotcheck_<dataset>_<fc>.csv   development score: grid (stopped at dev_end) against full-stream run
  results/diagnostics/ext_<dataset>_<fc>_<variant>.parquet  per-step diagnostics of the main variants
  results/ext_bandwidth_scale_<dataset>.csv         median pairwise feature distance and the h grids
ext_combine.py then builds the combined ext_*.csv files.
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import datasets as ds
import ext_lib as E
import tuning
from dasc_lib import DIAG_KEYS, STEP_KEYS, spectral_feature_matrix
from metrics import paired_block_bootstrap, summarize_run

R = E.RESULTS
PARTS = R / "ext_parts"

PAIRS = [
    # primary design
    ("dasc_clip_ext", "aci_clip_ext"), ("dasc_clip_ext", "spectral_aci_ext"), ("dasc_clip_ext", "decay"),
    ("dasc_clip_ext", "pid_ext"), ("dasc_noclip_ext", "aci_noclip_ext"), ("dasc_noclip_ext", "spectral_aci_ext"),
    ("dasc_noclip_ext", "decay"), ("spectral_aci_ext", "aci_clip_ext"), ("dasc_fixed_ext", "spectral_fixed_ext"),
    ("dasc_clip_ext", "dasc_clip_prev"), ("aci_clip_ext", "aci_clip_main"),
    # sensitivity design (stage 0)
    ("dasc_clip_wide", "aci_clip_ext"), ("dasc_clip_wide", "spectral_aci_ext"), ("dasc_clip_wide", "decay"),
    ("dasc_noclip_wide", "aci_noclip_ext"), ("dasc_clip_wide", "dasc_clip_ext"),
]
DIAG_VARIANTS = ["dasc_clip_ext", "dasc_noclip_ext", "dasc_clip_wide", "aci_clip_ext", "aci_noclip_ext",
                 "spectral_aci_ext", "spectral_fixed_ext"]


def _fmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _cfg_from_row(row):
    cfg = {}
    for k in E.CFG_KEYS:
        if k in row.index and not pd.isna(row[k]):
            cfg[k] = row[k]
    for k in ("window", "fw"):
        if k in cfg:
            cfg[k] = int(cfg[k])
    return cfg


def read_main(name, fname):
    ch = pd.read_csv(R / "tuning" / f"{name}_{fname}_chosen.csv").set_index("method")
    return {m: _cfg_from_row(ch.loc[m]) for m in ch.index}


def read_prev(name, fname):
    ch = pd.read_csv(R / "tuning" / f"{name}_{fname}_dascgamma_chosen.csv").set_index("variant")
    return {m: _cfg_from_row(ch.loc[m]) for m in ch.index}


def run_cell(name, fname, pred, y, feats, cfg, meta, hgrid, medians, wide):
    t0 = time.time()
    n = len(y)
    start, dev_end = cfg["start"], int(cfg["dev_frac"] * n)
    key = f"{name}_{fname}"
    main = read_main(name, fname)
    prev = read_prev(name, fname)
    streams = [{"y": y, "pred": pred, "feats": feats, "start": start, "dev_end": dev_end,
                "clip_zero": cfg["clip_zero"], "wbl": cfg["worst_block_len"], "default_fw": cfg["default_fw"]}]
    print(f"[{key}] tuning (pools {cfg['pools']}, fws {cfg['fws']}, wide={wide})", flush=True)
    with Pool(E.N_PROC, initializer=tuning._init, initargs=(streams,)) as pool:
        grid, chosen, budgets = E.tune_cell(pool, 1, cfg["pools"], cfg["fws"], hgrid, main, wide=wide,
                                            log=lambda s: print(s, flush=True))
        grid.insert(0, "forecaster", fname)
        grid.insert(0, "dataset", name)
        grid.to_csv(R / "tuning" / f"ext_{key}_grid.csv", index=False)

        # ---- final full-stream runs
        runs = {}                                    # label -> (method, cfg)
        for lab, (nm, c) in chosen.items():
            suffix = "_wide" if lab.endswith("_wide") else "_ext"
            base = lab.replace("_wide", "")
            runs[base + suffix] = (nm, c)
        for m in ("rolling", "decay"):
            runs[m] = (m, main[m])
        for m in ("aci_clip", "aci_noclip", "pid", "spectral_fixed", "spectral_aci", "dasc_fixed",
                  "dasc_clip", "dasc_noclip"):
            runs[m + "_main"] = (m, main[m])
        runs["dasc_clip_prev"] = ("dasc_clip", prev["dasc_clip"])
        runs["dasc_noclip_prev"] = ("dasc_noclip", prev["dasc_noclip"])
        labels = list(runs)
        res = pool.map(E.full_job, [(lab, runs[lab][0], runs[lab][1], 0) for lab in labels], chunksize=1)
    outs = {lab: out for lab, out in res}
    print(f"[{key}] final runs done ({time.time() - t0:.0f} s)", flush=True)

    # ---- summaries
    main_sum = pd.read_csv(R / "real_summary.csv")
    prev_sum = pd.read_csv(R / "real_summary_dascgamma.csv")
    rows, ch_rows, sp = [], [], []
    for lab in labels:
        nm, c = runs[lab]
        out = outs[lab]
        ev = out["t"] >= dev_end
        s = summarize_run(out, ev, cfg["worst_block_len"])
        s_dev = summarize_run(out, ~ev, cfg["worst_block_len"])
        row = {"dataset": name, "forecaster": fname, "variant": lab, "method": nm, "config": str(c),
               "gamma": c.get("gamma", c.get("kp", np.nan)), "h": c.get("h", np.nan), "window": c.get("window", np.nan),
               "fw": c.get("fw", np.nan), **s, "dev_score": s_dev["score_mean"],
               **E.neff_stats(out, ev)}
        # reproduction check for the re-run main-run and previous-study configurations
        ref = None
        if lab in ("rolling", "decay") or lab.endswith("_main"):
            ref = main_sum[(main_sum["dataset"] == name) & (main_sum["forecaster"] == fname) & (main_sum["method"] == nm)]
        elif lab.endswith("_prev"):
            ref = prev_sum[(prev_sum["dataset"] == name) & (prev_sum["forecaster"] == fname)
                           & (prev_sum["variant"] == nm + "_tg")]
        if ref is not None and len(ref):
            row["matches_reference"] = bool(abs(float(ref["coverage"].iloc[0]) - s["coverage"]) < 1e-9
                                            and abs(float(ref["width_mean"].iloc[0]) - s["width_mean"]) < 1e-6)
        rows.append(row)
        if lab.endswith("_ext") or lab.endswith("_wide"):
            key0 = lab[:-4] if lab.endswith("_ext") else lab          # chosen-label of this run
            bud = budgets[key0]
            full_dev = float(out["score"][~ev].mean())
            ch_rows.append({"dataset": name, "forecaster": fname, "variant": lab, "method": nm, **bud,
                            **{k: c.get(k, np.nan) for k in E.CFG_KEYS}, "dev_score_fullrun": full_dev})
            gs = bud["dev_score_grid"]
            sp.append({"dataset": name, "forecaster": fname, "variant": lab, "dev_score_grid": gs,
                       "dev_score_fullrun": full_dev,
                       "abs_diff": abs(gs - full_dev) if np.isfinite(gs) and np.isfinite(full_dev) else np.nan})
    pd.DataFrame(rows).to_csv(PARTS / f"summary_{key}.csv", index=False)
    pd.DataFrame(sp).to_csv(PARTS / f"spotcheck_{key}.csv", index=False)
    pd.DataFrame(ch_rows).to_csv(R / "tuning" / f"ext_{key}_chosen.csv", index=False)

    # ---- diagnostics
    for lab in DIAG_VARIANTS:
        if lab in outs:
            out = outs[lab]
            d = pd.DataFrame({k: out[k] for k in STEP_KEYS + DIAG_KEYS})
            d["datetime"] = meta["datetime"][out["t"]]
            d["in_eval"] = out["t"] >= dev_end
            d.to_parquet(R / "diagnostics" / f"ext_{key}_{lab}.parquet", index=False)

    # ---- paired bootstrap
    boot = []
    for a, b in PAIRS:
        if a not in outs or b not in outs:
            continue
        ev = outs[a]["t"] >= dev_end
        r = paired_block_bootstrap(outs[a], outs[b], cfg["block_len"], n_boot=2000, seed=0, mask=ev)
        boot.append({"dataset": name, "forecaster": fname, "method_a": a, "method_b": b,
                     "coverage_a": 1 - outs[a]["miss"][ev].mean(), "coverage_b": 1 - outs[b]["miss"][ev].mean(),
                     "width_a": _fmean(outs[a]["width"][ev]), "width_b": _fmean(outs[b]["width"][ev]), **r})
    pd.DataFrame(boot).to_csv(PARTS / f"paired_{key}.csv", index=False)
    print(f"[{key}] done ({time.time() - t0:.0f} s)", flush=True)
    for r in rows:
        print(f"   {r['variant']:18s} gamma={r['gamma']!s:6s} h={r['h']!s:8s} cov={r['coverage']:.4f} "
              f"width={r['width_mean']:.4f} IS={r['score_mean']:.4f} neff_med={r['neff_median']:.1f} "
              f"pool={r['pool_median']:.0f} inf={r['n_inf_width']}"
              + ("" if r.get("matches_reference", True) else "  MISMATCH"), flush=True)


def run_dataset(name, fcs, wide, force):
    t0 = time.time()
    cfg = ds.REAL_DATASETS[name]
    meta = cfg["load"]()
    y = meta["y"]
    n = len(y)
    start, dev_end = cfg["start"], int(cfg["dev_frac"] * n)
    preds = cfg["forecasts"](y, meta, with_xgb=("xgb" in fcs))
    feats = {fw: spectral_feature_matrix(y, fw) for fw in cfg["fws"]}
    medians = {fw: E.median_pairwise(feats[fw], start, dev_end) for fw in cfg["fws"]}
    hgrid = E.make_hgrid(medians)
    sc = pd.DataFrame([{"dataset": name, "fw": fw, "n_vectors": dev_end - start, "median_pairwise": medians[fw],
                        **{f"h_{i + 1}": h for i, h in enumerate(hgrid[fw])},
                        **{f"orig_h_over_median_{h:g}": h / medians[fw] for h in E.H_ORIG}} for fw in cfg["fws"]])
    sc.to_csv(R / f"ext_bandwidth_scale_{name}.csv", index=False)
    print(f"[{name}] n={n} start={start} dev_end={dev_end} medians={medians} hgrid={hgrid} ({time.time() - t0:.0f} s)",
          flush=True)
    for fname, pred in preds.items():
        short = "naive" if fname.startswith("naive") else fname
        if short not in fcs and fname not in fcs:
            continue
        if (PARTS / f"paired_{name}_{fname}.csv").exists() and not force:
            print(f"[{name}_{fname}] exists, skipped", flush=True)
            continue
        run_cell(name, fname, pred, y, feats, cfg, meta, hgrid, medians, wide)
    print(f"[{name}] total {time.time() - t0:.0f} s", flush=True)


def main(argv):
    fcs = argv[argv.index("--fc") + 1].split(",") if "--fc" in argv else ["naive", "ridge"]
    wide = "--no-wide" not in argv
    force = "--force" in argv
    PARTS.mkdir(exist_ok=True)
    for name in [a for a in argv if a in ds.REAL_DATASETS]:
        run_dataset(name, fcs, wide, force)


if __name__ == "__main__":
    main(sys.argv[1:])
