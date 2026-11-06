"""
run_real_data.py: electricity, weather and finance with every method and
every forecaster under the shared tune-then-freeze protocol.

Usage: python -X utf8 run_real_data.py [electricity weather finance] [--no-xgb]
                                        [--forecasters naive24,ridge,xgb]
The optional forecaster filter reruns only those forecasters; rows of the
other forecasters in the shared result files are left in place.

Outputs (all under ../results/):
  forecaster_accuracy.csv                 MAE / RMSE of each forecaster (eval steps)
  tuning/<dataset>_<forecaster>_grid.csv  every configuration with dev and eval scores
  tuning/<dataset>_<forecaster>_chosen.csv  chosen configuration and grid size per method
  real_summary.csv                        eval-period metrics per dataset/forecaster/method
  real_local_coverage.csv                 coverage by hour, month, volatility bucket, year
  real_paired_bootstrap.csv               paired moving-block bootstrap differences
  diagnostics/<dataset>_<forecaster>_<method>.parquet  per-step diagnostics
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
from dasc_lib import DIAG_KEYS, METHOD_NAMES, STEP_KEYS, spectral_feature_matrix
from metrics import local_coverage, paired_block_bootstrap, summarize_run
from tuning import run_chosen, tune

RESULTS = Path(__file__).resolve().parent.parent / "results"
PAIRS = [("dasc_clip", "aci_clip"), ("dasc_clip", "spectral_aci"), ("dasc_clip", "decay"),
         ("dasc_clip", "pid"), ("dasc_noclip", "aci_noclip"), ("dasc_clip", "rolling"),
         ("spectral_aci", "aci_clip"), ("dasc_noclip", "dasc_clip"),
         ("dasc_fixed", "spectral_fixed"), ("dasc_noclip", "spectral_aci")]
N_PROC = 8


def _fmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _append_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        key = [c for c in ["dataset", "forecaster", "method", "method_a", "method_b", "group_var", "group"]
               if c in df.columns and c in old.columns]
        if key:
            merged = old.merge(df[key].drop_duplicates(), on=key, how="left", indicator=True)
            old = old[merged["_merge"].to_numpy() == "left_only"]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def run_dataset(name, with_xgb=True, only=None):
    t0 = time.time()
    cfg = ds.REAL_DATASETS[name]
    meta = cfg["load"]()
    y = meta["y"]
    n = len(y)
    start = cfg["start"]
    dev_end = int(cfg["dev_frac"] * n)
    preds = cfg["forecasts"](y, meta, with_xgb=with_xgb)
    if only:
        preds = {k: v for k, v in preds.items() if k in only}
    feats = {fw: spectral_feature_matrix(y, fw) for fw in cfg["fws"]}
    groups = cfg["groups"](meta, dev_end)
    print(f"[{name}] n={n} start={start} dev_end={dev_end} forecasters={list(preds)} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # forecaster accuracy on the evaluation steps
    acc_rows = []
    for fname, pred in preds.items():
        ev = np.arange(dev_end, n)
        ok = np.isfinite(pred[ev])
        err = y[ev][ok] - pred[ev][ok]
        acc_rows.append({"dataset": name, "forecaster": fname, "n_eval": int(ok.sum()),
                         "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                         "mean_abs_residual_dev": float(np.nanmean(np.abs(y[start:dev_end] - pred[start:dev_end])))})
    _append_csv(pd.DataFrame(acc_rows), RESULTS / "forecaster_accuracy.csv")

    for fname, pred in preds.items():
        t1 = time.time()
        streams = [{"y": y, "pred": pred, "feats": feats, "start": start, "dev_end": dev_end,
                    "clip_zero": cfg["clip_zero"], "wbl": cfg["worst_block_len"]}]
        grid, chosen, counts = tune(streams, cfg["pools"], cfg["fws"], n_proc=N_PROC,
                                    default_fw=cfg["default_fw"])
        (RESULTS / "tuning").mkdir(parents=True, exist_ok=True)
        grid.insert(0, "forecaster", fname)
        grid.insert(0, "dataset", name)
        grid.to_csv(RESULTS / "tuning" / f"{name}_{fname}_grid.csv", index=False)
        chosen_rows = [{"dataset": name, "forecaster": fname, "method": m, "n_configs": counts[m],
                        **{k: v for k, v in chosen[m].items()}} for m in METHOD_NAMES]
        pd.DataFrame(chosen_rows).to_csv(RESULTS / "tuning" / f"{name}_{fname}_chosen.csv", index=False)
        print(f"[{name}/{fname}] tuning done: {sum(counts.values())} configs ({time.time() - t1:.0f} s)", flush=True)

        outs, summ_rows, local_rows = {}, [], []
        for m in METHOD_NAMES:
            out = run_chosen(y, pred, feats, m, chosen[m], start, cfg["clip_zero"], cfg["default_fw"])
            outs[m] = out
            ev = out["t"] >= dev_end
            s = summarize_run(out, ev, cfg["worst_block_len"])
            s_dev = summarize_run(out, ~ev, cfg["worst_block_len"])
            summ_rows.append({"dataset": name, "forecaster": fname, "method": m,
                              "n_configs": counts[m], "config": str(chosen[m]),
                              **s, "dev_coverage": s_dev["coverage"], "dev_width": s_dev["width_mean"],
                              "dev_score": s_dev["score_mean"]})
            lc = local_coverage(out, groups, ev)
            lc.insert(0, "method", m)
            lc.insert(0, "forecaster", fname)
            lc.insert(0, "dataset", name)
            local_rows.append(lc)
            diag = pd.DataFrame({k: out[k] for k in STEP_KEYS + DIAG_KEYS})
            diag["datetime"] = meta["datetime"][out["t"]]
            diag["in_eval"] = ev
            (RESULTS / "diagnostics").mkdir(parents=True, exist_ok=True)
            diag.to_parquet(RESULTS / "diagnostics" / f"{name}_{fname}_{m}.parquet", index=False)
        _append_csv(pd.DataFrame(summ_rows), RESULTS / "real_summary.csv")
        _append_csv(pd.concat(local_rows, ignore_index=True), RESULTS / "real_local_coverage.csv")

        boot_rows = []
        for a, b in PAIRS:
            ev = outs[a]["t"] >= dev_end
            r = paired_block_bootstrap(outs[a], outs[b], cfg["block_len"], n_boot=2000, seed=0, mask=ev)
            boot_rows.append({"dataset": name, "forecaster": fname, "method_a": a, "method_b": b,
                              "coverage_a": 1 - outs[a]["miss"][ev].mean(),
                              "coverage_b": 1 - outs[b]["miss"][ev].mean(),
                              "width_a": _fmean(outs[a]["width"][ev]), "width_b": _fmean(outs[b]["width"][ev]), **r})
        _append_csv(pd.DataFrame(boot_rows), RESULTS / "real_paired_bootstrap.csv")
        print(f"[{name}/{fname}] evaluation done ({time.time() - t1:.0f} s)", flush=True)
        for r in summ_rows:
            print(f"   {r['method']:15s} cov={r['coverage']:.4f} width={r['width_mean']:.4f} "
                  f"IS={r['score_mean']:.4f} worst_block={r['worst_block_coverage']:.3f} "
                  f"inf={r['n_inf_width']} empty={r['n_empty']}", flush=True)
    print(f"[{name}] total {time.time() - t0:.0f} s", flush=True)


def main(argv):
    with_xgb = "--no-xgb" not in argv
    only = None
    if "--forecasters" in argv:                    # e.g. --forecasters xgb
        only = argv[argv.index("--forecasters") + 1].split(",")
    names = [a for a in argv if a in ds.REAL_DATASETS] or list(ds.REAL_DATASETS)
    t0 = time.time()
    for name in names:
        run_dataset(name, with_xgb=with_xgb, only=only)
    print(f"run_real_data total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
