"""
run_dascgamma.py: close the step-size asymmetry between DASC and ACI.

In the main runs ACI's step size gamma was tuned (6 values) while DASC's was
fixed at 0.015.  Here DASC (projected and free) receives a step-size budget
at least as large as ACI's, on the same development data and with the same
criterion (mean interval score):

  stage 1  the DASC configuration chosen in the main run (pool, h, lambda,
           m_min, feature window) is re-run with gamma in the ACI grid
           {0.005, 0.01, 0.015, 0.02, 0.03, 0.05}  (6 configurations)
  stage 2  at the chosen gamma, h in {0.35, 0.55, 0.8} x lambda in {0.15, 0.45}
           (6 configurations, one of which repeats the stage-1 winner)
  budget   48 (main run) + 6 + 6 = 60 configurations per DASC variant,
           against 12 for each ACI variant, 36 for spectral ACI, 12 for decay.

The tuned-gamma DASC variants are then evaluated against the baselines of
the main run (their chosen configurations re-run; identical by determinism)
with the paired block bootstrap (real data) or standard errors over the 100
evaluation seeds (simulations).  Nothing from the main run is overwritten:
every output carries the suffix _dascgamma.

Usage: python -X utf8 run_dascgamma.py [finance electricity weather] [sim] [--seeds N]
Outputs (under ../results/):
  tuning/<dataset>_<forecaster>_dascgamma_grid.csv, _chosen.csv
  real_summary_dascgamma.csv, real_paired_bootstrap_dascgamma.csv
  diagnostics/<dataset>_<forecaster>_<variant>_dascgamma.parquet
  tuning/sim_<scenario>_<forecaster>_dascgamma_grid.csv, _chosen.csv
  sim_per_seed_dascgamma.csv, sim_summary_dascgamma.csv,
  sim_regime_summary_dascgamma.csv, sim_paired_dascgamma.csv
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
import tuning
from dasc_lib import DIAG_KEYS, STEP_KEYS, spectral_feature_matrix
from metrics import paired_block_bootstrap, summarize_run
from run_simulations import build_stream
from tuning import GAMMAS, HS, run_chosen

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_PROC = 6
LAMBDAS = [0.15, 0.45]
VARIANTS = ["dasc_clip", "dasc_noclip"]
BASELINES = ["aci_clip", "aci_noclip", "spectral_aci", "decay", "dasc_clip", "dasc_noclip"]
PAIRS = [("dasc_clip_tg", "aci_clip"), ("dasc_clip_tg", "spectral_aci"), ("dasc_clip_tg", "decay"),
         ("dasc_clip_tg", "dasc_clip"), ("dasc_noclip_tg", "aci_noclip"), ("dasc_noclip_tg", "spectral_aci"),
         ("dasc_noclip_tg", "decay"), ("dasc_noclip_tg", "dasc_noclip")]
CFG_KEYS = ["window", "gamma", "kp", "ki", "kd", "decay", "h", "fw", "drift_lambda", "m_min_frac"]


def _fmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _chosen_from(path):
    ch = pd.read_csv(path).set_index("method")
    out = {}
    for m in ch.index:
        cfg = {k: ch.loc[m, k] for k in CFG_KEYS if k in ch.columns and not pd.isna(ch.loc[m, k])}
        for k in ("window", "fw"):
            if k in cfg:
                cfg[k] = int(cfg[k])
        out[m] = cfg
    return out


def _stage_configs(base, stage, gamma=None):
    if stage == 1:
        return [{**base, "gamma": g} for g in GAMMAS]
    return [{**base, "gamma": gamma, "h": h, "drift_lambda": lam} for h in HS for lam in LAMBDAS]


def _evaluate(streams, jobs, default_fw):
    """jobs: list of (method, config_id, cfg); returns DataFrame averaged over streams."""
    for s in streams:
        s["default_fw"] = default_fw
    tasks = [(m, ci, cfg, si) for (m, ci, cfg) in jobs for si in range(len(streams))]
    with Pool(N_PROC, initializer=tuning._init, initargs=(streams,)) as pool:
        rows = pool.map(tuning._job, tasks, chunksize=2)
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "config_id"]).agg(
        dev_score=("dev_score", "mean"), dev_coverage=("dev_coverage", "mean"), dev_width=("dev_width", "mean"),
        eval_coverage=("eval_coverage", "mean"), eval_width=("eval_width", "mean"),
        eval_score=("eval_score", "mean"), n_streams=("stream", "size")).reset_index()
    return agg


def tune_dasc(streams, chosen, default_fw, label):
    """Two-stage gamma tuning for both DASC variants; returns (grid, chosen_tg, budgets)."""
    grid_rows, chosen_tg = [], {}
    for v in VARIANTS:
        base = {k: chosen[v][k] for k in ("window", "h", "fw", "drift_lambda", "m_min_frac") if k in chosen[v]}
        # stage 1: gamma
        cfgs1 = _stage_configs(base, 1)
        agg1 = _evaluate(streams, [(v, i, c) for i, c in enumerate(cfgs1)], default_fw)
        agg1["stage"] = 1
        agg1["config"] = [str(cfgs1[i]) for i in agg1["config_id"]]
        for k in CFG_KEYS:
            agg1[k] = [cfgs1[i].get(k, np.nan) for i in agg1["config_id"]]
        best1 = int(agg1.sort_values(["dev_score", "config_id"]).iloc[0]["config_id"])
        g_star = cfgs1[best1]["gamma"]
        # stage 2: h x lambda at the chosen gamma
        cfgs2 = _stage_configs(base, 2, gamma=g_star)
        agg2 = _evaluate(streams, [(v, i, c) for i, c in enumerate(cfgs2)], default_fw)
        agg2["stage"] = 2
        agg2["config"] = [str(cfgs2[i]) for i in agg2["config_id"]]
        for k in CFG_KEYS:
            agg2[k] = [cfgs2[i].get(k, np.nan) for i in agg2["config_id"]]
        both = pd.concat([agg1, agg2], ignore_index=True)
        best = both.sort_values(["dev_score", "stage", "config_id"]).iloc[0]
        final = cfgs1[int(best["config_id"])] if best["stage"] == 1 else cfgs2[int(best["config_id"])]
        both["selected"] = [(r["stage"] == best["stage"]) and (r["config_id"] == best["config_id"]) for _, r in both.iterrows()]
        both.insert(0, "variant", v)
        grid_rows.append(both)
        chosen_tg[v] = final
        print(f"   [{label}] {v}: stage-1 gamma* = {g_star:g}; final {final}", flush=True)
    grid = pd.concat(grid_rows, ignore_index=True)
    budgets = {v: {"main_run": 48, "stage1_gamma": 6, "stage2_h_lambda": 6, "total": 60} for v in VARIANTS}
    return grid, chosen_tg, budgets


def _append(df, path, key_cols):
    if path.exists():
        old = pd.read_csv(path)
        merged = old.merge(df[key_cols].drop_duplicates(), on=key_cols, how="left", indicator=True)
        old = old[merged["_merge"].to_numpy() == "left_only"]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


# ----------------------------------------------------------------------------
# real data
# ----------------------------------------------------------------------------


def run_real(name):
    t0 = time.time()
    cfg = ds.REAL_DATASETS[name]
    meta = cfg["load"]()
    y = meta["y"]
    n = len(y)
    start, dev_end = cfg["start"], int(cfg["dev_frac"] * n)
    preds = cfg["forecasts"](y, meta, with_xgb=True)
    feats = {fw: spectral_feature_matrix(y, fw) for fw in cfg["fws"]}
    print(f"[{name}] n={n} dev_end={dev_end} forecasters={list(preds)} ({time.time() - t0:.0f} s)", flush=True)
    main_summary = pd.read_csv(RESULTS / "real_summary.csv")
    for fname, pred in preds.items():
        t1 = time.time()
        chosen = _chosen_from(RESULTS / "tuning" / f"{name}_{fname}_chosen.csv")
        streams = [{"y": y, "pred": pred, "feats": feats, "start": start, "dev_end": dev_end,
                    "clip_zero": cfg["clip_zero"], "wbl": cfg["worst_block_len"]}]
        grid, chosen_tg, budgets = tune_dasc(streams, chosen, cfg["default_fw"], f"{name}/{fname}")
        grid.insert(0, "forecaster", fname)
        grid.insert(0, "dataset", name)
        grid.to_csv(RESULTS / "tuning" / f"{name}_{fname}_dascgamma_grid.csv", index=False)
        pd.DataFrame([{"dataset": name, "forecaster": fname, "variant": v, **budgets[v], **chosen_tg[v]}
                      for v in VARIANTS]).to_csv(RESULTS / "tuning" / f"{name}_{fname}_dascgamma_chosen.csv", index=False)

        outs, rows = {}, []
        runs = {v + "_tg": (v, chosen_tg[v]) for v in VARIANTS}
        runs.update({b: (b, chosen[b]) for b in BASELINES})
        for label, (m, c) in runs.items():
            out = run_chosen(y, pred, feats, m, c, start, cfg["clip_zero"], cfg["default_fw"])
            outs[label] = out
            ev = out["t"] >= dev_end
            s = summarize_run(out, ev, cfg["worst_block_len"])
            s_dev = summarize_run(out, ~ev, cfg["worst_block_len"])
            rows.append({"dataset": name, "forecaster": fname, "variant": label, "method": m,
                         "config": str(c), "gamma": c.get("gamma", np.nan), "tuned_gamma": label.endswith("_tg"),
                         **s, "dev_score": s_dev["score_mean"]})
            if label.endswith("_tg"):
                diag = pd.DataFrame({k: out[k] for k in STEP_KEYS + DIAG_KEYS})
                diag["datetime"] = meta["datetime"][out["t"]]
                diag["in_eval"] = ev
                diag.to_parquet(RESULTS / "diagnostics" / f"{name}_{fname}_{m}_dascgamma.parquet", index=False)
            # determinism check against the main run for the baselines
            if not label.endswith("_tg"):
                ref = main_summary[(main_summary["dataset"] == name) & (main_summary["forecaster"] == fname)
                                   & (main_summary["method"] == m)]
                if len(ref):
                    dc = abs(float(ref["coverage"].iloc[0]) - s["coverage"])
                    dw = abs(float(ref["width_mean"].iloc[0]) - s["width_mean"])
                    rows[-1]["matches_main_run"] = bool(dc < 1e-9 and dw < 1e-6)
        _append(pd.DataFrame(rows), RESULTS / "real_summary_dascgamma.csv", ["dataset", "forecaster", "variant"])
        boot = []
        for a, b in PAIRS:
            ev = outs[a]["t"] >= dev_end
            r = paired_block_bootstrap(outs[a], outs[b], cfg["block_len"], n_boot=2000, seed=0, mask=ev)
            boot.append({"dataset": name, "forecaster": fname, "method_a": a, "method_b": b,
                         "coverage_a": 1 - outs[a]["miss"][ev].mean(), "coverage_b": 1 - outs[b]["miss"][ev].mean(),
                         "width_a": _fmean(outs[a]["width"][ev]), "width_b": _fmean(outs[b]["width"][ev]), **r})
        _append(pd.DataFrame(boot), RESULTS / "real_paired_bootstrap_dascgamma.csv", ["dataset", "forecaster", "method_a", "method_b"])
        print(f"[{name}/{fname}] done ({time.time() - t1:.0f} s)", flush=True)
        for r in rows:
            print(f"   {r['variant']:15s} gamma={r['gamma']!s:6s} cov={r['coverage']:.4f} width={r['width_mean']:.4f} "
                  f"IS={r['score_mean']:.4f} inf={r['n_inf_width']}" + ("" if r.get("matches_main_run", True) else "  MISMATCH"), flush=True)
    print(f"[{name}] total {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------
# simulations
# ----------------------------------------------------------------------------

SIM_CFG = ds.SIM_CFG


def _eval_seed(args):
    scenario, seed, runs_by_fc = args
    y, preds, feats, regime = build_stream(scenario, seed)
    rows, pair_rows = [], []
    for fname, pred in preds.items():
        outs = {}
        for label, (m, c) in runs_by_fc[fname].items():
            out = run_chosen(y, pred, feats, m, c, SIM_CFG["start"], SIM_CFG["clip_zero"], SIM_CFG["default_fw"])
            outs[label] = out
            s = summarize_run(out, None, SIM_CFG["worst_block_len"])
            row = {"scenario": scenario, "seed": seed, "forecaster": fname, "variant": label, "method": m,
                   "gamma": c.get("gamma", np.nan), **s}
            reg = regime[out["t"]]
            for g in np.unique(reg):
                sel = reg == g
                row[f"cov_{g}"] = 1 - out["miss"][sel].mean()
            rows.append(row)
        for a, b in PAIRS:
            wa, wb = _fmean(outs[a]["width"]), _fmean(outs[b]["width"])
            pair_rows.append({"scenario": scenario, "seed": seed, "forecaster": fname, "method_a": a, "method_b": b,
                              "width_diff": wa - wb, "rel_width_diff": (wa - wb) / wb,
                              "coverage_diff": outs[b]["miss"].mean() - outs[a]["miss"].mean(),
                              "score_diff": _fmean(outs[a]["score"]) - _fmean(outs[b]["score"])})
    return rows, pair_rows


def run_sim(scenario, n_seeds=100):
    t0 = time.time()
    dev = [build_stream(scenario, s) for s in ds.TUNING_SEEDS]
    fnames = list(dev[0][1])
    runs_by_fc = {}
    for fname in fnames:
        chosen = _chosen_from(RESULTS / "tuning" / f"sim_{scenario}_{fname}_chosen.csv")
        streams = [{"y": y, "pred": preds[fname], "feats": feats, "start": SIM_CFG["start"], "dev_end": len(y),
                    "clip_zero": SIM_CFG["clip_zero"], "wbl": SIM_CFG["worst_block_len"]} for (y, preds, feats, _) in dev]
        grid, chosen_tg, budgets = tune_dasc(streams, chosen, SIM_CFG["default_fw"], f"{scenario}/{fname}")
        grid.insert(0, "forecaster", fname)
        grid.insert(0, "scenario", scenario)
        grid.to_csv(RESULTS / "tuning" / f"sim_{scenario}_{fname}_dascgamma_grid.csv", index=False)
        pd.DataFrame([{"scenario": scenario, "forecaster": fname, "variant": v, **budgets[v], **chosen_tg[v]}
                      for v in VARIANTS]).to_csv(RESULTS / "tuning" / f"sim_{scenario}_{fname}_dascgamma_chosen.csv", index=False)
        runs = {v + "_tg": (v, chosen_tg[v]) for v in VARIANTS}
        runs.update({b: (b, chosen[b]) for b in BASELINES})
        runs_by_fc[fname] = runs
    print(f"[{scenario}] tuning done ({time.time() - t0:.0f} s)", flush=True)
    with Pool(N_PROC) as pool:
        res = pool.map(_eval_seed, [(scenario, s, runs_by_fc) for s in ds.EVAL_SEEDS[:n_seeds]], chunksize=2)
    per_seed = pd.DataFrame([r for rr in res for r in rr[0]])
    pairs = pd.DataFrame([r for rr in res for r in rr[1]])
    _append(per_seed, RESULTS / "sim_per_seed_dascgamma.csv", ["scenario", "seed", "forecaster", "variant"])
    cols = ["coverage", "width_mean", "width_median", "score_mean", "worst_block_coverage", "n_inf_width"]
    g = per_seed.groupby(["scenario", "forecaster", "variant", "method"])
    summ = pd.concat([g[cols].mean(), g[cols].sem().add_suffix("_se"), g["gamma"].first(), g.size().rename("n_reps")], axis=1).reset_index()
    _append(summ, RESULTS / "sim_summary_dascgamma.csv", ["scenario", "forecaster", "variant"])
    reg_cols = [c for c in per_seed.columns if c.startswith("cov_")]
    gr = per_seed.groupby(["scenario", "forecaster", "variant"])
    regs = pd.concat([gr[reg_cols].mean(), gr[reg_cols].sem().add_suffix("_se")], axis=1).reset_index()
    _append(regs, RESULTS / "sim_regime_summary_dascgamma.csv", ["scenario", "forecaster", "variant"])
    pc = ["width_diff", "rel_width_diff", "coverage_diff", "score_diff"]
    gp = pairs.groupby(["scenario", "forecaster", "method_a", "method_b"])
    psum = pd.concat([gp[pc].mean(), gp[pc].sem().add_suffix("_se"), gp.size().rename("n_reps")], axis=1).reset_index()
    _append(psum, RESULTS / "sim_paired_dascgamma.csv", ["scenario", "forecaster", "method_a", "method_b"])
    print(f"[{scenario}] {n_seeds} seeds done ({time.time() - t0:.0f} s)", flush=True)
    for _, r in psum[psum["method_a"] == "dasc_clip_tg"].iterrows():
        print(f"   {r['forecaster']:7s} dasc_clip_tg - {r['method_b']:13s} width {100 * r['rel_width_diff']:+.2f}% ({100 * r['rel_width_diff_se']:.2f}) "
              f"cov {r['coverage_diff']:+.4f} ({r['coverage_diff_se']:.4f}) IS {r['score_diff']:+.4f} ({r['score_diff_se']:.4f})", flush=True)


def main(argv):
    t0 = time.time()
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 100
    names = [a for a in argv if a in ds.REAL_DATASETS]
    for name in names:
        run_real(name)
    if "sim" in argv:
        scen = [a for a in argv if a in ds.SIM_SCENARIOS] or list(ds.SIM_SCENARIOS)
        for s in scen:
            run_sim(s, n_seeds)
    print(f"run_dascgamma total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
