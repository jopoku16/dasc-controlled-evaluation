"""
run_simulations.py: regime-switching and control simulations, 100 replications.

Usage: python -X utf8 run_simulations.py [scenario ...] [--seeds N]

Protocol: for each scenario and forecaster, every method is tuned on 10
development seeds (1000-1009, full streams, mean interval score), frozen, and
evaluated on seeds 0-99.  Uncertainty is the standard error over the 100
independent replications; paired differences are computed per replication.

Outputs (under ../results/):
  tuning/sim_<scenario>_<forecaster>_grid.csv, _chosen.csv
  sim_per_seed.csv        per replication, per method: coverage, width, score, worst block
  sim_summary.csv         mean and SE over replications
  sim_regime_summary.csv  coverage and width by regime, mean and SE
  sim_paired.csv          paired differences (A - B) per scenario, mean and SE over replications
  drift_detection.csv     detection delay and AUC of D_t and the rolling coverage deficit
  diagnostics/sim_<scenario>_<forecaster>_<method>.parquet  per-step diagnostics (seeds 0-2)
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
from dasc_lib import DIAG_KEYS, METHOD_NAMES, STEP_KEYS, spectral_feature_matrix
from metrics import summarize_run, worst_block
from tuning import run_chosen, tune

RESULTS = Path(__file__).resolve().parent.parent / "results"
PAIRS = [("dasc_clip", "aci_clip"), ("dasc_clip", "spectral_aci"), ("dasc_clip", "decay"),
         ("dasc_clip", "pid"), ("dasc_noclip", "aci_noclip"), ("dasc_clip", "rolling"),
         ("spectral_aci", "aci_clip"), ("dasc_noclip", "dasc_clip"),
         ("dasc_fixed", "spectral_fixed"), ("dasc_noclip", "spectral_aci")]


def _fmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan
N_PROC = 8
CFG = ds.SIM_CFG
DETECT_WINDOW = 50       # rolling window of the coverage-deficit detector
DETECT_HORIZON = 300     # detection censored after this many steps
AUC_POS = 100            # steps after the break counted as positives
AUC_NEG = 300            # steps before the break counted as negatives


def build_stream(scenario, seed):
    sim = ds.make_sim(scenario, seed)
    y = sim["y"]
    preds = ds.sim_forecasts(y)
    feats = {fw: spectral_feature_matrix(y, fw) for fw in CFG["fws"]}
    return y, preds, feats, sim["regime"]


def _auc(pos, neg):
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def detection_stats(t, stat, brk, start):
    """Threshold = 99th percentile of stat on [start, brk - 50); delay to first
    exceedance at t >= brk (censored at DETECT_HORIZON); AUC post vs pre."""
    ref = (t >= start) & (t < brk - 50)
    if ref.sum() < 20 or not np.isfinite(stat[ref]).any():
        return {"delay": np.nan, "detected": np.nan, "auc": np.nan, "false_alarm_rate": np.nan}
    thr = np.nanpercentile(stat[ref], 99)
    post = (t >= brk) & (t < brk + DETECT_HORIZON)
    exceed = post & (stat > thr)
    detected = bool(exceed.any())
    delay = float(t[exceed][0] - brk) if detected else np.nan
    pre_fa = (t >= start) & (t < brk) & (stat > thr)
    pos = stat[(t >= brk) & (t < brk + AUC_POS)]
    neg = stat[(t >= max(start, brk - AUC_NEG)) & (t < brk)]
    return {"delay": delay, "detected": float(detected), "auc": _auc(pos, neg),
            "false_alarm_rate": float(pre_fa.sum() / max(1, ((t >= start) & (t < brk)).sum()))}


def _eval_seed(args):
    scenario, seed, chosen_by_fc, breaks, keep_diag = args
    y, preds, feats, regime = build_stream(scenario, seed)
    rows, reg_rows, pair_rows, det_rows, diags = [], [], [], [], {}
    for fname, pred in preds.items():
        outs = {}
        for m in METHOD_NAMES:
            out = run_chosen(y, pred, feats, m, chosen_by_fc[fname][m], CFG["start"],
                             CFG["clip_zero"], CFG["default_fw"])
            outs[m] = out
            s = summarize_run(out, None, CFG["worst_block_len"])
            rows.append({"scenario": scenario, "seed": seed, "forecaster": fname, "method": m, **s})
            reg = regime[out["t"]]
            for g in np.unique(reg):
                sel = reg == g
                reg_rows.append({"scenario": scenario, "seed": seed, "forecaster": fname, "method": m,
                                 "regime": g, "coverage": 1 - out["miss"][sel].mean(),
                                 "width_mean": _fmean(out["width"][sel]), "n": int(sel.sum())})
            if keep_diag:
                diags[(fname, m)] = pd.DataFrame({k: out[k] for k in STEP_KEYS + DIAG_KEYS}).assign(seed=seed, regime=reg)
        for a, b in PAIRS:
            wa, wb = _fmean(outs[a]["width"]), _fmean(outs[b]["width"])
            pair_rows.append({"scenario": scenario, "seed": seed, "forecaster": fname,
                              "method_a": a, "method_b": b,
                              "width_diff": wa - wb, "rel_width_diff": (wa - wb) / wb,
                              "coverage_diff": outs[b]["miss"].mean() - outs[a]["miss"].mean(),
                              "score_diff": _fmean(outs[a]["score"]) - _fmean(outs[b]["score"])})
        # drift detection: D_t from DASC vs rolling coverage deficit of rolling conformal
        if breaks and fname == list(preds)[0]:
            t = outs["dasc_clip"]["t"]
            d_stat = outs["dasc_clip"]["drift"]
            miss_roll = outs["rolling"]["miss"].astype(float)
            deficit = np.full(len(t), np.nan)
            cs = np.concatenate([[0.0], np.cumsum(miss_roll)])
            for i in range(DETECT_WINDOW, len(t)):
                deficit[i] = (cs[i] - cs[i - DETECT_WINDOW]) / DETECT_WINDOW - 0.10
            for brk in breaks:
                for sname, stat in [("drift_score_D_t", d_stat), ("rolling_coverage_deficit", deficit)]:
                    det_rows.append({"scenario": scenario, "seed": seed, "forecaster": fname, "break": brk,
                                     "statistic": sname, **detection_stats(t, stat, brk, CFG["start"])})
    return rows, reg_rows, pair_rows, det_rows, diags


def _mean_se(df, keys, cols):
    g = df.groupby(keys)
    out = g[cols].mean()
    se = g[cols].sem()
    se.columns = [c + "_se" for c in cols]
    n = g.size().rename("n_reps")
    return pd.concat([out, se, n], axis=1).reset_index()


def run_scenario(scenario, n_seeds=100, n_dev=len(ds.TUNING_SEEDS)):
    t0 = time.time()
    gen, expectation, breaks = ds.SIM_SCENARIOS[scenario]
    # tuning on development seeds, per forecaster
    dev = [build_stream(scenario, s) for s in ds.TUNING_SEEDS[:n_dev]]
    fnames = list(dev[0][1])
    chosen_by_fc = {}
    (RESULTS / "tuning").mkdir(parents=True, exist_ok=True)
    for fname in fnames:
        streams = [{"y": y, "pred": preds[fname], "feats": feats, "start": CFG["start"],
                    "dev_end": len(y), "clip_zero": CFG["clip_zero"], "wbl": CFG["worst_block_len"]}
                   for (y, preds, feats, _) in dev]
        grid, chosen, counts = tune(streams, CFG["pools"], CFG["fws"], n_proc=N_PROC,
                                    default_fw=CFG["default_fw"])
        grid.insert(0, "forecaster", fname)
        grid.insert(0, "scenario", scenario)
        grid.to_csv(RESULTS / "tuning" / f"sim_{scenario}_{fname}_grid.csv", index=False)
        pd.DataFrame([{"scenario": scenario, "forecaster": fname, "method": m, "n_configs": counts[m],
                       **chosen[m]} for m in METHOD_NAMES]).to_csv(
            RESULTS / "tuning" / f"sim_{scenario}_{fname}_chosen.csv", index=False)
        chosen_by_fc[fname] = chosen
    print(f"[{scenario}] tuning done ({time.time() - t0:.0f} s)", flush=True)

    jobs = [(scenario, s, chosen_by_fc, breaks, s < 3) for s in ds.EVAL_SEEDS[:n_seeds]]
    with Pool(N_PROC) as pool:
        results = pool.map(_eval_seed, jobs, chunksize=2)
    rows = [r for res in results for r in res[0]]
    reg_rows = [r for res in results for r in res[1]]
    pair_rows = [r for res in results for r in res[2]]
    det_rows = [r for res in results for r in res[3]]
    (RESULTS / "diagnostics").mkdir(parents=True, exist_ok=True)
    diag_frames = {}
    for res in results:
        for key, df in res[4].items():
            diag_frames.setdefault(key, []).append(df)
    for (fname, m), frames in diag_frames.items():
        pd.concat(frames, ignore_index=True).to_parquet(
            RESULTS / "diagnostics" / f"sim_{scenario}_{fname}_{m}.parquet", index=False)

    per_seed = pd.DataFrame(rows)
    per_seed["expectation"] = expectation
    _append(per_seed, RESULTS / "sim_per_seed.csv", ["scenario"])
    cols = ["coverage", "width_mean", "width_median", "score_mean", "worst_block_coverage",
            "frac_blocks_below_085", "n_inf_width", "n_empty", "neff_median", "drift_mean",
            "dbar_mean", "dbar_pre_mean", "pi_mean", "wmax_mean", "frac_alpha_at_hi_cap",
            "frac_alpha_at_lo_cap", "frac_alpha_le_0", "frac_alpha_ge_1", "frac_bind_cap",
            "frac_bind_floor", "telescoping_term", "projection_term", "free_bound",
            "lb_005_mean", "lb_005_q10"]
    for c in cols:
        if c not in per_seed.columns:
            per_seed[c] = np.nan
    summ = _mean_se(per_seed, ["scenario", "forecaster", "method"], cols)
    summ["expectation"] = expectation
    for fname in fnames:
        for m in METHOD_NAMES:
            summ.loc[(summ["forecaster"] == fname) & (summ["method"] == m), "config"] = str(chosen_by_fc[fname][m])
    _append(summ, RESULTS / "sim_summary.csv", ["scenario"])
    reg = _mean_se(pd.DataFrame(reg_rows), ["scenario", "forecaster", "method", "regime"], ["coverage", "width_mean"])
    _append(reg, RESULTS / "sim_regime_summary.csv", ["scenario"])
    pair = _mean_se(pd.DataFrame(pair_rows), ["scenario", "forecaster", "method_a", "method_b"],
                    ["width_diff", "rel_width_diff", "coverage_diff", "score_diff"])
    _append(pair, RESULTS / "sim_paired.csv", ["scenario"])
    if det_rows:
        det = pd.DataFrame(det_rows)
        g = det.groupby(["scenario", "forecaster", "break", "statistic"])
        det_s = pd.DataFrame({
            "detection_rate": g["detected"].mean(),
            "mean_delay_detected": g["delay"].mean(),
            "median_delay_detected": g["delay"].median(),
            "mean_auc": g["auc"].mean(), "auc_se": g["auc"].sem(),
            "false_alarm_rate": g["false_alarm_rate"].mean(), "n_reps": g.size()}).reset_index()
        _append(det_s, RESULTS / "drift_detection.csv", ["scenario"])
    print(f"[{scenario}] {n_seeds} seeds evaluated ({time.time() - t0:.0f} s)", flush=True)
    for fname in fnames:
        sub = summ[summ["forecaster"] == fname]
        for _, r in sub.iterrows():
            print(f"   {fname:8s} {r['method']:15s} cov={r['coverage']:.4f}({r['coverage_se']:.4f}) "
                  f"width={r['width_mean']:.4f}({r['width_mean_se']:.4f}) IS={r['score_mean']:.4f}", flush=True)


def _append(df, path, key):
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old[key[0]].isin(df[key[0]].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def main(argv):
    n_seeds = 100
    n_dev = len(ds.TUNING_SEEDS)
    if "--seeds" in argv:
        n_seeds = int(argv[argv.index("--seeds") + 1])
    if "--dev-seeds" in argv:                      # smoke tests only
        n_dev = int(argv[argv.index("--dev-seeds") + 1])
    names = [a for a in argv if a in ds.SIM_SCENARIOS] or list(ds.SIM_SCENARIOS)
    t0 = time.time()
    for s in names:
        run_scenario(s, n_seeds, n_dev)
    print(f"run_simulations total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
