"""
lead2_run.py: equal-budget tuning of the phase-aware and calendar-hour DASC variants and the paired evaluation.

Protocol: results/lead2_protocol.txt (written before any run).  Reuses the equal-budget study's machinery
(datasets, tuning.params_from_config, dasc_lib.run_stream, metrics) without editing it.

Usage (from code/):  python -X utf8 lead2_run.py tune [electricity weather] [--forecasters naive24,ridge]
                     python -X utf8 lead2_run.py evaluate
Outputs (results/): lead2_grid_<dataset>_<forecaster>_<kind>.csv, lead2_chosen.csv  (tune)
                    lead2_summary.csv, lead2_hour_coverage.csv, lead2_mad_paired.csv, lead2_cell_tests.csv,
                    lead2_noise_floor.csv, lead2_hindsight.csv                          (evaluate)
At most two worker processes are used.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("DASC_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data" / "real"))   # same default as datasets.py

import numpy as np
import pandas as pd

import datasets as ds
import tuning
from dasc_lib import run_stream, spectral_feature_matrix
from lead2_variants import PhaseKernelConformal, calendar_angles, phase_angles
from metrics import _block_indices, summarize_run

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_PROC = 2
TARGET = 0.90
KINDS = ("phase", "cal")
CELLS = [("electricity", "naive24"), ("electricity", "ridge"), ("weather", "naive24"), ("weather", "ridge")]

GAMMAS = tuning.GAMMAS
HS = tuning.HS
LAMBDAS = [0.15, 0.45]
MMIN = [0.25, 0.5]
WS = [0.5, 1.0, 2.0, 4.0]


# ----------------------------------------------------------------------------
# streams, methods, jobs
# ----------------------------------------------------------------------------

def build_stream(name, fname):
    cfg = ds.REAL_DATASETS[name]
    meta = cfg["load"]()
    y = meta["y"]
    n = len(y)
    preds = cfg["forecasts"](y, meta, with_xgb=False)
    pred = preds[fname]
    feats = {fw: spectral_feature_matrix(y, fw) for fw in cfg["fws"]}
    aux = {"cal": (calendar_angles(meta["hour"]), np.ones(n, dtype=bool))}
    for fw in cfg["fws"]:
        ang, _, valid = phase_angles(y, fw)
        aux[("phase", fw)] = (ang, valid)
    return {"name": name, "fname": fname, "y": y, "pred": pred, "feats": feats, "aux": aux,
            "hour": np.asarray(meta["hour"]), "datetime": meta["datetime"], "start": cfg["start"],
            "dev_end": int(cfg["dev_frac"] * n), "clip_zero": cfg["clip_zero"], "wbl": cfg["worst_block_len"],
            "default_fw": cfg["default_fw"], "pools": cfg["pools"], "fws": cfg["fws"]}


def make_variant(cfg, S):
    p = tuning.params_from_config(cfg)
    kind = cfg["kind"]
    if kind == "cal":
        ang, valid = S["aux"]["cal"]
    elif kind == "phase":
        ang, valid = S["aux"][("phase", int(cfg["fw"]))]
    else:
        raise ValueError(kind)
    return PhaseKernelConformal(alpha=0.10, window=int(p["window"]), first_valid=1, h=float(p["h"]),
                                w=float(p["w"]), aux_angle=ang, aux_valid=valid,
                                drift_lambda=float(p["drift_lambda"]), m_min=int(p["m_min"]),
                                gamma=float(p["gamma"]))


def run_variant(S, cfg, end=None):
    m = make_variant(cfg, S)
    f = S["feats"][int(cfg["fw"])]
    return run_stream(S["y"], S["pred"], m, features=f, start=S["start"], end=end, clip_zero=S["clip_zero"])


def hour_cov(out, S, mask):
    t = out["t"][mask]
    h = S["hour"][t]
    miss = out["miss"][mask].astype(float)
    cnt = np.bincount(h, minlength=24).astype(float)
    cov = 1.0 - np.bincount(h, weights=miss, minlength=24) / np.maximum(cnt, 1)
    return cov


def mad_from_cov(cov):
    return float(np.mean(np.abs(cov - TARGET)))


_STREAM = None


def _init(stream):
    global _STREAM
    _STREAM = stream


def _job(args):
    ci, cfg = args
    S = _STREAM
    out = run_variant(S, cfg)
    t = out["t"]
    dev = t < S["dev_end"]
    ev = ~dev
    s = summarize_run(out, ev, S["wbl"])
    cov = hour_cov(out, S, ev)
    return {"config_id": ci, "dev_score": float(out["score"][dev].mean()),
            "dev_coverage": float(1 - out["miss"][dev].mean()), "dev_width": float(out["width"][dev].mean()),
            "eval_coverage": s["coverage"], "eval_width": s["width_mean"], "eval_score": s["score_mean"],
            "eval_mad_hour": mad_from_cov(cov)}


def evaluate_configs(S, cfgs, pool_ctx):
    res = pool_ctx.map(_job, list(enumerate(cfgs)), chunksize=1)
    df = pd.DataFrame(res).sort_values("config_id").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# tuning: 48 + 6 + 12 = 66 configurations per variant and cell
# ----------------------------------------------------------------------------

def tune_cell(S, kind, pool_ctx, log):
    cfg_keys = ["window", "h", "drift_lambda", "m_min_frac", "fw", "gamma", "w"]
    rows = []
    # stage 1
    c1 = [dict(window=pw, h=h, drift_lambda=lam, m_min_frac=mm, fw=fw, gamma=0.015, w=1.0, kind=kind)
          for pw in S["pools"] for h in HS for lam in LAMBDAS for mm in MMIN for fw in S["fws"]]
    d1 = evaluate_configs(S, c1, pool_ctx)
    d1["stage"] = 1
    best1 = c1[int(d1.sort_values(["dev_score", "config_id"]).iloc[0]["config_id"])]
    # stage 2
    c2 = [{**best1, "gamma": g} for g in GAMMAS]
    d2 = evaluate_configs(S, c2, pool_ctx)
    d2["stage"] = 2
    gstar = c2[int(d2.sort_values(["dev_score", "config_id"]).iloc[0]["config_id"])]["gamma"]
    # stage 3
    c3 = [{**best1, "gamma": gstar, "h": h, "w": w} for h in HS for w in WS]
    d3 = evaluate_configs(S, c3, pool_ctx)
    d3["stage"] = 3
    allc = []
    for st, cs, d in ((1, c1, d1), (2, c2, d2), (3, c3, d3)):
        d = d.copy()
        for k in cfg_keys:
            d[k] = [cs[i][k] for i in d["config_id"]]
        allc.append(d)
    grid = pd.concat(allc, ignore_index=True)
    best = grid.sort_values(["dev_score", "stage", "config_id"]).iloc[0]
    grid["selected"] = (grid["stage"] == best["stage"]) & (grid["config_id"] == best["config_id"])
    grid.insert(0, "kind", kind)
    grid.insert(0, "forecaster", S["fname"])
    grid.insert(0, "dataset", S["name"])
    grid.to_csv(RESULTS / f"lead2_grid_{S['name']}_{S['fname']}_{kind}.csv", index=False)
    chosen = {k: best[k] for k in cfg_keys}
    chosen["window"] = int(chosen["window"])
    chosen["fw"] = int(chosen["fw"])
    log(f"   [{S['name']}/{S['fname']}] {kind}: stage-1 winner {({k: best1[k] for k in cfg_keys})}; gamma* {gstar}; final {chosen} "
        f"(dev IS {best['dev_score']:.4f}, eval cov {best['eval_coverage']:.4f}, eval MAD {best['eval_mad_hour']:.4f}), {len(grid)} configs")
    return chosen, grid


def tune_main(argv):
    names = [a for a in argv if a in ("electricity", "weather")] or ["electricity", "weather"]
    fcs = argv[argv.index("--forecasters") + 1].split(",") if "--forecasters" in argv else ["naive24", "ridge"]
    logf = open(RESULTS / "lead2_tune.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    chosen_rows = []
    old = RESULTS / "lead2_chosen.csv"
    if old.exists():
        chosen_rows = pd.read_csv(old).to_dict("records")
    for name in names:
        for fname in fcs:
            t0 = time.time()
            S = build_stream(name, fname)
            for kind in KINDS:
                with Pool(N_PROC, initializer=_init, initargs=(S,)) as pool_ctx:
                    ch, grid = tune_cell(S, kind, pool_ctx, log)
                chosen_rows = [r for r in chosen_rows if not (r["dataset"] == name and r["forecaster"] == fname and r["kind"] == kind)]
                chosen_rows.append({"dataset": name, "forecaster": fname, "kind": kind, "n_configs": len(grid), **ch})
                pd.DataFrame(chosen_rows).to_csv(old, index=False)
                log(f"[{name}/{fname}/{kind}] done {time.time() - t0:.0f} s elapsed in cell")
    logf.close()


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

def baseline_cfgs(name, fname):
    ch = pd.read_csv(RESULTS / "tuning" / f"{name}_{fname}_chosen.csv").set_index("method")
    aci = {"window": int(ch.loc["aci_clip", "window"]), "gamma": float(ch.loc["aci_clip", "gamma"])}
    dg = pd.read_csv(RESULTS / "tuning" / f"{name}_{fname}_dascgamma_chosen.csv")
    r = dg[dg["variant"] == "dasc_clip"].iloc[0]
    dasc = {"window": int(r["window"]), "h": float(r["h"]), "fw": int(r["fw"]), "drift_lambda": float(r["drift_lambda"]),
            "m_min_frac": float(r["m_min_frac"]), "gamma": float(r["gamma"])}
    return aci, dasc


def noise_floor(miss, hour, block=168, n_perm=300, seed=0):
    """MAD of hourly coverage when hour labels are shuffled within blocks of 168 steps (no hour effect, week-level variation kept)."""
    rng = np.random.default_rng(seed)
    n = len(miss)
    mads = np.empty(n_perm)
    nb = int(np.ceil(n / block))
    for p in range(n_perm):
        hp = hour.copy()
        for b in range(nb):
            sl = slice(b * block, min((b + 1) * block, n))
            hp[sl] = rng.permutation(hour[sl])
        cnt = np.bincount(hp, minlength=24)
        cov = 1.0 - np.bincount(hp, weights=miss, minlength=24) / np.maximum(cnt, 1)
        mads[p] = mad_from_cov(cov)
    return float(mads.mean()), float(np.percentile(mads, 2.5)), float(np.percentile(mads, 97.5))


def boot_stats(runs, hours_ev, n, block_len=168, n_boot=2000, seed=0):
    """One bootstrap loop (same sequence of blocks as metrics.paired_block_bootstrap) for width, coverage, score and hour-coverage MAD."""
    rng = np.random.default_rng(seed)
    labs = list(runs)
    W = {k: runs[k]["width"] for k in labs}
    M = {k: runs[k]["miss"].astype(float) for k in labs}
    Sc = {k: runs[k]["score"] for k in labs}
    out = {k: np.empty((n_boot, 4)) for k in labs}       # mean width, coverage, mean score, MAD
    for b in range(n_boot):
        idx = _block_indices(n, block_len, rng)
        h = hours_ev[idx]
        cnt = np.maximum(np.bincount(h, minlength=24), 1)
        for k in labs:
            mi = M[k][idx]
            cov = 1.0 - np.bincount(h, weights=mi, minlength=24) / cnt
            out[k][b] = (W[k][idx].mean(), 1.0 - mi.mean(), Sc[k][idx].mean(), mad_from_cov(cov))
    return out


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def evaluate_main():
    chosen = pd.read_csv(RESULTS / "lead2_chosen.csv")
    summ_rows, hour_rows, mad_rows, floor_rows, hind_rows, cell_rows = [], [], [], [], [], []
    ref = pd.read_csv(RESULTS / "real_summary_dascgamma.csv")
    ref_boot = pd.read_csv(RESULTS / "real_paired_bootstrap_dascgamma.csv")
    for name, fname in CELLS:
        t0 = time.time()
        S = build_stream(name, fname)
        dev_end, n_all = S["dev_end"], len(S["y"])
        aci_c, dasc_c = baseline_cfgs(name, fname)
        feats_dict = S["feats"]
        runs = {}
        runs["aci"] = tuning.run_chosen(S["y"], S["pred"], feats_dict, "aci_clip", aci_c, S["start"], S["clip_zero"], S["default_fw"])
        runs["dasc_tg"] = tuning.run_chosen(S["y"], S["pred"], feats_dict, "dasc_clip", dasc_c, S["start"], S["clip_zero"], S["default_fw"])
        # determinism checks against the stored equal-budget summaries
        ev = runs["dasc_tg"]["t"] >= dev_end
        for lab, meth, key in (("aci", "aci_clip", "aci"), ("dasc_tg", "dasc_clip_tg", "dasc_tg")):
            r = ref[(ref["dataset"] == name) & (ref["forecaster"] == fname) & (ref["variant"] == meth)].iloc[0]
            s = summarize_run(runs[lab], runs[lab]["t"] >= dev_end, S["wbl"])
            assert abs(r["coverage"] - s["coverage"]) < 1e-9 and abs(r["width_mean"] - s["width_mean"]) < 1e-6, (name, fname, lab)
        # w = 0 control: the new class with no auxiliary weight must reproduce DASC tuned
        ctrl_cfg = {**dasc_c, "w": 0.0, "kind": "phase"}
        runs["w0_control"] = run_variant(S, ctrl_cfg)
        same = all(np.array_equal(runs["w0_control"][k], runs["dasc_tg"][k]) for k in ("lo", "hi", "miss", "width"))
        assert same, "w=0 control differs from DASC tuned"
        for kind in KINDS:
            c = chosen[(chosen["dataset"] == name) & (chosen["forecaster"] == fname) & (chosen["kind"] == kind)].iloc[0]
            cfg = {"window": int(c["window"]), "h": float(c["h"]), "fw": int(c["fw"]), "drift_lambda": float(c["drift_lambda"]),
                   "m_min_frac": float(c["m_min_frac"]), "gamma": float(c["gamma"]), "w": float(c["w"]), "kind": kind}
            runs[kind] = run_variant(S, cfg)
            runs[kind + "_cfg"] = cfg
        labs = ["aci", "dasc_tg", "phase", "cal"]
        t_ev = runs["aci"]["t"][runs["aci"]["t"] >= dev_end]
        for l in labs:
            assert np.array_equal(runs[l]["t"][runs[l]["t"] >= dev_end], t_ev)
        evm = {l: runs[l]["t"] >= dev_end for l in labs}
        ev_runs = {l: {k: runs[l][k][evm[l]] for k in ("width", "miss", "score", "t")} for l in labs}
        n = len(t_ev)
        hours_ev = S["hour"][t_ev]
        # block-sequence check: reproduce the stored interval of (dasc_tg - aci) relative width
        bs = boot_stats({l: ev_runs[l] for l in ("aci", "dasc_tg")}, hours_ev, n)
        rel = bs["dasc_tg"][:, 0] / bs["aci"][:, 0]
        # stored bootstrap used dw[idx].mean()/wb[idx].mean() = mean_a/mean_b - ... identical to ratio of means minus 1 via difference
        rel_stored = ref_boot[(ref_boot["dataset"] == name) & (ref_boot["forecaster"] == fname) &
                              (ref_boot["method_a"] == "dasc_clip_tg") & (ref_boot["method_b"] == "aci_clip")].iloc[0]
        lo_r, hi_r = ci(rel - 1.0)
        block_match = abs(lo_r - rel_stored["rel_width_diff_lo"]) < 1e-9 and abs(hi_r - rel_stored["rel_width_diff_hi"]) < 1e-9
        # full bootstrap for the four methods
        boots = boot_stats(ev_runs, hours_ev, n)
        # summaries
        for l in labs:
            s = summarize_run(runs[l], evm[l], S["wbl"])
            cov = hour_cov(runs[l], S, evm[l])
            mad_own = float(np.mean(np.abs(cov - cov.mean())))
            row = {"dataset": name, "forecaster": fname, "method": l, "coverage": s["coverage"], "width_mean": s["width_mean"],
                   "score_mean": s["score_mean"], "worst_block_coverage": s["worst_block_coverage"],
                   "mad_hour": mad_from_cov(cov), "mad_hour_own_mean": mad_own, "min_hour_cov": cov.min(), "max_hour_cov": cov.max(),
                   "hours_below_085": int((cov < 0.85).sum()), "neff_median": s["neff_median"], "n_eval": n,
                   "block_sequence_matches_stored": bool(block_match)}
            if l in ("phase", "cal"):
                row.update({f"cfg_{k}": v for k, v in runs[l + "_cfg"].items()})
            summ_rows.append(row)
            for h in range(24):
                hour_rows.append({"dataset": name, "forecaster": fname, "method": l, "hour": h, "coverage": cov[h]})
        # paired MAD and tests
        def pair(a, b):
            mad_a, mad_b = boots[a][:, 3], boots[b][:, 3]
            pt_a = mad_from_cov(hour_cov(runs[a], S, evm[a]))
            pt_b = mad_from_cov(hour_cov(runs[b], S, evm[b]))
            red = (mad_b - mad_a) / mad_b
            wrel = boots[a][:, 0] / boots[b][:, 0] - 1.0
            cdiff = boots[a][:, 1] - boots[b][:, 1]
            sdiff = boots[a][:, 2] - boots[b][:, 2]
            sa = summarize_run(runs[a], evm[a], S["wbl"])
            sb = summarize_run(runs[b], evm[b], S["wbl"])
            return {"dataset": name, "forecaster": fname, "method_a": a, "method_b": b,
                    "mad_a": pt_a, "mad_b": pt_b, "mad_diff": pt_a - pt_b, "mad_diff_lo": ci(mad_a - mad_b)[0], "mad_diff_hi": ci(mad_a - mad_b)[1],
                    "mad_reduction": (pt_b - pt_a) / pt_b, "mad_reduction_lo": ci(red)[0], "mad_reduction_hi": ci(red)[1],
                    "coverage_a": sa["coverage"], "coverage_b": sb["coverage"],
                    "rel_width_diff": sa["width_mean"] / sb["width_mean"] - 1.0, "rel_width_diff_lo": ci(wrel)[0], "rel_width_diff_hi": ci(wrel)[1],
                    "coverage_diff": sa["coverage"] - sb["coverage"], "coverage_diff_lo": ci(cdiff)[0], "coverage_diff_hi": ci(cdiff)[1],
                    "score_diff": sa["score_mean"] - sb["score_mean"], "score_diff_lo": ci(sdiff)[0], "score_diff_hi": ci(sdiff)[1],
                    "n": n, "n_boot": 2000, "block_len": 168}
        pairs = [pair("phase", "aci"), pair("phase", "dasc_tg"), pair("cal", "aci"), pair("cal", "dasc_tg"),
                 pair("phase", "cal"), pair("dasc_tg", "aci")]
        mad_rows += pairs
        P = {(p["method_a"], p["method_b"]): p for p in pairs}
        # noise floor (no hour effect) from the ACI and the new variants' miss series
        fl = {}
        for l in labs:
            fl[l] = noise_floor(runs[l]["miss"][evm[l]].astype(float), hours_ev)
            floor_rows.append({"dataset": name, "forecaster": fname, "method": l, "floor_mean": fl[l][0], "floor_lo": fl[l][1],
                               "floor_hi": fl[l][2], "mad_hour": mad_from_cov(hour_cov(runs[l], S, evm[l]))})
        max_red = 1.0 - fl["aci"][0] / P[("dasc_tg", "aci")]["mad_b"]
        # hindsight over the evaluated grid
        for kind in KINDS:
            g = pd.read_csv(RESULTS / f"lead2_grid_{name}_{fname}_{kind}.csv")
            ok = g[(g["eval_coverage"] - TARGET).abs() <= 0.01]
            best = ok.sort_values("eval_mad_hour").iloc[0] if len(ok) else g.sort_values("eval_mad_hour").iloc[0]
            hind_rows.append({"dataset": name, "forecaster": fname, "kind": kind, "n_configs": len(g),
                              "best_mad_hindsight": best["eval_mad_hour"], "best_cfg_w": best["w"], "best_cfg_h": best["h"],
                              "best_cfg_gamma": best["gamma"], "best_cfg_fw": best["fw"], "best_cfg_window": best["window"],
                              "best_cfg_cov": best["eval_coverage"], "best_cfg_width": best["eval_width"],
                              "mad_aci": P[("dasc_tg", "aci")]["mad_b"], "hindsight_reduction_vs_aci": 1 - best["eval_mad_hour"] / P[("dasc_tg", "aci")]["mad_b"],
                              "hindsight_reduction_vs_dasc": 1 - best["eval_mad_hour"] / P[("dasc_tg", "aci")]["mad_a"],
                              "n_configs_with_cov_ok": int(len(ok))})
        # cell tests (pre-stated)
        ph, pa, pd_ = P[("phase", "aci")], P[("phase", "dasc_tg")], P[("phase", "cal")]
        t1 = pa_ok = None
        row = {"dataset": name, "forecaster": fname,
               "mad_aci": ph["mad_b"], "mad_dasc_tuned": pa["mad_b"], "mad_phase": ph["mad_a"], "mad_calendar": P[("cal", "aci")]["mad_a"],
               "noise_floor_aci": fl["aci"][0], "max_possible_reduction_vs_aci": max_red,
               "phase_red_vs_aci": ph["mad_reduction"], "phase_red_vs_aci_lo": ph["mad_reduction_lo"], "phase_red_vs_aci_hi": ph["mad_reduction_hi"],
               "phase_red_vs_dasc": pa["mad_reduction"], "phase_red_vs_dasc_lo": pa["mad_reduction_lo"], "phase_red_vs_dasc_hi": pa["mad_reduction_hi"],
               "cal_red_vs_aci": P[("cal", "aci")]["mad_reduction"], "cal_red_vs_dasc": P[("cal", "dasc_tg")]["mad_reduction"],
               "phase_cov": ph["coverage_a"], "phase_rel_width_vs_aci": ph["rel_width_diff"], "phase_rel_width_vs_aci_hi": ph["rel_width_diff_hi"],
               "phase_rel_width_vs_dasc": pa["rel_width_diff"], "phase_rel_width_vs_dasc_hi": pa["rel_width_diff_hi"],
               "phase_minus_cal_mad": pd_["mad_diff"], "phase_minus_cal_lo": pd_["mad_diff_lo"], "phase_minus_cal_hi": pd_["mad_diff_hi"]}
        row["T1_red_vs_aci_ge_30"] = bool(row["phase_red_vs_aci"] >= 0.30)
        row["T2_red_vs_dasc_ge_30"] = bool(row["phase_red_vs_dasc"] >= 0.30)
        row["T3_cov_within_001"] = bool(abs(row["phase_cov"] - TARGET) <= 0.01)
        row["T4_width_le_1pct"] = bool(row["phase_rel_width_vs_aci"] <= 0.01 and row["phase_rel_width_vs_dasc"] <= 0.01)
        row["T4_width_upper95_le_1pct"] = bool(row["phase_rel_width_vs_aci_hi"] <= 0.01 and row["phase_rel_width_vs_dasc_hi"] <= 0.01)
        row["T5_not_matched_by_calendar"] = bool(row["phase_minus_cal_hi"] < 0.0)
        row["CELL_ALIVE"] = bool(row["T1_red_vs_aci_ge_30"] and row["T2_red_vs_dasc_ge_30"] and row["T3_cov_within_001"]
                                 and row["T4_width_le_1pct"] and row["T5_not_matched_by_calendar"])
        row["red_below_10pct"] = bool(min(row["phase_red_vs_aci"], row["phase_red_vs_dasc"]) < 0.10)
        row["CELL_KILL_FLAG"] = bool(row["red_below_10pct"] or not row["T5_not_matched_by_calendar"])
        cell_rows.append(row)
        print(f"[{name}/{fname}] done {time.time() - t0:.0f} s; block sequence matches stored: {block_match}; "
              f"MAD aci {ph['mad_b']:.4f} dasc {pa['mad_b']:.4f} phase {ph['mad_a']:.4f} cal {P[('cal', 'aci')]['mad_a']:.4f}; "
              f"phase red {ph['mad_reduction']:.3f}/{pa['mad_reduction']:.3f}, cal red {P[('cal', 'aci')]['mad_reduction']:.3f}; "
              f"floor {fl['aci'][0]:.4f}", flush=True)
        pd.DataFrame(summ_rows).to_csv(RESULTS / "lead2_summary.csv", index=False)
        pd.DataFrame(hour_rows).to_csv(RESULTS / "lead2_hour_coverage.csv", index=False)
        pd.DataFrame(mad_rows).to_csv(RESULTS / "lead2_mad_paired.csv", index=False)
        pd.DataFrame(floor_rows).to_csv(RESULTS / "lead2_noise_floor.csv", index=False)
        pd.DataFrame(hind_rows).to_csv(RESULTS / "lead2_hindsight.csv", index=False)
        pd.DataFrame(cell_rows).to_csv(RESULTS / "lead2_cell_tests.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        tune_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate_main()
    else:
        print(__doc__)
