"""
ext_lib.py: shared machinery of the extended-grid experiment (extended step-size set, bandwidths scaled to the
feature spread).  Nothing here edits or imports-and-patches the existing study code; it only calls
dasc_lib, tuning, metrics and datasets.

Design (see ../results/ext_design.txt):
  * one step-size set for every method with a step: GAMMAS7
  * bandwidth set per feature window: {0.35, 0.55, 0.8} + c x (median pairwise feature distance
    on the development steps), c in {0.25, 0.5, 1, 2}
  * tuning runs stop at dev_end (the development score is causal in the steps before dev_end)
  * at most N_PROC worker processes
"""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "data" / "real")   # same default as datasets.py
os.environ.setdefault("DASC_DATA_ROOT", DEFAULT_DATA)

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

import tuning
from dasc_lib import make_method, run_stream

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_PROC = 2
GAMMAS7 = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]
H_ORIG = [0.35, 0.55, 0.80]
C_SCALE = [0.25, 0.5, 1.0, 2.0]
LAMBDAS = [0.15, 0.45]
MMIN = [0.25, 0.5]
KP7 = GAMMAS7
CFG_KEYS = ["window", "gamma", "kp", "ki", "kd", "decay", "h", "fw", "drift_lambda", "m_min_frac"]


# ----------------------------------------------------------------------------
# bandwidth scale and grids
# ----------------------------------------------------------------------------


def median_pairwise(F, lo, hi):
    """Median Euclidean distance between the feature vectors F[lo:hi] (all pairs)."""
    return float(np.median(pdist(F[lo:hi])))


def make_hgrid(medians):
    """medians: dict fw -> median pairwise distance.  Returns dict fw -> sorted list of h values."""
    out = {}
    for fw, m in medians.items():
        vals = list(H_ORIG) + [float(f"{c * m:.4g}") for c in C_SCALE]
        out[int(fw)] = sorted(set(vals))
    return out


def h_label(h, med):
    return h / med


def build_baseline_grid(name, pools, fws, hgrid):
    """Full-factorial grids of the extended study (own builder; tuning.build_grid is not used)."""
    pools, fws = list(pools), list(fws)
    if name in ("aci_clip", "aci_noclip"):
        dims = [{"window": w, "gamma": g} for w in pools for g in GAMMAS7]
    elif name == "pid":
        dims = [{"window": w, "kp": kp, "ki": ki, "kd": kd}
                for w in pools for kp in KP7 for ki in (0.001, 0.003) for kd in (0.0, 0.01)]
    elif name == "spectral_fixed":
        dims = [{"window": w, "h": h, "fw": fw} for w in pools for fw in fws for h in hgrid[fw]]
    elif name == "spectral_aci":
        dims = [{"window": w, "h": h, "gamma": g, "fw": fw}
                for w in pools for fw in fws for h in hgrid[fw] for g in GAMMAS7]
    else:
        raise ValueError(name)
    return dims


def dasc_stage0_grid(pools, fws, hgrid, gamma=0.015):
    return [{"window": w, "h": h, "drift_lambda": lam, "m_min_frac": mm, "fw": fw, "gamma": gamma}
            for w in pools for fw in fws for h in hgrid[fw] for lam in LAMBDAS for mm in MMIN]


def dasc_stage1_grid(base):
    return [{**base, "gamma": g} for g in GAMMAS7]


def dasc_stage2_grid(base, gamma, hgrid, with_gamma=True):
    out = []
    for h in hgrid[int(base["fw"])]:
        for lam in LAMBDAS:
            c = {**base, "h": h, "drift_lambda": lam}
            if with_gamma:
                c["gamma"] = gamma
            out.append(c)
    return out


# ----------------------------------------------------------------------------
# worker jobs (streams are set once per process by tuning._init)
# ----------------------------------------------------------------------------


def _build(name, cfg, s):
    p = tuning.params_from_config(cfg)
    m = make_method(name, alpha=0.10, first_valid=1, **p)
    f = s["feats"][int(cfg.get("fw", s["default_fw"]))] if m.needs_features else None
    return m, f


def dev_job(args):
    """Run one configuration on one stream up to dev_end only; development summary."""
    tag, name, ci, cfg, si = args
    s = tuning._STREAMS[si]
    m, f = _build(name, cfg, s)
    out = run_stream(s["y"], s["pred"], m, features=f, start=s["start"], end=s["dev_end"],
                     clip_zero=s["clip_zero"])
    sc = out["score"]
    w = out["width"]
    return {"tag": tag, "config_id": ci, "stream": si,
            "dev_score": float(sc.mean()) if len(sc) else np.nan,
            "dev_coverage": float(1 - out["miss"].mean()) if len(sc) else np.nan,
            "dev_width": float(w.mean()) if len(sc) else np.nan,
            "dev_n_inf": int(np.sum(~np.isfinite(w))),
            "dev_neff_med": float(np.nanmedian(out["neff"])) if len(sc) else np.nan,
            "dev_neff_min": float(np.nanmin(out["neff"])) if len(sc) else np.nan}


def full_job(args):
    """Full-stream run of a chosen configuration; returns the per-step arrays."""
    label, name, cfg, si = args
    s = tuning._STREAMS[si]
    m, f = _build(name, cfg, s)
    out = run_stream(s["y"], s["pred"], m, features=f, start=s["start"], clip_zero=s["clip_zero"])
    return label, out


def evaluate(pool, jobs, n_streams):
    """jobs: list of (tag, name, config_id, cfg).  Returns DataFrame averaged over streams, one row per
    (tag, config_id)."""
    tasks = [(tag, name, ci, cfg, si) for (tag, name, ci, cfg) in jobs for si in range(n_streams)]
    rows = list(pool.imap_unordered(dev_job, tasks, chunksize=1))
    df = pd.DataFrame(rows)
    agg = df.groupby(["tag", "config_id"]).agg(
        dev_score=("dev_score", "mean"), dev_coverage=("dev_coverage", "mean"),
        dev_width=("dev_width", "mean"), dev_n_inf=("dev_n_inf", "sum"),
        dev_neff_med=("dev_neff_med", "mean"), dev_neff_min=("dev_neff_min", "min"),
        n_streams=("stream", "size")).reset_index()
    return agg


def pick_best(agg_tag, extra_order=()):
    """Lowest development score; ties go to the lower config_id (and stage), as in the main study."""
    cols = ["dev_score"] + list(extra_order) + ["config_id"]
    return agg_tag.sort_values(cols, kind="mergesort").iloc[0]


# ----------------------------------------------------------------------------
# one-cell tuning
# ----------------------------------------------------------------------------

BASELINES = ["aci_clip", "aci_noclip", "pid", "spectral_fixed", "spectral_aci"]
DASC_GAMMA = ["dasc_clip", "dasc_noclip"]


def _annotate(df, cfgs_by_tag):
    for k in CFG_KEYS:
        df[k] = [cfgs_by_tag[(t, c)].get(k, np.nan) for t, c in zip(df["tag"], df["config_id"])]
    return df


def tune_cell(pool, n_streams, pools, fws, hgrid, main_chosen, wide=True, log=print):
    """Tune every method of the extended study on the development data of one cell.

    main_chosen: dict method -> main-run chosen configuration (base for the two-stage DASC tuning).
    Returns (grid DataFrame, chosen dict label -> (method name, config), budgets dict label -> dict).
    Labels: aci_clip, aci_noclip, pid, spectral_fixed, spectral_aci, dasc_clip, dasc_noclip,
    dasc_fixed, and with wide=True dasc_clip_wide, dasc_noclip_wide.
    """
    t0 = time.time()
    cfgs = {}               # (tag, config_id) -> cfg
    grids = []
    chosen, budgets = {}, {}

    # ---- baselines: one batch, full factorial
    jobs = []
    for name in BASELINES:
        g = build_baseline_grid(name, pools, fws, hgrid)
        for ci, c in enumerate(g):
            cfgs[(name, ci)] = c
            jobs.append((name, name, ci, c))
    # ---- DASC stage 1 (gamma at the main-run base) and stage-0-free fixed-level h x lambda
    for v in DASC_GAMMA:
        base = {k: main_chosen[v][k] for k in ("window", "h", "fw", "drift_lambda", "m_min_frac")}
        base["fw"] = int(base["fw"]); base["window"] = int(base["window"])
        for ci, c in enumerate(dasc_stage1_grid(base)):
            cfgs[(v + "|s1", ci)] = c
            jobs.append((v + "|s1", v, ci, c))
    base_f = {k: main_chosen["dasc_fixed"][k] for k in ("window", "h", "fw", "drift_lambda", "m_min_frac")}
    base_f["fw"] = int(base_f["fw"]); base_f["window"] = int(base_f["window"])
    for ci, c in enumerate(dasc_stage2_grid(base_f, None, hgrid, with_gamma=False)):
        cfgs[("dasc_fixed|s2", ci)] = c
        jobs.append(("dasc_fixed|s2", "dasc_fixed", ci, c))
    # ---- wide sensitivity, stage 0
    if wide:
        for v in DASC_GAMMA:
            for ci, c in enumerate(dasc_stage0_grid(pools, fws, hgrid)):
                cfgs[(v + "|w0", ci)] = c
                jobs.append((v + "|w0", v, ci, c))
    log(f"   batch 1: {len(jobs)} configurations x {n_streams} stream(s)")
    agg = evaluate(pool, jobs, n_streams)
    log(f"   batch 1 done ({time.time() - t0:.0f} s)")
    agg["stage"] = agg["tag"].map(lambda t: t.split("|")[1] if "|" in t else "full")
    agg["label"] = agg["tag"].map(lambda t: t.split("|")[0])
    agg = _annotate(agg, cfgs)
    grids.append(agg)

    for name in BASELINES:
        sub = agg[agg["tag"] == name]
        best = pick_best(sub)
        chosen[name] = (name, cfgs[(name, int(best["config_id"]))])
        budgets[name] = {"n_configs": int(len(sub)), "stages": "factorial", "dev_score_grid": float(best["dev_score"])}
    for v in DASC_GAMMA:
        s1 = agg[agg["tag"] == v + "|s1"]
        b1 = pick_best(s1)
        g_star = cfgs[(v + "|s1", int(b1["config_id"]))]["gamma"]
        base = cfgs[(v + "|s1", 0)]
        base = {k: base[k] for k in ("window", "h", "fw", "drift_lambda", "m_min_frac")}
        # stage 2 for the primary design
        g2 = dasc_stage2_grid(base, g_star, hgrid)
        for ci, c in enumerate(g2):
            cfgs[(v + "|s2", ci)] = c
        a2 = evaluate(pool, [(v + "|s2", v, ci, c) for ci, c in enumerate(g2)], n_streams)
        a2["stage"] = "s2"; a2["label"] = v
        a2 = _annotate(a2, cfgs)
        grids.append(a2)
        both = pd.concat([s1.assign(_stage=1), a2.assign(_stage=2)], ignore_index=True)
        bb = both.sort_values(["dev_score", "_stage", "config_id"], kind="mergesort").iloc[0]
        key = (v + "|s1", int(bb["config_id"])) if bb["_stage"] == 1 else (v + "|s2", int(bb["config_id"]))
        chosen[v] = (v, cfgs[key])
        budgets[v] = {"n_configs": 48 + len(s1) + len(a2), "stages": f"main 48 + stage1 {len(s1)} + stage2 {len(a2)}",
                      "stage1_gamma": g_star, "dev_score_grid": float(bb["dev_score"])}
        log(f"   {v}: stage-1 gamma* = {g_star:g}; final {chosen[v][1]}")
        # wide: stage 0 winner -> stages 1 and 2
        if wide:
            w0 = agg[agg["tag"] == v + "|w0"]
            bw = pick_best(w0)
            base_w = cfgs[(v + "|w0", int(bw["config_id"]))]
            base_w = {k: base_w[k] for k in ("window", "h", "fw", "drift_lambda", "m_min_frac")}
            g1w = dasc_stage1_grid(base_w)
            for ci, c in enumerate(g1w):
                cfgs[(v + "|w1", ci)] = c
            a1w = evaluate(pool, [(v + "|w1", v, ci, c) for ci, c in enumerate(g1w)], n_streams)
            a1w["stage"] = "w1"; a1w["label"] = v
            a1w = _annotate(a1w, cfgs)
            grids.append(a1w)
            b1w = pick_best(a1w)
            gw = cfgs[(v + "|w1", int(b1w["config_id"]))]["gamma"]
            g2w = dasc_stage2_grid(base_w, gw, hgrid)
            for ci, c in enumerate(g2w):
                cfgs[(v + "|w2", ci)] = c
            a2w = evaluate(pool, [(v + "|w2", v, ci, c) for ci, c in enumerate(g2w)], n_streams)
            a2w["stage"] = "w2"; a2w["label"] = v
            a2w = _annotate(a2w, cfgs)
            grids.append(a2w)
            bothw = pd.concat([w0.assign(_stage=0), a1w.assign(_stage=1), a2w.assign(_stage=2)], ignore_index=True)
            # stage 0 ran at gamma = 0.015 and stage 1 repeats the winner at 0.015; pick the lowest score
            order = {0: 0, 1: 1, 2: 2}
            bbw = bothw.sort_values(["dev_score", "_stage", "config_id"], kind="mergesort").iloc[0]
            tagmap = {0: v + "|w0", 1: v + "|w1", 2: v + "|w2"}
            chosen[v + "_wide"] = (v, cfgs[(tagmap[int(bbw["_stage"])], int(bbw["config_id"]))])
            budgets[v + "_wide"] = {"n_configs": len(w0) + len(a1w) + len(a2w),
                                    "stages": f"stage0 {len(w0)} + stage1 {len(a1w)} + stage2 {len(a2w)}",
                                    "stage1_gamma": gw, "dev_score_grid": float(bbw["dev_score"])}
            log(f"   {v}_wide: final {chosen[v + '_wide'][1]}")
    # fixed-level DASC: h x lambda at the main-run base
    sf = agg[agg["tag"] == "dasc_fixed|s2"]
    bf = pick_best(sf)
    chosen["dasc_fixed"] = ("dasc_fixed", cfgs[("dasc_fixed|s2", int(bf["config_id"]))])
    budgets["dasc_fixed"] = {"n_configs": 48 + len(sf), "stages": f"main 48 + h x lambda {len(sf)}",
                          "dev_score_grid": float(bf["dev_score"])}
    grid = pd.concat(grids, ignore_index=True)
    # mark selected rows
    sel = set()
    for lab, (nm, cfg) in chosen.items():
        for (tag, ci), c in cfgs.items():
            if c == cfg and tag.split("|")[0] == nm:
                sel.add((lab, tag, ci))
    grid["selected_for"] = ""
    for lab, tag, ci in sel:
        m = (grid["tag"] == tag) & (grid["config_id"] == ci)
        grid.loc[m, "selected_for"] = grid.loc[m, "selected_for"].map(lambda s: (s + "," if s else "") + lab)
    log(f"   tuning done ({time.time() - t0:.0f} s); chosen: " + ", ".join(sorted(chosen)))
    return grid, chosen, budgets


def edge_flag(value, grid_values):
    gv = sorted(grid_values)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if np.isclose(value, gv[0]):
        return "bottom"
    if np.isclose(value, gv[-1]):
        return "top"
    return ""


def neff_stats(out, mask):
    """Median n_eff, mean gated pool, median n_eff/pool and the share of steps with n_eff < 30."""
    neff = out["neff"][mask]
    ps = out["pool_size"][mask].astype(float)
    ok = np.isfinite(neff) & np.isfinite(ps) & (ps > 0)
    return {"neff_median": float(np.median(neff[ok])) if ok.any() else np.nan,
            "pool_median": float(np.median(ps[ok])) if ok.any() else np.nan,
            "neff_over_pool_median": float(np.median(neff[ok] / ps[ok])) if ok.any() else np.nan,
            "neff_min": float(np.min(neff[ok])) if ok.any() else np.nan,
            "frac_neff_lt30": float(np.mean(neff[ok] < 30)) if ok.any() else np.nan}
