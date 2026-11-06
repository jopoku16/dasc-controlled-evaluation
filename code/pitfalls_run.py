"""
pitfalls_run.py: decomposition of evaluation pitfalls for ACI (projected) at the
paper's own tuned settings.  Descriptive: it measures how far each pitfall moves
the apparent width, coverage and interval score of one method; it tests nothing.

Reference in every computed cell: ACI (projected) at the configuration selected
by the paper's tuning rule (results/tuning/<cell>_chosen.csv: pool W*, step
size gamma*), plain weighted-quantile rank, pool that rolls with every new
residual.  Each pitfall changes ONE thing and nothing is re-tuned:

  b_step     gamma = 0.015 (DASC's fixed step) instead of gamma*, same pool W*
  c_rank     finite-sample rank ceil((n+1)(1-alpha_t)) instead of ceil(n(1-alpha_t))
  d_frozen   the pool stops receiving residuals at freeze time (real data: the
             first evaluation step dev_end; simulations: start + W*, so that the
             pool is full); ACI's level update keeps running
  e_half     pool W*/2
  e_double   pool 2 W*
  (a) centring is NOT computed here: pitfalls_reused.csv carries the numbers of
  results/reproduction_check.csv.

Supplement (fixed level, no level update, i.e. the paper's rolling conformal
method with its tuned pool Wr; gamma = 0 reproduces it step for step):
  sc_rank, sd_frozen: the same rank and frozen-pool changes on that method.

Effect = pitfall minus reference; width and interval score relative to the
reference value.  Real data: evaluation steps t >= dev_end, paired
moving-block bootstrap (2000 resamples, same blocks for both runs, 95 per cent
percentile intervals, block length 168, 168, 21 as in the paper).  Simulations:
mean and standard error over the replications of the per-replication relative
change; all steps t >= 220 except the frozen-pool pair, which uses t >= 220 + W.

Usage (from code/):  python -X utf8 pitfalls_run.py real [--xgb]
                     python -X utf8 pitfalls_run.py sim [--seeds N]
                     python -X utf8 pitfalls_run.py reuse
                     python -X utf8 pitfalls_run.py check
At most two worker processes.  Outputs (../results/, new files only):
  pitfalls_real.csv, pitfalls_sim_per_seed.csv, pitfalls_sim_summary.csv,
  pitfalls_reused.csv, pitfalls_anchor_check.csv
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
# data root: DASC_DATA_ROOT, or ../data/real (see datasets.py)

import datasets as ds  # noqa: E402
import pitfalls_lib as pl  # noqa: E402

N_PROC = 2
GAMMA_FIXED = 0.015
COVER_TOL = 0.005          # |coverage change| above this: not at matched coverage

# pitfall id -> (reference run, pitfall run, family, short label)
PITFALLS = {
    "b_step": ("ref", "b", "b", "Step size fixed at 0.015 (pool and rank unchanged)"),
    "c_rank": ("ref", "c", "c", "Finite-sample rank instead of plain rank"),
    "d_frozen": ("ref", "d", "d", "Pool frozen at the first evaluated step (level update kept)"),
    "e_half": ("ref", "eh", "e", "Pool half as long"),
    "e_double": ("ref", "e2", "e", "Pool twice as long"),
    "sc_rank": ("roll_ref", "roll_c", "c", "Fixed-level rolling conformal: finite-sample rank"),
    "sd_frozen": ("roll_ref", "roll_d", "d", "Fixed-level rolling conformal: pool frozen"),
}
FREEZE_PAIRS = {"d_frozen", "sd_frozen"}


def _chosen(path):
    ch = pd.read_csv(path).set_index("method")
    return ch


def run_specs(aci_w, aci_g, roll_w, freeze_a, freeze_r):
    """The run specs of one cell; freeze_a / freeze_r are the freeze times of the two methods."""
    return {
        "ref": dict(window=aci_w, gamma=aci_g),
        "b": dict(window=aci_w, gamma=GAMMA_FIXED),
        "c": dict(window=aci_w, gamma=aci_g, rank="finite"),
        "d": dict(window=aci_w, gamma=aci_g, freeze_at=freeze_a),
        "eh": dict(window=max(aci_w // 2, 1), gamma=aci_g),
        "e2": dict(window=2 * aci_w, gamma=aci_g),
        "roll_ref": dict(window=roll_w, gamma=0.0),
        "roll_c": dict(window=roll_w, gamma=0.0, rank="finite"),
        "roll_d": dict(window=roll_w, gamma=0.0, freeze_at=freeze_r),
    }


def _mean_rank_shift(out, mask):
    a, n = out["alpha_t"][mask], out["pool_size"][mask]
    return float(np.mean(np.ceil((n + 1) * (1 - a) - 1e-9) - np.ceil(n * (1 - a) - 1e-9)))


# ----------------------------------------------------------------------------
# real data
# ----------------------------------------------------------------------------


def run_real_dataset(args):
    name, with_xgb = args
    t0 = time.time()
    cfg = ds.REAL_DATASETS[name]
    meta = cfg["load"]()
    y = meta["y"]
    n = len(y)
    start, dev_end = cfg["start"], int(cfg["dev_frac"] * n)
    preds = cfg["forecasts"](y, meta, with_xgb=with_xgb)
    main = pd.read_csv(RES / "real_summary.csv")
    rows, anchors = [], []
    for fname, pred in preds.items():
        ch = _chosen(RES / "tuning" / f"{name}_{fname}_chosen.csv")
        grid = pd.read_csv(RES / "tuning" / f"{name}_{fname}_grid.csv")
        aw, ag = int(ch.loc["aci_clip", "window"]), float(ch.loc["aci_clip", "gamma"])
        rw = int(ch.loc["rolling", "window"])
        specs = run_specs(aw, ag, rw, dev_end, dev_end)
        outs = {k: pl.run_spec(y, pred, s, start, cfg["clip_zero"]) for k, s in specs.items()}
        mask = outs["ref"]["t"] >= dev_end
        # anchors against the main run and the tuning grid
        for key, meth in (("ref", "aci_clip"), ("roll_ref", "rolling")):
            s = pl.summarize(outs[key], mask)
            ref = main[(main.dataset == name) & (main.forecaster == fname) & (main.method == meth)].iloc[0]
            anchors.append({"kind": "real_vs_real_summary", "dataset": name, "forecaster": fname, "run": key,
                            "d_coverage": s["coverage"] - ref["coverage"], "d_width": s["width_mean"] - ref["width_mean"],
                            "d_score": s["score_mean"] - ref["score_mean"],
                            "ok": bool(abs(s["coverage"] - ref["coverage"]) < 1e-9
                                       and abs(s["width_mean"] - ref["width_mean"]) < 1e-6
                                       and abs(s["score_mean"] - ref["score_mean"]) < 1e-6)})
        ga = grid[(grid.method == "aci_clip") & (grid.window == aw)]
        gref = ga[np.isclose(ga.gamma, ag)].iloc[0]
        gb = ga[np.isclose(ga.gamma, GAMMA_FIXED)].iloc[0]
        for key, g in (("ref", gref), ("b", gb)):
            s = pl.summarize(outs[key], mask)
            anchors.append({"kind": "real_vs_tuning_grid", "dataset": name, "forecaster": fname, "run": key,
                            "d_coverage": s["coverage"] - g["eval_coverage"], "d_width": s["width_mean"] - g["eval_width"],
                            "d_score": s["score_mean"] - g["eval_score"],
                            "ok": bool(abs(s["coverage"] - g["eval_coverage"]) < 1e-6
                                       and abs(s["width_mean"] - g["eval_width"]) < 1e-6
                                       and abs(s["score_mean"] - g["eval_score"]) < 1e-6)})
        for pid, (rk, pk, fam, label) in PITFALLS.items():
            a, b = outs[pk], outs[rk]
            sa, sb = specs[pk], specs[rk]
            d = pl.per_run_deltas(a, b, mask)
            bt = pl.paired_boot(a, b, cfg["block_len"], mask, n_boot=2000, seed=0)
            row = {
                "dataset": name, "forecaster": fname, "pitfall": pid, "family": fam, "label": label,
                "fixed_level_supplement": pid.startswith("s"),
                "ref_window": sb["window"], "ref_gamma": sb["gamma"], "ref_rank": sb.get("rank", "plain"), "ref_freeze": sb.get("freeze_at"),
                "pit_window": sa["window"], "pit_gamma": sa["gamma"], "pit_rank": sa.get("rank", "plain"), "pit_freeze": sa.get("freeze_at"),
                "identical_by_construction": bool(sa == sb),
                "mask": f"t >= {dev_end} (evaluation steps)", "n_eval": int(mask.sum()),
                "cov_ref": d["cov_b"], "cov_pit": d["cov_a"], "width_ref": d["width_b"], "width_pit": d["width_a"],
                "score_ref": d["score_b"], "score_pit": d["score_a"],
                "pool_mean_ref": float(np.mean(b["pool_size"][mask])), "pool_mean_pit": float(np.mean(a["pool_size"][mask])),
                "alpha_hi_cap_ref": pl.summarize(b, mask)["frac_alpha_hi_cap"], "alpha_hi_cap_pit": pl.summarize(a, mask)["frac_alpha_hi_cap"],
                "alpha_lo_cap_ref": pl.summarize(b, mask)["frac_alpha_lo_cap"], "alpha_lo_cap_pit": pl.summarize(a, mask)["frac_alpha_lo_cap"],
                "mean_rank_shift": _mean_rank_shift(a, mask) if sa.get("rank") == "finite" else np.nan,
                "rel_width": bt["rel_width"], "rel_width_lo": bt["rel_width_lo"], "rel_width_hi": bt["rel_width_hi"],
                "cov_change": bt["cov_change"], "cov_change_lo": bt["cov_change_lo"], "cov_change_hi": bt["cov_change_hi"],
                "rel_score": bt["rel_score"], "rel_score_lo": bt["rel_score_lo"], "rel_score_hi": bt["rel_score_hi"],
                "score_change": bt["score_change"], "n_masked": bt["n_masked"], "n_boot": bt["n_boot"], "block_len": bt["block_len"],
                "covered_matched": bool(abs(bt["cov_change"]) <= COVER_TOL),
                "source": "tuning grid numbers, re-run and verified to 1e-6; bootstrap from the re-run" if pid == "b_step"
                else "computed here (pitfalls_run.py)",
            }
            rows.append(row)
        print(f"[{name}/{fname}] done ({time.time() - t0:.0f} s)", flush=True)
    return rows, anchors


def run_real(with_xgb):
    names = ["weather", "electricity", "finance"]
    with Pool(N_PROC) as pool:
        res = pool.map(run_real_dataset, [(nm, with_xgb) for nm in names], chunksize=1)
    rows = [r for rr in res for r in rr[0]]
    anchors = [a for rr in res for a in rr[1]]
    df = pd.DataFrame(rows)
    order = {"electricity": 0, "weather": 1, "finance": 2}
    df = df.sort_values(["dataset", "forecaster"], key=lambda s: s.map(order) if s.name == "dataset" else s, kind="stable")
    df.to_csv(RES / "pitfalls_real.csv", index=False)
    _write_anchors(pd.DataFrame(anchors), "real")
    print("wrote pitfalls_real.csv", len(df), "rows")


def _write_anchors(df, kind):
    path = RES / "pitfalls_anchor_check.csv"
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["kind"].str.startswith(kind)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


# ----------------------------------------------------------------------------
# simulations
# ----------------------------------------------------------------------------

SIM = ds.SIM_CFG


def sim_task(args):
    scenario, seeds = args
    rows, anchors = [], []
    chosen = {}
    for seed in seeds:
        y = ds.make_sim(scenario, seed)["y"]
        preds = ds.sim_forecasts(y)
        for fname, pred in preds.items():
            if fname not in chosen:
                chosen[fname] = _chosen(RES / "tuning" / f"sim_{scenario}_{fname}_chosen.csv")
            ch = chosen[fname]
            aw, ag = int(ch.loc["aci_clip", "window"]), float(ch.loc["aci_clip", "gamma"])
            rw = int(ch.loc["rolling", "window"])
            start = SIM["start"]
            specs = run_specs(aw, ag, rw, start + aw, start + rw)
            outs = {k: pl.run_spec(y, pred, s, start, SIM["clip_zero"]) for k, s in specs.items()}
            t = outs["ref"]["t"]
            std = t >= start
            for key, meth in (("ref", "aci_clip"), ("roll_ref", "rolling")):
                s = pl.summarize(outs[key], std)
                anchors.append({"scenario": scenario, "seed": seed, "forecaster": fname, "run": key, "method": meth,
                                "coverage": s["coverage"], "width_mean": s["width_mean"], "score_mean": s["score_mean"]})
            for pid, (rk, pk, fam, label) in PITFALLS.items():
                a, b = outs[pk], outs[rk]
                sa, sb = specs[pk], specs[rk]
                m0 = sa.get("freeze_at") if pid in FREEZE_PAIRS else start
                mask = t >= m0
                d = pl.per_run_deltas(a, b, mask)
                rows.append({
                    "scenario": scenario, "seed": seed, "forecaster": fname, "pitfall": pid, "family": fam,
                    "mask_start": int(m0), "ref_window": sb["window"], "ref_gamma": sb["gamma"],
                    "pit_window": sa["window"], "pit_gamma": sa["gamma"], "n_eval": int(mask.sum()),
                    "cov_ref": d["cov_b"], "cov_pit": d["cov_a"], "width_ref": d["width_b"], "width_pit": d["width_a"],
                    "score_ref": d["score_b"], "score_pit": d["score_a"], "rel_width": d["rel_width"],
                    "cov_change": d["cov_change"], "score_change": d["score_change"], "rel_score": d["rel_score"],
                    "n_inf_pit": d["n_inf_a"], "n_inf_ref": d["n_inf_b"],
                    "identical_by_construction": bool(sa == sb),
                    "mean_rank_shift": _mean_rank_shift(a, mask) if sa.get("rank") == "finite" else np.nan,
                })
    return rows, anchors


def summarize_sim(per_seed):
    g = per_seed.groupby(["scenario", "forecaster", "pitfall"], sort=False)
    cols = ["rel_width", "cov_change", "rel_score", "score_change", "cov_ref", "cov_pit", "width_ref", "width_pit",
            "score_ref", "score_pit"]
    out = pd.concat([g[cols].mean(), g[["rel_width", "cov_change", "rel_score", "score_change"]].sem().add_suffix("_se"),
                     g.size().rename("n_reps"), g["identical_by_construction"].first(), g["mask_start"].first(),
                     g["ref_window"].first(), g["ref_gamma"].first(), g["pit_window"].first(), g["pit_gamma"].first(),
                     g["n_inf_pit"].sum().rename("n_inf_pit_total"), g["mean_rank_shift"].mean()], axis=1).reset_index()
    out["covered_matched"] = out["cov_change"].abs() <= COVER_TOL
    out["family"] = per_seed.groupby(["scenario", "forecaster", "pitfall"], sort=False)["family"].first().to_numpy()
    out["mask"] = [f"t >= {int(m)}" for m in out["mask_start"]]
    return out


def run_sim(n_seeds):
    t0 = time.time()
    seeds = ds.EVAL_SEEDS[:n_seeds]
    chunks = [seeds[i:i + 5] for i in range(0, len(seeds), 5)]
    tasks = [(sc, ch) for sc in ds.SIM_SCENARIOS for ch in chunks]
    rows, anchors = [], []
    with Pool(N_PROC) as pool:
        for k, (r, a) in enumerate(pool.imap_unordered(sim_task, tasks, chunksize=1)):
            rows += r
            anchors += a
            if (k + 1) % 12 == 0:
                print(f"  {k + 1}/{len(tasks)} tasks ({time.time() - t0:.0f} s)", flush=True)
    per_seed = pd.DataFrame(rows).sort_values(["scenario", "forecaster", "pitfall", "seed"], kind="stable")
    # keep scenario order of the paper
    order = {s: i for i, s in enumerate(ds.SIM_SCENARIOS)}
    per_seed = per_seed.sort_values(["scenario", "forecaster", "pitfall", "seed"], key=lambda s: s.map(order) if s.name == "scenario" else s, kind="stable")
    per_seed.to_csv(RES / "pitfalls_sim_per_seed.csv", index=False)
    summary = summarize_sim(per_seed)
    summary["n_reps_requested"] = n_seeds
    summary.to_csv(RES / "pitfalls_sim_summary.csv", index=False)
    # anchors against the paper's per-seed file
    an = pd.DataFrame(anchors)
    ps = pd.read_csv(RES / "sim_per_seed.csv")
    m = an.merge(ps[["scenario", "seed", "forecaster", "method", "coverage", "width_mean", "score_mean"]],
                 on=["scenario", "seed", "forecaster", "method"], suffixes=("", "_paper"))
    m["d_coverage"] = m["coverage"] - m["coverage_paper"]
    m["d_width"] = m["width_mean"] - m["width_mean_paper"]
    m["d_score"] = m["score_mean"] - m["score_mean_paper"]
    m["ok"] = (m.d_coverage.abs() < 1e-9) & (m.d_width.abs() < 1e-6) & (m.d_score.abs() < 1e-6)
    m.insert(0, "kind", "sim_vs_sim_per_seed")
    keep = ["kind", "scenario", "seed", "forecaster", "run", "method", "d_coverage", "d_width", "d_score", "ok"]
    _write_anchors(m[keep], "sim")
    print(f"wrote sim files: {len(per_seed)} per-seed rows, anchors ok {int(m.ok.sum())}/{len(m)} ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------
# reused rows: (a) centring from the reproduction anchor
# ----------------------------------------------------------------------------


def run_reuse():
    rc = pd.read_csv(RES / "reproduction_check.csv")
    PIT = "as published (DASC centred on y[t-1]), dasc_lib features"
    REF = "recentred on y[t-24] (same forecaster as baselines)"
    rows = []
    for dset in ("electricity", "weather", "finance"):
        sub = rc[rc.dataset == dset]
        p, r = sub[sub.variant == PIT], sub[sub.variant == REF]
        aci = sub[sub.variant == "as published"]
        base = {"dataset": dset, "forecaster": "naive24" if dset != "finance" else "naive1", "pitfall": "a_centring",
                "family": "a", "label": "DASC centred on y[t-1] instead of the shared forecast",
                "source": "results/reproduction_check.csv (not recomputed)"}
        if len(p) and len(r):
            p, r = p.iloc[0], r.iloc[0]
            rows.append({**base, "reference": "DASC recentred on y[t-24] (the baselines' forecaster)",
                         "pitfall_def": "DASC as published, centred on y[t-1], new feature code",
                         "cov_ref": r.coverage, "cov_pit": p.coverage, "width_ref": r.avg_width, "width_pit": p.avg_width,
                         "rel_width": p.avg_width / r.avg_width - 1, "cov_change": p.coverage - r.coverage,
                         "aci_published_cov": float(aci.coverage.iloc[0]) if len(aci) else np.nan,
                         "aci_published_width": float(aci.avg_width.iloc[0]) if len(aci) else np.nan,
                         "applicable": True,
                         "note": "single deterministic run, no interval; interval score not in the file"
                                 + ("; coverage differs by more than 5 points, so the width change is not at matched coverage"
                                    if abs(p.coverage - r.coverage) > 0.05 else "")})
        else:
            rows.append({**base, "applicable": False,
                         "note": "no centring difference: the finance forecaster is y[t-1] for every method, so the pitfall cannot arise"})
    rows.append({"dataset": "simulations", "forecaster": "naive1, ridge", "pitfall": "a_centring", "family": "a",
                 "label": "DASC centred on y[t-1] instead of the shared forecast", "applicable": False,
                 "source": "results/reproduction_check.csv (not recomputed)",
                 "note": "reproduction_check.csv has no recentred simulation row (synthetic rows are published-versus-published), so no number"})
    pd.DataFrame(rows).to_csv(RES / "pitfalls_reused.csv", index=False)
    print("wrote pitfalls_reused.csv")


def main(argv):
    if "reuse" in argv:
        run_reuse()
    if "real" in argv:
        run_real("--xgb" in argv)
    if "sim" in argv:
        n = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 50
        run_sim(n)


if __name__ == "__main__":
    main(sys.argv[1:])
