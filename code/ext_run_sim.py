"""
ext_run_sim.py: the extended-grid experiment on simulated scenarios (extended step-size set, scaled bandwidths).

Usage: python -X utf8 ext_run_sim.py recurring_freq_noise [scenario ...] [--seeds 50] [--no-wide] [--sweep] [--force]

Same protocol as run_dascgamma.py: tuning on the ten development seeds (full streams), frozen, evaluation on
seeds 0 .. n_seeds-1.  Scaled bandwidths use, per feature window, the median over the ten development streams of
each stream's median pairwise feature distance.  --sweep also runs the matched-step sweep (ACI, spectral ACI and
DASC at every gamma of the extended set) used to test "the weights substitute for a faster step".

Outputs (new files only):
  results/tuning/ext_sim_<scenario>_<fc>_grid.csv, _chosen.csv
  results/ext_parts/sim_per_seed_<scenario>_<fc>.csv, sim_summary_<...>.csv, sim_paired_<...>.csv
  results/ext_parts/sim_sweep_<scenario>_<fc>.csv (with --sweep)
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
from metrics import summarize_run
from run_simulations import build_stream

R = E.RESULTS
PARTS = R / "ext_parts"
CFG = ds.SIM_CFG
PAIRS = [
    ("dasc_clip_ext", "aci_clip_ext"), ("dasc_clip_ext", "spectral_aci_ext"), ("dasc_clip_ext", "decay"),
    ("dasc_clip_ext", "pid_ext"), ("dasc_noclip_ext", "aci_noclip_ext"), ("dasc_noclip_ext", "spectral_aci_ext"),
    ("dasc_noclip_ext", "decay"), ("spectral_aci_ext", "aci_clip_ext"), ("dasc_fixed_ext", "spectral_fixed_ext"),
    ("dasc_clip_ext", "dasc_clip_prev"), ("aci_clip_ext", "aci_clip_main"),
    ("dasc_clip_wide", "aci_clip_ext"), ("dasc_clip_wide", "spectral_aci_ext"), ("dasc_clip_wide", "decay"),
    ("dasc_noclip_wide", "aci_noclip_ext"), ("dasc_clip_wide", "dasc_clip_ext"),
]


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


def _eval_seed(args):
    scenario, seed, runs_by_fc = args
    y, preds, feats, regime = build_stream(scenario, seed)
    rows, pair_rows = [], []
    for fname, pred in preds.items():
        if fname not in runs_by_fc:
            continue
        outs = {}
        for label, (m, c) in runs_by_fc[fname].items():
            out = tuning.run_chosen(y, pred, feats, m, c, CFG["start"], CFG["clip_zero"], CFG["default_fw"])
            outs[label] = out
            s = summarize_run(out, None, CFG["worst_block_len"])
            row = {"scenario": scenario, "seed": seed, "forecaster": fname, "variant": label, "method": m,
                   "gamma": c.get("gamma", c.get("kp", np.nan)), "coverage": s["coverage"],
                   "width_mean": s["width_mean"], "score_mean": s["score_mean"],
                   "worst_block_coverage": s["worst_block_coverage"], "n_inf_width": s["n_inf_width"]}
            neff = out["neff"]; ps = out["pool_size"].astype(float)
            ok = np.isfinite(neff) & np.isfinite(ps) & (ps > 0)
            row["neff_median"] = float(np.median(neff[ok])) if ok.any() else np.nan
            row["pool_median"] = float(np.median(ps[ok])) if ok.any() else np.nan
            reg = regime[out["t"]]
            for g in np.unique(reg):
                row[f"cov_{g}"] = 1 - out["miss"][reg == g].mean()
            rows.append(row)
        for a, b in PAIRS:
            if a in outs and b in outs:
                wa, wb = _fmean(outs[a]["width"]), _fmean(outs[b]["width"])
                pair_rows.append({"scenario": scenario, "seed": seed, "forecaster": fname, "method_a": a, "method_b": b,
                                  "width_diff": wa - wb, "rel_width_diff": (wa - wb) / wb,
                                  "coverage_diff": outs[b]["miss"].mean() - outs[a]["miss"].mean(),
                                  "score_diff": _fmean(outs[a]["score"]) - _fmean(outs[b]["score"])})
    return rows, pair_rows


def run_scenario(scenario, fcs, n_seeds, wide, sweep, force):
    t0 = time.time()
    dev = [build_stream(scenario, s) for s in ds.TUNING_SEEDS]
    fnames = [f for f in dev[0][1] if f in fcs or ("naive" in fcs and f.startswith("naive"))]
    main_per_seed = pd.read_csv(R / "sim_per_seed.csv")
    main_per_seed = main_per_seed[main_per_seed["scenario"] == scenario]
    prev_per_seed = pd.read_csv(R / "sim_per_seed_dascgamma.csv")
    prev_per_seed = prev_per_seed[prev_per_seed["scenario"] == scenario]
    for fname in fnames:
        key = f"{scenario}_{fname}"
        if (PARTS / f"sim_summary_{key}.csv").exists() and not force:
            print(f"[{key}] exists, skipped", flush=True)
            continue
        t1 = time.time()
        streams = [{"y": y, "pred": preds[fname], "feats": feats, "start": CFG["start"], "dev_end": len(y),
                    "clip_zero": CFG["clip_zero"], "wbl": CFG["worst_block_len"], "default_fw": CFG["default_fw"]}
                   for (y, preds, feats, _) in dev]
        medians = {fw: float(np.median([E.median_pairwise(f[2][fw], CFG["start"], len(f[0])) for f in dev]))
                   for fw in CFG["fws"]}
        hgrid = E.make_hgrid(medians)
        pd.DataFrame([{"scenario": scenario, "forecaster": fname, "fw": fw, "median_of_medians": medians[fw],
                       **{f"h_{i + 1}": h for i, h in enumerate(hgrid[fw])}} for fw in CFG["fws"]]).to_csv(
            R / "tuning" / f"ext_sim_{key}_bandwidth.csv", index=False)
        chm = pd.read_csv(R / "tuning" / f"sim_{key}_chosen.csv").set_index("method")
        main = {m: _cfg_from_row(chm.loc[m]) for m in chm.index}
        chp = pd.read_csv(R / "tuning" / f"sim_{key}_dascgamma_chosen.csv").set_index("variant")
        prev = {m: _cfg_from_row(chp.loc[m]) for m in chp.index}
        print(f"[{key}] medians {medians}; hgrid {hgrid}", flush=True)
        with Pool(E.N_PROC, initializer=tuning._init, initargs=(streams,)) as pool:
            grid, chosen, budgets = E.tune_cell(pool, len(streams), CFG["pools"], CFG["fws"], hgrid, main,
                                                wide=wide, log=lambda s: print(s, flush=True))
        grid.insert(0, "forecaster", fname); grid.insert(0, "scenario", scenario)
        grid.to_csv(R / "tuning" / f"ext_sim_{key}_grid.csv", index=False)
        runs = {}
        for lab, (nm, c) in chosen.items():
            suffix = "_wide" if lab.endswith("_wide") else "_ext"
            runs[lab.replace("_wide", "") + suffix] = (nm, c)
        for m in ("rolling", "decay"):
            runs[m] = (m, main[m])
        for m in ("aci_clip", "aci_noclip", "pid", "spectral_fixed", "spectral_aci", "dasc_fixed", "dasc_clip", "dasc_noclip"):
            runs[m + "_main"] = (m, main[m])
        runs["dasc_clip_prev"] = ("dasc_clip", prev["dasc_clip"])
        runs["dasc_noclip_prev"] = ("dasc_noclip", prev["dasc_noclip"])
        pd.DataFrame([{"scenario": scenario, "forecaster": fname, "variant": lab, "method": m, **{k: c.get(k, np.nan) for k in E.CFG_KEYS},
                       **budgets.get(lab.replace("_ext", ""), {})} for lab, (m, c) in runs.items()
                      if lab.endswith("_ext") or lab.endswith("_wide")]).to_csv(
            R / "tuning" / f"ext_sim_{key}_chosen.csv", index=False)
        if sweep:
            aci_c = runs["aci_clip_ext"][1]
            sac = runs["spectral_aci_ext"][1]
            dac = runs["dasc_clip_ext"][1]
            for g in E.GAMMAS7:
                runs[f"sweep_aci|{g}"] = ("aci_clip", {"window": aci_c["window"], "gamma": g})
                runs[f"sweep_aci_dpool|{g}"] = ("aci_clip", {"window": dac["window"], "gamma": g})
                runs[f"sweep_saci|{g}"] = ("spectral_aci", {**sac, "gamma": g})
                runs[f"sweep_dasc|{g}"] = ("dasc_clip", {**dac, "gamma": g})
        print(f"[{key}] tuning + setup done ({time.time() - t1:.0f} s); evaluating {n_seeds} seeds, {len(runs)} runs/seed", flush=True)
        with Pool(E.N_PROC) as pool:
            res = pool.map(_eval_seed, [(scenario, s, {fname: runs, **{}}) for s in ds.EVAL_SEEDS[:n_seeds]], chunksize=1)
        per_seed = pd.DataFrame([r for rr in res for r in rr[0] if r["forecaster"] == fname])
        pairs = pd.DataFrame([r for rr in res for r in rr[1] if r["forecaster"] == fname])
        # reproduction check: re-run main-run and previous-study configurations against the stored per-seed rows
        chk = []
        for lab in per_seed["variant"].unique():
            if lab in ("rolling", "decay") or lab.endswith("_main"):
                ref = main_per_seed[(main_per_seed["forecaster"] == fname) & (main_per_seed["method"] == runs[lab][0])]
                mine = per_seed[per_seed["variant"] == lab]
            elif lab.endswith("_prev"):
                ref = prev_per_seed[(prev_per_seed["forecaster"] == fname) & (prev_per_seed["variant"] == runs[lab][0] + "_tg")]
                mine = per_seed[per_seed["variant"] == lab]
            else:
                continue
            mg = mine.merge(ref[["seed", "coverage", "width_mean"]], on="seed", suffixes=("", "_ref"))
            chk.append({"scenario": scenario, "forecaster": fname, "variant": lab, "n_seeds": len(mg),
                        "max_abs_cov_diff": float((mg["coverage"] - mg["coverage_ref"]).abs().max()),
                        "max_abs_width_diff": float((mg["width_mean"] - mg["width_mean_ref"]).abs().max())})
        pd.DataFrame(chk).to_csv(PARTS / f"sim_repro_{key}.csv", index=False)
        per_seed.to_csv(PARTS / f"sim_per_seed_{key}.csv", index=False)
        cols = ["coverage", "width_mean", "score_mean", "worst_block_coverage", "n_inf_width", "neff_median", "pool_median"]
        reg = [c for c in per_seed.columns if c.startswith("cov_")]
        g = per_seed.groupby(["scenario", "forecaster", "variant", "method"])
        summ = pd.concat([g[cols + reg].mean(), g[cols + reg].sem().add_suffix("_se"), g["gamma"].first(),
                          g.size().rename("n_reps")], axis=1).reset_index()
        summ.to_csv(PARTS / f"sim_summary_{key}.csv", index=False)
        pc = ["width_diff", "rel_width_diff", "coverage_diff", "score_diff"]
        gp = pairs.groupby(["scenario", "forecaster", "method_a", "method_b"])
        psum = pd.concat([gp[pc].mean(), gp[pc].sem().add_suffix("_se"), gp.size().rename("n_reps")], axis=1).reset_index()
        psum.to_csv(PARTS / f"sim_paired_{key}.csv", index=False)
        print(f"[{key}] done ({time.time() - t1:.0f} s)", flush=True)
        show = summ[~summ["variant"].str.startswith("sweep")].sort_values("variant")
        for _, r in show.iterrows():
            print(f"   {r['variant']:18s} gamma={r['gamma']!s:6s} cov={r['coverage']:.4f} width={r['width_mean']:.3f} "
                  f"IS={r['score_mean']:.4f} ({r['score_mean_se']:.4f}) neff={r['neff_median']:.0f}/{r['pool_median']:.0f}", flush=True)
        for _, r in psum[psum["method_a"].isin(["dasc_clip_ext", "dasc_clip_wide"])].iterrows():
            print(f"   {r['method_a']} - {r['method_b']:18s} width {100 * r['rel_width_diff']:+.2f}% ({100 * r['rel_width_diff_se']:.2f}) "
                  f"cov {r['coverage_diff']:+.4f} IS {r['score_diff']:+.4f} ({r['score_diff_se']:.4f})", flush=True)
    print(f"[{scenario}] total {time.time() - t0:.0f} s", flush=True)


def main(argv):
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 50
    fcs = argv[argv.index("--fc") + 1].split(",") if "--fc" in argv else ["naive1", "ridge"]
    wide = "--no-wide" not in argv
    sweep = "--sweep" in argv
    force = "--force" in argv
    PARTS.mkdir(exist_ok=True)
    for sc in [a for a in argv if a in ds.SIM_SCENARIOS]:
        run_scenario(sc, fcs, n_seeds, wide, sweep, force)


if __name__ == "__main__":
    main(sys.argv[1:])
