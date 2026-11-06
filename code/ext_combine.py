"""
ext_combine.py: extended-grid experiment, combine the per-cell parts into the ext_*.csv files, apply the pre-stated
criterion.  Reads only ext_* files and (for the original-grid edge counts) the
existing tuning CSVs.  Writes only new files.

Outputs (results/): ext_real_summary.csv, ext_real_paired_bootstrap.csv, ext_tuning_chosen.csv,
  ext_bandwidth_scale.csv, ext_neff.csv, ext_edge_selection.csv, ext_edge_cells.csv, ext_verdict_cells.csv,
  ext_dev_spotcheck.csv, ext_reproduction_check.csv, ext_sim_summary.csv, ext_sim_paired.csv, ext_sim_sweep.csv,
  ext_sim_repro.csv
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
PARTS = R / "ext_parts"

DS_ORDER = ["electricity", "weather", "finance"]
FC_ORDER = ["naive24", "naive1", "ridge", "xgb"]
DS_LABEL = {"electricity": "Electricity", "weather": "Weather", "finance": "Finance"}
FC_LABEL = {"naive24": "Seasonal naive", "naive1": "Naive (lag 1)", "ridge": "Rolling ridge", "xgb": "Gradient boosting"}


def _cat(pattern):
    fs = sorted(glob.glob(str(PARTS / pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _sort(df):
    if df.empty:
        return df
    df = df.copy()
    df["_d"] = df["dataset"].map({d: i for i, d in enumerate(DS_ORDER)})
    df["_f"] = df["forecaster"].map({f: i for i, f in enumerate(FC_ORDER)})
    return df.sort_values(["_d", "_f"], kind="mergesort").drop(columns=["_d", "_f"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# real data
# ----------------------------------------------------------------------------


def combine_real():
    summ = _sort(_cat("summary_*.csv"))
    pair = _sort(_cat("paired_*.csv"))
    spot = _sort(_cat("spotcheck_*.csv"))
    if summ.empty:
        print("no real parts yet")
        return None
    summ.to_csv(R / "ext_real_summary.csv", index=False)
    pair.to_csv(R / "ext_real_paired_bootstrap.csv", index=False)
    spot.to_csv(R / "ext_dev_spotcheck.csv", index=False)
    rep = summ[summ["matches_reference"].notna()][["dataset", "forecaster", "variant", "matches_reference"]]
    rep.to_csv(R / "ext_reproduction_check.csv", index=False)
    chosen = _sort(pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "tuning" / "ext_*_chosen.csv")))
                              if "ext_sim_" not in f], ignore_index=True))
    bw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "ext_bandwidth_scale_*.csv")))], ignore_index=True)
    bw.to_csv(R / "ext_bandwidth_scale.csv", index=False)
    # edge flags and scaled-h indicators on the chosen configurations
    hcols = [c for c in bw.columns if c.startswith("h_") and c[2:].isdigit()]
    def hinfo(r):
        if pd.isna(r.get("h")) or pd.isna(r.get("fw")):
            return pd.Series({"h_scaled": np.nan, "h_edge": "", "h_over_median": np.nan, "h_rank": np.nan})
        b = bw[(bw["dataset"] == r["dataset"]) & (bw["fw"] == int(r["fw"]))].iloc[0]
        gv = sorted(float(b[c]) for c in hcols)
        h = float(r["h"])
        return pd.Series({"h_scaled": h not in (0.35, 0.55, 0.8),
                          "h_edge": "bottom" if np.isclose(h, gv[0]) else ("top" if np.isclose(h, gv[-1]) else ""),
                          "h_over_median": h / float(b["median_pairwise"]),
                          "h_rank": int(np.argmin(np.abs(np.array(gv) - h))) + 1})
    chosen = pd.concat([chosen, chosen.apply(hinfo, axis=1)], axis=1)
    def gedge(r):
        g = r.get("gamma")
        if pd.isna(g):
            return ""
        return "bottom" if np.isclose(g, 0.005) else ("top" if np.isclose(g, 0.1) else "")
    chosen["gamma_edge"] = chosen.apply(gedge, axis=1)
    # PID: the step-like parameter kp
    chosen["kp_edge"] = chosen["kp"].map(lambda g: "" if pd.isna(g) else ("bottom" if np.isclose(g, 0.005) else ("top" if np.isclose(g, 0.1) else "")))
    chosen.to_csv(R / "ext_tuning_chosen.csv", index=False)

    # ---- n_eff against the pool size
    sp = summ[summ["method"].isin(["spectral_fixed", "spectral_aci", "dasc_fixed", "dasc_clip", "dasc_noclip"])].copy()
    sp = sp.merge(bw[["dataset", "fw", "median_pairwise"]], on=["dataset", "fw"], how="left")
    sp["h_over_median"] = sp["h"] / sp["median_pairwise"]
    cols = ["dataset", "forecaster", "variant", "method", "window", "fw", "h", "median_pairwise", "h_over_median", "gamma",
            "neff_median", "pool_median", "neff_over_pool_median", "frac_neff_lt30", "neff_min", "coverage", "width_mean", "score_mean"]
    sp[cols].to_csv(R / "ext_neff.csv", index=False)

    # ---- edge selection
    edge_cells = chosen[["dataset", "forecaster", "variant", "method", "gamma", "kp", "gamma_edge", "kp_edge", "h", "h_scaled",
                         "h_edge", "h_over_median"]].copy()
    # original-grid choices on the same cells
    orig = []
    for _, c in chosen[["dataset", "forecaster"]].drop_duplicates().iterrows():
        mc = pd.read_csv(R / "tuning" / f"{c.dataset}_{c.forecaster}_chosen.csv").set_index("method")
        pc = pd.read_csv(R / "tuning" / f"{c.dataset}_{c.forecaster}_dascgamma_chosen.csv").set_index("variant")
        for var, g, lo, hi in [("aci_clip_ext", mc.loc["aci_clip", "gamma"], 0.005, 0.05),
                               ("aci_noclip_ext", mc.loc["aci_noclip", "gamma"], 0.005, 0.05),
                               ("spectral_aci_ext", mc.loc["spectral_aci", "gamma"], 0.01, 0.03),
                               ("pid_ext", mc.loc["pid", "kp"], 0.01, 0.05),
                               ("dasc_clip_ext", pc.loc["dasc_clip", "gamma"], 0.005, 0.05),
                               ("dasc_noclip_ext", pc.loc["dasc_noclip", "gamma"], 0.005, 0.05)]:
            orig.append({"dataset": c.dataset, "forecaster": c.forecaster, "variant": var, "orig_value": g,
                         "orig_edge": "bottom" if np.isclose(g, lo) else ("top" if np.isclose(g, hi) else "")})
    orig = pd.DataFrame(orig)
    edge_cells = edge_cells.merge(orig, on=["dataset", "forecaster", "variant"], how="left")
    edge_cells.to_csv(R / "ext_edge_cells.csv", index=False)
    rows = []
    for var, lab, stepcol in [("aci_clip_ext", "ACI (projected)", "gamma"), ("aci_noclip_ext", "ACI (free)", "gamma"),
                              ("pid_ext", "PID-style ($k_p$)", "kp"), ("spectral_aci_ext", "Spectral ACI", "gamma"),
                              ("dasc_clip_ext", "DASC (projected)", "gamma"), ("dasc_noclip_ext", "DASC (free)", "gamma"),
                              ("dasc_clip_wide", "DASC (projected), stage 0", "gamma"),
                              ("dasc_noclip_wide", "DASC (free), stage 0", "gamma")]:
        e = edge_cells[edge_cells["variant"] == var]
        if e.empty:
            continue
        ecol = "kp_edge" if stepcol == "kp" else "gamma_edge"
        has_h = e["h"].notna().any()
        rows.append({"variant": var, "label": lab, "n_cells": len(e),
                     "orig_bottom": int((e["orig_edge"] == "bottom").sum()) if e["orig_edge"].notna().any() else np.nan,
                     "orig_top": int((e["orig_edge"] == "top").sum()) if e["orig_edge"].notna().any() else np.nan,
                     "ext_bottom": int((e[ecol] == "bottom").sum()), "ext_top": int((e[ecol] == "top").sum()),
                     "h_scaled_chosen": int(sum(bool(x) for x in e["h_scaled"] if pd.notna(x))) if has_h else np.nan,
                     "h_edge_bottom": int((e["h_edge"] == "bottom").sum()) if has_h else np.nan,
                     "h_edge_top": int((e["h_edge"] == "top").sum()) if has_h else np.nan})
    pd.DataFrame(rows).to_csv(R / "ext_edge_selection.csv", index=False)

    # ---- pre-stated criterion
    v = []
    for design, a in [("primary", "dasc_clip_ext"), ("wide", "dasc_clip_wide")]:
        for comp in ["aci_clip_ext", "spectral_aci_ext", "decay"]:
            sub = pair[(pair["method_a"] == a) & (pair["method_b"] == comp)]
            for _, r in sub.iterrows():
                v.append({"dataset": r["dataset"], "forecaster": r["forecaster"], "design": design, "comparator": comp,
                          "rel_width_pct": 100 * r["rel_width_diff"], "rel_width_lo": 100 * r["rel_width_diff_lo"],
                          "rel_width_hi": 100 * r["rel_width_diff_hi"], "coverage_diff_pts": 100 * r["coverage_diff"],
                          "score_diff": r["score_diff"], "score_lo": r["score_diff_lo"], "score_hi": r["score_diff_hi"],
                          "width_advantage_excl_zero": bool(r["rel_width_diff_hi"] < 0),
                          "score_advantage_excl_zero": bool(r["score_diff_hi"] < 0),
                          "width_disadvantage_excl_zero": bool(r["rel_width_diff_lo"] > 0),
                          "score_disadvantage_excl_zero": bool(r["score_diff_lo"] > 0)})
    vd = _sort(pd.DataFrame(v))
    vd.to_csv(R / "ext_verdict_cells.csv", index=False)
    print("\n== pre-stated criterion (DASC projected against ACI projected, extended grids) ==")
    for design in ("primary", "wide"):
        s = vd[(vd["design"] == design) & (vd["comparator"] == "aci_clip_ext")]
        bad = s[s["width_advantage_excl_zero"] | s["score_advantage_excl_zero"]]
        print(f"{design}: cells {len(s)}; violating cells {len(bad)}")
        for _, r in bad.iterrows():
            print(f"   {r.dataset}/{r.forecaster}: width {r.rel_width_pct:+.2f}% [{r.rel_width_lo:+.2f}, {r.rel_width_hi:+.2f}] "
                  f"cov {r.coverage_diff_pts:+.2f}; IS {r.score_diff:+.4f} [{r.score_lo:+.4f}, {r.score_hi:+.4f}]")
    return summ, pair, chosen, edge_cells, vd


# ----------------------------------------------------------------------------
# simulations
# ----------------------------------------------------------------------------


def combine_sim():
    out = {}
    for tag, pat in [("sim_summary", "sim_summary_*.csv"), ("sim_paired", "sim_paired_*.csv"), ("sim_repro", "sim_repro_*.csv")]:
        df = _cat(pat)
        if not df.empty:
            df.to_csv(R / f"ext_{tag}.csv", index=False)
            out[tag] = df
    ps = [pd.read_csv(f) for f in sorted(glob.glob(str(PARTS / "sim_per_seed_*.csv")))]
    if ps:
        allp = pd.concat(ps, ignore_index=True)
        sw = allp[allp["variant"].str.startswith("sweep_")].copy()
        if not sw.empty:
            sw["step"] = sw["variant"].str.split("|").str[1].astype(float)
            sw["family"] = sw["variant"].str.split("|").str[0]
            g = sw.groupby(["scenario", "forecaster", "family", "step"])
            agg = pd.concat([g[["coverage", "width_mean", "score_mean"]].mean(),
                             g[["coverage", "width_mean", "score_mean"]].sem().add_suffix("_se"),
                             g.size().rename("n_reps")], axis=1).reset_index()
            agg.to_csv(R / "ext_sim_sweep.csv", index=False)
            out["sweep"] = agg
    return out


if __name__ == "__main__":
    res = combine_real()
    s = combine_sim()
    print("sim parts combined:", list(s))
