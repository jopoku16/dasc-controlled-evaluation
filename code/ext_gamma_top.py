"""
ext_gamma_top.py: SUPPLEMENT to the extended-grid experiment (outside the pre-stated design; not used for the verdict).

The extended step-size set stops at 0.1 and the top value is still selected in several cells.  In every real cell
where ACI (projected), spectral ACI or DASC (projected) chose 0.1, this script adds the step sizes 0.15, 0.2 and
0.3 for those three methods, keeping everything else fixed (ACI: pool x the three new values; spectral ACI and
DASC: their selected configuration with the new step), selects on the same development score, re-runs on the
evaluation steps and repeats the paired comparison of DASC (projected) with ACI (projected) and spectral ACI.

Usage: python -X utf8 ext_gamma_top.py   (2 worker processes)
Writes results/ext_gamma_top_summary.csv, results/ext_gamma_top_paired.csv.
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
from dasc_lib import spectral_feature_matrix
from metrics import paired_block_bootstrap, summarize_run

R = E.RESULTS
GNEW = [0.15, 0.2, 0.3]
FCN = {"naive24": "naive", "naive1": "naive", "ridge": "ridge", "xgb": "xgb"}


def cfg_row(row):
    c = {k: row[k] for k in E.CFG_KEYS if k in row.index and not pd.isna(row[k])}
    for k in ("window", "fw"):
        if k in c:
            c[k] = int(c[k])
    return c


def main():
    cells = pd.read_csv(R / "ext_edge_cells.csv")
    top = cells[(cells["variant"].isin(["aci_clip_ext", "spectral_aci_ext", "dasc_clip_ext"])) & (cells["gamma_edge"] == "top")]
    keys = top[["dataset", "forecaster"]].drop_duplicates().values.tolist()
    print("cells with a top-edge choice:", keys, flush=True)
    summ_rows, pair_rows = [], []
    cache = {}
    for name, fname in keys:
        t0 = time.time()
        cfg = ds.REAL_DATASETS[name]
        if name not in cache:
            meta = cfg["load"]()
            y = meta["y"]
            preds = cfg["forecasts"](y, meta, with_xgb=True)
            feats = {fw: spectral_feature_matrix(y, fw) for fw in cfg["fws"]}
            cache[name] = (meta, y, preds, feats)
        meta, y, preds, feats = cache[name]
        n = len(y)
        start, dev_end = cfg["start"], int(cfg["dev_frac"] * n)
        pred = preds[fname]
        ch = pd.read_csv(R / "tuning" / f"ext_{name}_{fname}_chosen.csv").set_index("variant")
        base = {"aci_clip": ("aci_clip", cfg_row(ch.loc["aci_clip_ext"]), float(ch.loc["aci_clip_ext", "dev_score_grid"])),
                "spectral_aci": ("spectral_aci", cfg_row(ch.loc["spectral_aci_ext"]), float(ch.loc["spectral_aci_ext", "dev_score_grid"])),
                "dasc_clip": ("dasc_clip", cfg_row(ch.loc["dasc_clip_ext"]), float(ch.loc["dasc_clip_ext", "dev_score_grid"]))}
        streams = [{"y": y, "pred": pred, "feats": feats, "start": start, "dev_end": dev_end,
                    "clip_zero": cfg["clip_zero"], "wbl": cfg["worst_block_len"], "default_fw": cfg["default_fw"]}]
        jobs, cfgs = [], {}
        for w in cfg["pools"]:
            for g in GNEW:
                c = {"window": w, "gamma": g}
                cfgs[("aci_clip", len(cfgs))] = c
        for lab in ("spectral_aci", "dasc_clip"):
            for g in GNEW:
                c = {**base[lab][1], "gamma": g}
                cfgs[(lab, len(cfgs))] = c
        jobs = [(lab, lab, ci, c) for (lab, ci), c in cfgs.items()]
        with Pool(E.N_PROC, initializer=tuning._init, initargs=(streams,)) as pool:
            agg = E.evaluate(pool, jobs, 1)
            final = {}
            for lab in ("aci_clip", "spectral_aci", "dasc_clip"):
                a = agg[agg["tag"] == lab].sort_values(["dev_score", "config_id"])
                best_new = a.iloc[0]
                old_cfg, old_score = base[lab][1], base[lab][2]
                if best_new["dev_score"] < old_score:
                    final[lab] = (cfgs[(lab, int(best_new["config_id"]))], float(best_new["dev_score"]), True)
                else:
                    final[lab] = (old_cfg, old_score, False)
            res = pool.map(E.full_job, [(lab + "_top", lab, final[lab][0], 0) for lab in final], chunksize=1)
        outs = {lab: out for lab, out in res}
        for lab in final:
            out = outs[lab + "_top"]
            ev = out["t"] >= dev_end
            s = summarize_run(out, ev, cfg["worst_block_len"])
            summ_rows.append({"dataset": name, "forecaster": fname, "method": lab, "changed_by_new_steps": final[lab][2],
                              "gamma": final[lab][0].get("gamma"), "dev_score": final[lab][1], "config": str(final[lab][0]),
                              "coverage": s["coverage"], "width_mean": s["width_mean"], "score_mean": s["score_mean"]})
        for a, b in [("dasc_clip_top", "aci_clip_top"), ("dasc_clip_top", "spectral_aci_top"), ("spectral_aci_top", "aci_clip_top")]:
            ev = outs[a]["t"] >= dev_end
            r = paired_block_bootstrap(outs[a], outs[b], cfg["block_len"], n_boot=2000, seed=0, mask=ev)
            pair_rows.append({"dataset": name, "forecaster": fname, "method_a": a, "method_b": b,
                              "coverage_a": 1 - outs[a]["miss"][ev].mean(), "coverage_b": 1 - outs[b]["miss"][ev].mean(), **r})
        print(f"[{name}_{fname}] done ({time.time() - t0:.0f} s): " +
              ", ".join(f"{l}: gamma {final[l][0]['gamma']} ({'new' if final[l][2] else 'unchanged'})" for l in final), flush=True)
    pd.DataFrame(summ_rows).to_csv(R / "ext_gamma_top_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(R / "ext_gamma_top_paired.csv", index=False)


if __name__ == "__main__":
    main()
