"""
tuning.py: hyperparameter grids and the shared tune-then-freeze protocol.

Every method is tuned with the same criterion (mean interval score at the
target level on the development steps) on the same development data, then
frozen and evaluated on the remaining steps.  Pool size is part of every
method's grid, with the same candidate pools for all methods.

Development data
  real data   : steps with start <= t < dev_end (dev_end = 0.25 n) of the
                single stream; the configuration then continues over the
                evaluation steps t >= dev_end with its state carried over
  simulations : the full streams (t >= start) of separate development seeds;
                evaluation seeds are disjoint

Each configuration is run once; the development score and the evaluation
summary come from the same run.  Selection uses the development score only.
The grid file records both so that a reader can check this.
"""

from __future__ import annotations

import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from dasc_lib import METHOD_NAMES, make_method, run_stream
from metrics import summarize_run

GAMMAS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
HS = [0.35, 0.55, 0.80]
DECAYS = [0.97, 0.985, 0.99, 0.995, 0.998, 0.999]


def build_grid(name, pools, fws):
    pools = list(pools)
    fws = list(fws)
    if name == "rolling":
        dims = {"window": pools}
    elif name in ("aci_clip", "aci_noclip"):
        dims = {"window": pools, "gamma": GAMMAS}
    elif name == "pid":
        dims = {"window": pools, "kp": [0.01, 0.025, 0.05], "ki": [0.001, 0.003], "kd": [0.0, 0.01]}
    elif name == "decay":
        dims = {"window": pools, "decay": DECAYS}
    elif name == "spectral_fixed":
        dims = {"window": pools, "h": HS, "fw": fws}
    elif name == "spectral_aci":
        dims = {"window": pools, "h": HS, "gamma": [0.01, 0.015, 0.03], "fw": fws}
    elif name in ("dasc_clip", "dasc_noclip", "dasc_fixed"):
        dims = {"window": pools, "h": HS, "drift_lambda": [0.15, 0.45],
                "m_min_frac": [0.25, 0.5], "fw": fws}
    else:
        raise ValueError(name)
    keys = list(dims)
    return [dict(zip(keys, vals)) for vals in itertools.product(*dims.values())]


def params_from_config(cfg):
    """fw selects the feature matrix (handled by the runner); the drift score's
    recent window is always l_r = 64 (dasc_lib.RECENT_LEN)."""
    p = dict(cfg)
    if "m_min_frac" in p:
        p["m_min"] = int(round(p["m_min_frac"] * p["window"]))
    return p


# ----------------------------------------------------------------------------
# worker state (set once per process by the Pool initializer)
# ----------------------------------------------------------------------------

_STREAMS = None     # list of dicts: y, pred, feats (fw -> matrix), start, dev_mask_end, clip_zero, wbl


def _init(streams):
    global _STREAMS
    _STREAMS = streams


def _run_one(y, pred, feats, name, cfg, start, clip_zero, default_fw):
    p = params_from_config(cfg)
    m = make_method(name, alpha=0.10, first_valid=1, **p)
    f = feats[int(cfg.get("fw", default_fw))] if m.needs_features else None
    return run_stream(y, pred, m, features=f, start=start, clip_zero=clip_zero)


def _job(args):
    name, ci, cfg, si = args
    s = _STREAMS[si]
    out = _run_one(s["y"], s["pred"], s["feats"], name, cfg, s["start"], s["clip_zero"], s["default_fw"])
    t = out["t"]
    dev = t < s["dev_end"]
    ev = ~dev
    dev_score = float(out["score"][dev].mean()) if dev.any() else np.nan
    res = {"method": name, "config_id": ci, "stream": si, "dev_score": dev_score,
           "dev_coverage": float(1 - out["miss"][dev].mean()) if dev.any() else np.nan,
           "dev_width": float(out["width"][dev].mean()) if dev.any() else np.nan,
           "eval_coverage": np.nan, "eval_width": np.nan, "eval_score": np.nan}
    if ev.any():
        summ = summarize_run(out, ev, s["wbl"])
        res.update({"eval_coverage": summ["coverage"], "eval_width": summ["width_mean"],
                    "eval_score": summ["score_mean"]})
    return res


def tune(streams, pools, fws, methods=METHOD_NAMES, n_proc=8, default_fw=64):
    """Run every grid configuration on every development stream.

    streams: list of dicts with keys y, pred, feats (dict fw -> features),
             start, dev_end (steps t < dev_end are development; use n for
             "whole stream"), clip_zero, wbl (worst-block length)
    Returns (grid DataFrame, chosen dict method -> config, counts dict).
    """
    for s in streams:
        s["default_fw"] = default_fw
    jobs = []
    grids = {}
    for name in methods:
        grids[name] = build_grid(name, pools, fws)
        for ci, cfg in enumerate(grids[name]):
            for si in range(len(streams)):
                jobs.append((name, ci, cfg, si))
    with Pool(n_proc, initializer=_init, initargs=(streams,)) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "config_id"]).agg(
        dev_score=("dev_score", "mean"), dev_coverage=("dev_coverage", "mean"),
        dev_width=("dev_width", "mean"), eval_coverage=("eval_coverage", "mean"),
        eval_width=("eval_width", "mean"), eval_score=("eval_score", "mean"),
        n_streams=("stream", "size")).reset_index()
    agg["config"] = [str(grids[m][c]) for m, c in zip(agg["method"], agg["config_id"])]
    for key in ["window", "gamma", "kp", "ki", "kd", "decay", "h", "fw", "drift_lambda", "m_min_frac"]:
        agg[key] = [grids[m][c].get(key, np.nan) for m, c in zip(agg["method"], agg["config_id"])]
    chosen, counts = {}, {}
    for name in methods:
        sub = agg[agg["method"] == name].sort_values(["dev_score", "config_id"])
        best = int(sub.iloc[0]["config_id"])
        chosen[name] = grids[name][best]
        counts[name] = len(grids[name])
    agg["selected"] = [chosen[m] == grids[m][c] for m, c in zip(agg["method"], agg["config_id"])]
    return agg, chosen, counts


def run_chosen(y, pred, feats, name, cfg, start, clip_zero, default_fw=64):
    """Full per-step run of a chosen configuration (main process)."""
    return _run_one(y, pred, feats, name, cfg, start, clip_zero, default_fw)
