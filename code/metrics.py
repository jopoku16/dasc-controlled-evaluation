"""
metrics.py: evaluation metrics for streaming interval predictors.

All functions take per-step arrays aligned on time.  Coverage is always
reported next to width; width is never reported alone.  Infinite widths
(free update with alpha_t <= 0) are counted separately and excluded from the
width and score means, which are therefore means over finite steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.10
CAP_HI = 0.35
CAP_LO = 0.01


def _fmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _fmedian(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def summarize_run(out, mask=None, worst_block_len=100, alpha=ALPHA):
    """Scalar summary of one run (dict of arrays from dasc_lib.run_stream).

    mask selects the evaluated steps (e.g. t >= dev_end).  The adaptive-update
    identity is evaluated on the selected window: with alpha_1 the level at its
    first step and alpha_{T+1} the state after its last update,
        mean(E_t) - alpha = (alpha_1 - alpha_{T+1}) / (gamma T) + sum_t r_t / (gamma T),
    where r_t is the projection residual (0 for the free update).
    """
    T_all = len(out["t"])
    if mask is None:
        mask = np.ones(T_all, dtype=bool)
    idx = np.where(mask)[0]
    miss = out["miss"][mask].astype(float)
    width = out["width"][mask]
    score = out["score"][mask]
    a = out["alpha_t"][mask]
    T = int(mask.sum())
    res = {
        "n": T,
        "coverage": 1.0 - miss.mean() if T else np.nan,
        "width_mean": _fmean(width),
        "width_median": _fmedian(width),
        "score_mean": _fmean(score),
        "n_inf_width": int(np.sum(~np.isfinite(width))),
        "n_empty": int(np.nansum(out["empty"][mask])) if "empty" in out else 0,
        "neff_median": _fmedian(out["neff"][mask]),
        "drift_mean": _fmean(out["drift"][mask]),
        "dbar_mean": _fmean(out["dbar"][mask]),
        "dbar_pre_mean": _fmean(out["dbar_pre"][mask]) if "dbar_pre" in out else np.nan,
        "recent_gap_mean": _fmean(out["recent_gap"][mask]) if "recent_gap" in out else np.nan,
        "pool_mean": _fmean(out["pool_size"][mask]),
        "pi_mean": _fmean(out["pi"][mask]) if "pi" in out else np.nan,
        "wmax_mean": _fmean(out["wmax"][mask]) if "wmax" in out else np.nan,
        "alpha_mean": float(np.mean(a)) if T else np.nan,
        "alpha_min": float(np.min(a)) if T else np.nan,
        "alpha_max": float(np.max(a)) if T else np.nan,
        "frac_alpha_at_hi_cap": float(np.mean(a >= CAP_HI - 1e-12)) if T else np.nan,
        "frac_alpha_at_lo_cap": float(np.mean(a <= CAP_LO + 1e-12)) if T else np.nan,
        "frac_alpha_le_0": float(np.mean(a <= 0.0)) if T else np.nan,
        "frac_alpha_ge_1": float(np.mean(a >= 1.0)) if T else np.nan,
    }
    # computable part of the per-step lower bound, 1 - alpha_t - eps_t(eta)
    for eta_key in ("eps_005", "eps_010"):
        if eta_key in out:
            lb = 1.0 - a - out[eta_key][mask]
            lb = lb[np.isfinite(lb)]
            tag = eta_key.replace("eps_", "lb_")
            res[tag + "_mean"] = float(lb.mean()) if len(lb) else np.nan
            res[tag + "_q10"] = float(np.quantile(lb, 0.10)) if len(lb) else np.nan
            res[tag + "_q50"] = float(np.quantile(lb, 0.50)) if len(lb) else np.nan
            res[tag + "_q90"] = float(np.quantile(lb, 0.90)) if len(lb) else np.nan
    # adaptive-update identity on the window
    gamma = float(out.get("gamma", np.nan))
    if T and np.isfinite(gamma) and "proj_resid" in out:
        r = out["proj_resid"][mask]
        u = out["U"][mask]
        alpha_1 = float(a[0])
        alpha_T1 = float(u[-1] + r[-1])            # state after the last update
        res.update({
            "gamma": gamma,
            "alpha_1": alpha_1,
            "alpha_T1": alpha_T1,
            "frac_bind_cap": float(np.mean(r < 0)),      # N+_T / T
            "frac_bind_floor": float(np.mean(r > 0)),    # N-_T / T
            "telescoping_term": (alpha_1 - alpha_T1) / (gamma * T),
            "projection_term": float(np.sum(r)) / (gamma * T),
            "free_bound": (max(alpha_1, 1.0 - alpha_1) + gamma) / (gamma * T),
        })
        res["identity_gap"] = (miss.mean() - alpha) - res["telescoping_term"] - res["projection_term"]
    res.update(worst_block(miss, worst_block_len))
    return res


def worst_block(miss, block_len):
    """Minimum coverage over consecutive non-overlapping blocks and the share below 0.85."""
    miss = np.asarray(miss, dtype=float)
    nb = len(miss) // block_len
    if nb == 0:
        return {"worst_block_coverage": np.nan, "frac_blocks_below_085": np.nan, "n_blocks": 0}
    blocks = miss[: nb * block_len].reshape(nb, block_len)
    cov = 1.0 - blocks.mean(axis=1)
    return {"worst_block_coverage": float(cov.min()),
            "frac_blocks_below_085": float(np.mean(cov < 0.85)),
            "n_blocks": int(nb)}


def local_coverage(out, groups, mask=None):
    """Coverage and finite-width mean by group; groups is a dict name -> array indexed by t."""
    t = out["t"]
    if mask is None:
        mask = np.ones(len(t), dtype=bool)
    rows = []
    for gname, garr in groups.items():
        g = np.asarray(garr)[t[mask]]
        w = out["width"][mask].astype(float)
        w = np.where(np.isfinite(w), w, np.nan)
        df = pd.DataFrame({"g": g, "miss": out["miss"][mask], "width": w})
        agg = df.groupby("g").agg(coverage=("miss", lambda x: 1 - x.mean()),
                                  width_mean=("width", "mean"), n=("miss", "size")).reset_index()
        for _, r in agg.iterrows():
            rows.append({"group_var": gname, "group": str(r["g"]), "coverage": r["coverage"],
                         "width_mean": r["width_mean"], "n": int(r["n"])})
    return pd.DataFrame(rows)


def _block_indices(n, block_len, rng):
    k = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=k)
    idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()
    return idx[:n]


def paired_block_bootstrap(out_a, out_b, block_len, n_boot=2000, seed=0, mask=None):
    """Paired moving-block bootstrap of mean differences (A minus B).

    Both runs must share the same time index (same forecaster, so the same
    skipped steps).  The same blocks are drawn for both methods, so the
    interval is for the paired difference.  Steps where either method has an
    infinite width or score are excluded (their number is reported as
    n_masked).  Returns point estimates and percentile 95% intervals for the
    differences in mean width, coverage and mean interval score, plus the
    relative width difference.
    """
    ta, tb = out_a["t"], out_b["t"]
    assert np.array_equal(ta, tb), "runs are not aligned on t"
    if mask is None:
        mask = np.ones(len(ta), dtype=bool)
    finite = (np.isfinite(out_a["width"]) & np.isfinite(out_b["width"])
              & np.isfinite(out_a["score"]) & np.isfinite(out_b["score"]))
    n_masked = int(np.sum(mask & ~finite))
    mask = mask & finite
    dw = out_a["width"][mask] - out_b["width"][mask]
    dc = out_b["miss"][mask].astype(float) - out_a["miss"][mask].astype(float)   # coverage A - B
    dsc = out_a["score"][mask] - out_b["score"][mask]
    wb = out_b["width"][mask]
    n = len(dw)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, 4))
    for b in range(n_boot):
        idx = _block_indices(n, block_len, rng)
        boots[b, 0] = dw[idx].mean()
        boots[b, 1] = dc[idx].mean()
        boots[b, 2] = dsc[idx].mean()
        boots[b, 3] = dw[idx].mean() / wb[idx].mean()
    est = np.array([dw.mean(), dc.mean(), dsc.mean(), dw.mean() / wb.mean()])
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    names = ["width_diff", "coverage_diff", "score_diff", "rel_width_diff"]
    res = {"n": n, "n_masked": n_masked, "block_len": block_len, "n_boot": n_boot}
    for i, nm in enumerate(names):
        res[nm] = est[i]
        res[nm + "_lo"] = lo[i]
        res[nm + "_hi"] = hi[i]
    return res
