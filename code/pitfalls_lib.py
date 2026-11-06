"""
pitfalls_lib.py: building blocks for the pitfall decomposition (pitfalls_run.py).

Nothing in dasc_lib.py is edited.  The reference method is ACI with the
projected update, exactly as in the paper (dasc_lib.ACI with clip=True).  Each
pitfall changes ONE thing about it:

  rank     the quantile step.  "plain" calls dasc_lib.weighted_quantile with
           uniform weights, which returns the order statistic of rank
           ceil(n (1 - alpha_t)) (smallest v with cdf >= 1 - alpha_t, cdf_j =
           j / n); this is the convention of every method in the paper.
           "finite" uses the finite-sample rank ceil((n + 1)(1 - alpha_t)) of
           split conformal prediction.  If that rank exceeds n the quantile is
           +infinity (never happens for alpha_t >= 0.01 and a pool of at least
           99 residuals; counted anyway).
  freeze_at the pool stops receiving residuals: at every t > freeze_at the pool
           is the pool as it stood at t = freeze_at.  The level update of ACI
           keeps running (freezing the pool is the one change).
  window   the maximum pool size.
  gamma    the ACI step size.  gamma = 0 gives a fixed level alpha_t = alpha,
           which is the rolling conformal method of the paper step for step.

No method here reads a residual at an index >= t (frozen pools read indices
below freeze_at <= t), so causality is the same as for the paper's methods.
"""

from __future__ import annotations

import numpy as np

from dasc_lib import ACI, run_stream, weighted_quantile
from metrics import _block_indices

ALPHA = 0.10
RANKS = ("plain", "finite")


def finite_rank(n, alpha_level):
    """ceil((n + 1)(1 - alpha_level)); the tolerance only guards float noise."""
    return int(np.ceil((n + 1) * (1.0 - alpha_level) - 1e-9))


def plain_rank(n, alpha_level):
    """Rank returned by dasc_lib.weighted_quantile with uniform weights (reference rule)."""
    return int(np.ceil(n * (1.0 - alpha_level) - 1e-9))


def quantile_step(r, alpha_level, rank):
    """Quantile of the pool residuals r (uniform weights) under the chosen rank convention."""
    n = len(r)
    if rank == "plain":
        return weighted_quantile(r, np.ones(n), 1.0 - alpha_level)
    if rank == "finite":
        k = max(finite_rank(n, alpha_level), 1)
        if k > n:
            return float("inf")
        return float(np.partition(np.asarray(r, dtype=float), k - 1)[k - 1])
    raise ValueError(rank)


class PitfallACI(ACI):
    """ACI (projected) with a switchable quantile rank and an optional frozen pool."""

    name = "aci_pitfall"

    def __init__(self, alpha=ALPHA, window=360, first_valid=1, gamma=0.015, rank="plain",
                 freeze_at=None, alpha_min=0.01, alpha_max=0.35):
        if rank not in RANKS:
            raise ValueError(rank)
        self.rank = rank
        self.freeze_at = None if freeze_at is None else int(freeze_at)
        super().__init__(alpha, window, first_valid, gamma=gamma, clip=True,
                         alpha_min=alpha_min, alpha_max=alpha_max)

    def pool_indices(self, t, residuals):
        if self.freeze_at is not None and t > self.freeze_at:
            t = self.freeze_at
        return super().pool_indices(t, residuals)

    def interval(self, t, pred_t, residuals, features=None):
        pool = self.pool_indices(t, residuals)
        r = residuals[pool]
        n = len(r)
        q = quantile_step(r, self.alpha_t, self.rank)
        w = np.ones(n)
        pi = float(np.sum(w[r == q]) / n) if np.isfinite(q) else 0.0
        return pred_t - q, pred_t + q, self._diag(
            alpha_t=self.alpha_t, alpha_eff=self.alpha_t, q=q, neff=float(n), pool_size=n,
            gate_m=self.window, pi=pi, wmax=1.0 / n)


def run_spec(y, pred, spec, start, clip_zero):
    """Run one configuration spec (dict with window, gamma, rank, freeze_at)."""
    m = PitfallACI(ALPHA, int(spec["window"]), 1, gamma=float(spec["gamma"]),
                   rank=spec.get("rank", "plain"), freeze_at=spec.get("freeze_at"))
    return run_stream(y, pred, m, features=None, start=start, clip_zero=clip_zero)


# ----------------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------------


def _fmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def summarize(out, mask):
    """Coverage, mean finite width, mean finite interval score on the masked steps."""
    miss = out["miss"][mask].astype(float)
    return {"n": int(mask.sum()), "coverage": float(1.0 - miss.mean()),
            "width_mean": _fmean(out["width"][mask]), "score_mean": _fmean(out["score"][mask]),
            "n_inf_width": int(np.sum(~np.isfinite(out["width"][mask]))),
            "frac_alpha_hi_cap": float(np.mean(out["alpha_t"][mask] >= 0.35 - 1e-12)),
            "frac_alpha_lo_cap": float(np.mean(out["alpha_t"][mask] <= 0.01 + 1e-12))}


def per_run_deltas(out_a, out_b, mask):
    """Scalar paired contrasts A (pitfall) minus B (reference) on the masked steps."""
    assert np.array_equal(out_a["t"], out_b["t"])
    sa, sb = summarize(out_a, mask), summarize(out_b, mask)
    return {"cov_a": sa["coverage"], "cov_b": sb["coverage"],
            "width_a": sa["width_mean"], "width_b": sb["width_mean"],
            "score_a": sa["score_mean"], "score_b": sb["score_mean"],
            "rel_width": (sa["width_mean"] - sb["width_mean"]) / sb["width_mean"],
            "cov_change": sa["coverage"] - sb["coverage"],
            "score_change": sa["score_mean"] - sb["score_mean"],
            "rel_score": (sa["score_mean"] - sb["score_mean"]) / sb["score_mean"],
            "n_inf_a": sa["n_inf_width"], "n_inf_b": sb["n_inf_width"]}


def paired_boot(out_a, out_b, block_len, mask, n_boot=2000, seed=0):
    """Paired moving-block bootstrap of A minus B (A = pitfall, B = reference).

    Same resampling scheme and random stream as metrics.paired_block_bootstrap
    (same blocks for both runs, percentile 95% intervals), and in addition the
    interval-score change relative to the reference.  Steps with a non-finite
    width or score in either run are excluded (n_masked).
    """
    assert np.array_equal(out_a["t"], out_b["t"])
    finite = (np.isfinite(out_a["width"]) & np.isfinite(out_b["width"])
              & np.isfinite(out_a["score"]) & np.isfinite(out_b["score"]))
    n_masked = int(np.sum(mask & ~finite))
    mk = mask & finite
    wa, wb = out_a["width"][mk], out_b["width"][mk]
    sa, sb = out_a["score"][mk], out_b["score"][mk]
    ca, cb = 1.0 - out_a["miss"][mk].astype(float), 1.0 - out_b["miss"][mk].astype(float)
    dw, ds_, dc = wa - wb, sa - sb, ca - cb
    n = len(dw)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, 4))
    for b in range(n_boot):
        idx = _block_indices(n, block_len, rng)
        boots[b, 0] = dw[idx].mean() / wb[idx].mean()
        boots[b, 1] = dc[idx].mean()
        boots[b, 2] = ds_[idx].mean() / sb[idx].mean()
        boots[b, 3] = ds_[idx].mean()
    est = np.array([dw.mean() / wb.mean(), dc.mean(), ds_.mean() / sb.mean(), ds_.mean()])
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    res = {"n_boot": n_boot, "block_len": block_len, "n_used": n, "n_masked": n_masked}
    for i, nm in enumerate(["rel_width", "cov_change", "rel_score", "score_change"]):
        res[nm] = float(est[i])
        res[nm + "_lo"] = float(lo[i])
        res[nm + "_hi"] = float(hi[i])
    return res
