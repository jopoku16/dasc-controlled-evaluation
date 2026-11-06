"""
dasc_lib.py: one framework for streaming conformal prediction methods.

Every method implements the same interface:

    lo, hi, diag = method.interval(t, pred_t, residuals, features)
    method.update(miss)          # adaptive methods only; others ignore it

At time t a method may use ONLY information from times < t:
  * residuals[i] = |y[i] - pred[i]| for i < t (residuals[i] for i >= t is never read),
  * features[i] for i <= t, where features[t] is computed from y[t-l], ..., y[t-1]
    (window ends at t-1, exclusive of t), so features[t] is measurable with
    respect to the history up to t-1,
  * its own state (alpha_t, PID integrator), which depends on misses at times < t.

The point forecast pred_t is supplied by the caller and is identical for every
method.  No method computes a forecast internally.  test_causality.py checks
that perturbing y[t:] (including y[t]) leaves the interval at t unchanged.

Definitions follow the original project code (dasc.py, run_first_simulation.py
and the run_real_*.py drivers) and Sections 4-6 of the original manuscript:
banded periodogram features over K = 6 bands, l2-normalised; Gaussian kernel
exp(-d^2 / h^2); drift score D_t = || mean of the last l_r = 64 features -
weighted pool mean ||_2; linear gate m_t = m_max - (m_max - m_min) min(1, D_t /
lambda) with a floor of 20 residuals; weighted quantile inf{q : sum_i w_i
1{R_i <= q} >= 1 - alpha_t}; adaptive update alpha_{t+1} = alpha_t + gamma
(alpha - miss), projected onto [alpha_min, alpha_max] in the clipped variants.

Free (unprojected) update, conventions of the revised theory: when
alpha_t <= 0 the interval is the whole real line (miss = 0, width infinite);
when alpha_t >= 1 the interval is empty (miss = 1, width 0, scored as the
degenerate interval {pred_t}); otherwise the weighted quantile at level
1 - alpha_t.

Per-step diagnostics returned by every method (NaN where not defined):
  alpha_t   level used at t (state before the update)
  alpha_eff level passed to the quantile (equal to alpha_t; no relaxation)
  q         quantile; neff = 1 / sum w^2; pool_size (gated pool); gate_m (m_t)
  dbar      sum_i w_i ||S_i - S_t|| under the final (gated) weights
  dbar_pre  the same under the pre-gate weights
  drift     D_t;  recent_gap = || mean of the last l_r features - S_t ||
  pi        total final weight on pool residuals equal to q
  wmax      largest final weight
  eps_005, eps_010   grid term sqrt(log(4K/eta)/(2 neff)) + 1/K, K = max(2, ceil(neff))
  empty     1 when the interval is the empty set (free update, alpha_t >= 1)
The stream runner adds, from the update at t:
  U         pre-projection value alpha_t + gamma (alpha - miss)
  proj_resid  alpha_{t+1} - U (positive at the floor, negative at the cap, 0 if free)
"""

from __future__ import annotations

import numpy as np

RECENT_LEN = 64      # l_r: length of the recent-feature average in the drift score

# ----------------------------------------------------------------------------
# Quantile and features
# ----------------------------------------------------------------------------


def weighted_quantile(values, weights, level):
    """Smallest v with sum_{i: v_i <= v} w_i / sum_i w_i >= level.

    This is the weighted conformal quantile of the original code.  The level is
    truncated to [0, 1]: level <= 0 returns the smallest value, level >= 1 the
    largest.  No point mass is placed on the test point (same convention for all
    methods; see README).
    """
    level = min(max(float(level), 0.0), 1.0)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    k = int(np.searchsorted(cdf, level, side="left"))
    return float(v[min(k, len(v) - 1)])


def spectral_feature(y, end, window=64, n_bands=6):
    """Banded periodogram of y[end-window:end] (window ends at end-1, exclusive).

    Steps: subtract the window mean, squared modulus of the real FFT, drop the
    zero frequency, split the positive frequencies into n_bands equal groups,
    average within bands, l2-normalise.  Returns zeros if fewer than 8 points.
    """
    start = max(0, end - window)
    chunk = np.asarray(y[start:end], dtype=float)
    if len(chunk) < 8:
        return np.zeros(n_bands)
    chunk = chunk - np.mean(chunk)
    power = np.abs(np.fft.rfft(chunk)) ** 2
    bands = np.array_split(power[1:], n_bands)
    feature = np.array([b.mean() if len(b) else 0.0 for b in bands])
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def spectral_feature_matrix(y, window=64, n_bands=6):
    """features[t] = spectral_feature(y, t, window): uses y[t-window .. t-1] only."""
    y = np.asarray(y, dtype=float)
    return np.vstack([spectral_feature(y, t, window, n_bands) for t in range(len(y))])


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return float(1.0 / np.sum(w ** 2))


def grid_term(neff, eta):
    """Uniform concentration term sqrt(log(4K/eta)/(2 neff)) + 1/K, K = max(2, ceil(neff))."""
    if not np.isfinite(neff) or neff <= 0:
        return np.nan
    K = max(2, int(np.ceil(neff)))
    return float(np.sqrt(np.log(4.0 * K / eta) / (2.0 * neff)) + 1.0 / K)


def _mass_at(values, weights, q):
    """Total normalised weight on pool values equal to q."""
    if not np.isfinite(q):
        return 0.0
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w[np.asarray(values) == q]) / np.sum(w))


# ----------------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------------

DIAG_KEYS = ["alpha_t", "alpha_eff", "q", "neff", "dbar", "dbar_pre", "drift", "recent_gap",
             "pool_size", "gate_m", "pi", "wmax", "eps_005", "eps_010", "empty"]


class ConformalMethod:
    """Common state and pool construction.

    window      maximum calibration pool size (number of most recent residuals)
    first_valid index of the first usable residual (pool never reaches before it)
    """

    name = "base"
    needs_features = False
    gamma = np.nan

    def __init__(self, alpha=0.10, window=360, first_valid=1):
        self.alpha = float(alpha)
        self.window = int(window)
        self.first_valid = int(first_valid)
        self.reset()

    def reset(self):
        self.alpha_t = self.alpha

    def pool_indices(self, t, residuals):
        idx = np.arange(max(self.first_valid, t - self.window), t)
        if len(idx) == 0:
            return idx
        finite = np.isfinite(residuals[idx])
        return idx[finite]

    def interval(self, t, pred_t, residuals, features=None):
        raise NotImplementedError

    def update(self, miss):
        """Return (alpha_{t+1}, U_t, projection residual); U_t = NaN if not adaptive."""
        return self.alpha_t, np.nan, 0.0

    @staticmethod
    def _diag(**kw):
        base = {k: np.nan for k in DIAG_KEYS}
        base["empty"] = 0
        base.update(kw)
        if np.isfinite(base["neff"]):
            base["eps_005"] = grid_term(base["neff"], 0.05)
            base["eps_010"] = grid_term(base["neff"], 0.10)
        return base

    @staticmethod
    def _free_interval(alpha_t, pred_t):
        """Interval of the free update outside (0, 1), or None inside."""
        if alpha_t <= 0.0:
            return -np.inf, np.inf, {"q": np.inf, "empty": 0}
        if alpha_t >= 1.0:
            return pred_t, pred_t, {"q": 0.0, "empty": 1}
        return None


def _uniform_interval(self, t, pred_t, residuals, alpha_level, free):
    """Shared body of the uniform-weight methods (rolling, ACI, PID)."""
    pool = self.pool_indices(t, residuals)
    r = residuals[pool]
    n = len(r)
    if free:
        fi = self._free_interval(alpha_level, pred_t)
        if fi is not None:
            lo, hi, extra = fi
            return lo, hi, self._diag(alpha_t=alpha_level, alpha_eff=alpha_level, neff=float(n),
                                      pool_size=n, gate_m=self.window, pi=0.0, wmax=1.0 / n, **extra)
    w = np.ones(n)
    q = weighted_quantile(r, w, 1.0 - alpha_level)
    return pred_t - q, pred_t + q, self._diag(
        alpha_t=alpha_level, alpha_eff=alpha_level, q=q, neff=float(n), pool_size=n,
        gate_m=self.window, pi=_mass_at(r, w, q), wmax=1.0 / n)


# ----------------------------------------------------------------------------
# (a) rolling-window split conformal
# ----------------------------------------------------------------------------


class RollingConformal(ConformalMethod):
    name = "rolling"

    def interval(self, t, pred_t, residuals, features=None):
        return _uniform_interval(self, t, pred_t, residuals, self.alpha, free=False)


# ----------------------------------------------------------------------------
# (b) adaptive conformal inference (Gibbs and Candes 2021)
# ----------------------------------------------------------------------------


class ACI(ConformalMethod):
    """Rolling pool with the online update alpha_{t+1} = alpha_t + gamma (alpha - miss).

    clip=True projects alpha_t onto [alpha_min, alpha_max] after each update (the
    original project's convention, cap 0.35).  clip=False is the free update with
    the whole-line / empty-set conventions described in the module docstring.
    """

    name = "aci"

    def __init__(self, alpha=0.10, window=360, first_valid=1, gamma=0.015,
                 clip=True, alpha_min=0.01, alpha_max=0.35):
        self.gamma = float(gamma)
        self.clip = bool(clip)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        super().__init__(alpha, window, first_valid)

    def interval(self, t, pred_t, residuals, features=None):
        return _uniform_interval(self, t, pred_t, residuals, self.alpha_t, free=not self.clip)

    def update(self, miss):
        u = self.alpha_t + self.gamma * (self.alpha - float(miss))
        a = min(max(u, self.alpha_min), self.alpha_max) if self.clip else u
        self.alpha_t = float(a)
        return self.alpha_t, float(u), float(a - u)


# ----------------------------------------------------------------------------
# (c) PID-style controller on alpha_t (as in the original code)
# ----------------------------------------------------------------------------


class PIDConformal(ConformalMethod):
    """Rolling pool; alpha_t driven by a PID-style rule on (alpha - miss).

    This is the original project's controller on the miscoverage level, not the
    quantile-tracking conformal PID of Angelopoulos, Candes and Tibshirani (2023).
    """

    name = "pid"

    def __init__(self, alpha=0.10, window=360, first_valid=1, kp=0.025, ki=0.003,
                 kd=0.01, alpha_min=0.01, alpha_max=0.35):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.alpha_min, self.alpha_max = float(alpha_min), float(alpha_max)
        super().__init__(alpha, window, first_valid)

    def reset(self):
        super().reset()
        self.integral = 0.0
        self.previous = 0.0

    def interval(self, t, pred_t, residuals, features=None):
        return _uniform_interval(self, t, pred_t, residuals, self.alpha_t, free=False)

    def update(self, miss):
        centred = self.alpha - float(miss)
        self.integral += centred
        derivative = centred - self.previous
        self.previous = centred
        u = self.alpha_t + self.kp * centred + self.ki * self.integral + self.kd * derivative
        a = min(max(u, self.alpha_min), self.alpha_max)
        self.alpha_t = float(a)
        return self.alpha_t, float(u), float(a - u)


# ----------------------------------------------------------------------------
# (d) fixed-weight conformal with exponential time-decay weights
# ----------------------------------------------------------------------------


class DecayWeighted(ConformalMethod):
    """Weights decay^(t - i) over the pool, fixed alpha (no online update).

    This is the non-exchangeable weighted split conformal of Barber, Candes,
    Ramdas and Tibshirani (2023) with time-decay weights, implemented as in the
    original project: the weights are normalised over the pool only (no point
    mass on the test point).  See README for the difference.
    """

    name = "decay"

    def __init__(self, alpha=0.10, window=360, first_valid=1, decay=0.99):
        self.decay = float(decay)
        super().__init__(alpha, window, first_valid)

    def interval(self, t, pred_t, residuals, features=None):
        pool = self.pool_indices(t, residuals)
        r = residuals[pool]
        w = self.decay ** (t - pool).astype(float)
        w = w / w.sum()
        q = weighted_quantile(r, w, 1.0 - self.alpha)
        return pred_t - q, pred_t + q, self._diag(
            alpha_t=self.alpha, alpha_eff=self.alpha, q=q,
            neff=effective_sample_size(w), pool_size=len(r), gate_m=self.window,
            pi=_mass_at(r, w, q), wmax=float(w.max()))


# ----------------------------------------------------------------------------
# (e), (f) spectral-weighted conformal: fixed, ACI-updated, and DASC
# ----------------------------------------------------------------------------


class SpectralConformal(ConformalMethod):
    """Spectral kernel weights over the pool, with optional ACI update and gate.

    adaptive=False, gate=False : spectral-only conformal (fixed alpha)
    adaptive=False, gate=True  : DASC-fixed (gate on, fixed alpha)
    adaptive=True,  gate=False : spectral ACI (no drift gate)
    adaptive=True,  gate=True  : DASC (spectral weights + drift score + gate)

    Parameters (original names in brackets):
      window        maximum pool size m_max
      h             kernel bandwidth, weights exp(-d^2 / h^2)
      recent_len    length l_r of the recent-feature average in the drift score (64)
      drift_lambda  drift threshold lambda
      m_min         minimum gated pool size
      gamma, clip, alpha_min, alpha_max  as for ACI
    The drift score D_t is computed whenever features are present so that it can
    be logged for every spectral variant; it changes the interval only when
    gate=True.
    """

    name = "spectral"
    needs_features = True

    def __init__(self, alpha=0.10, window=360, first_valid=1, h=0.55,
                 recent_len=RECENT_LEN, drift_lambda=0.45, m_min=80, gamma=0.015,
                 adaptive=True, gate=True, clip=True, alpha_min=0.01,
                 alpha_max=0.35, min_gated=20):
        self.h = float(h)
        self.recent_len = int(recent_len)
        self.drift_lambda = float(drift_lambda)
        self.m_min = int(m_min)
        self.gamma = float(gamma) if adaptive else np.nan
        self.adaptive = bool(adaptive)
        self.gate = bool(gate)
        self.clip = bool(clip)
        self.alpha_min, self.alpha_max = float(alpha_min), float(alpha_max)
        self.min_gated = int(min_gated)
        super().__init__(alpha, window, first_valid)

    def interval(self, t, pred_t, residuals, features):
        level_alpha = self.alpha_t if self.adaptive else self.alpha
        pool = self.pool_indices(t, residuals)
        current = features[t]
        fp = features[pool]
        # kernel weights over the full pool (unnormalised); the gated pool is a
        # suffix of the pool, so its weights are the same numbers restricted
        diff = fp - current
        d = np.sqrt((diff * diff).sum(axis=1))      # same arithmetic as np.linalg.norm(axis=1)
        wu = np.exp(-(d ** 2) / (self.h ** 2))
        if wu.sum() == 0:
            wu = np.ones_like(wu)
        prelim_w = wu / wu.sum()
        dbar_pre = float(np.sum(prelim_w * d))

        # drift score: recent mean of features (indices t-recent_len .. t-1)
        # against the spectrally weighted pool mean
        recent = np.arange(max(self.first_valid, t - self.recent_len), t)
        recent_feature = features[recent].mean(axis=0)
        weighted_feature = np.multiply(fp, prelim_w[:, None]).sum(axis=0) / prelim_w.sum()
        dd = recent_feature - weighted_feature
        drift = float(np.sqrt(np.dot(dd, dd)))
        rg = recent_feature - current
        recent_gap = float(np.sqrt(np.dot(rg, rg)))

        if self.gate:
            m_max = self.window
            m_t = int(m_max - (m_max - self.m_min) * min(1.0, drift / self.drift_lambda))
            k = int(np.searchsorted(pool, t - m_t, side="left"))
            if len(pool) - k < self.min_gated:
                k = max(0, len(pool) - self.min_gated)
        else:
            m_t = self.window
            k = 0
        gated = pool[k:]
        wg = wu[k:]
        dg = d[k:]
        w = wg / wg.sum()
        neff = float(1.0 / np.sum(w ** 2))
        common = dict(alpha_t=level_alpha, alpha_eff=level_alpha, neff=neff,
                      dbar=float(np.sum(w * dg)), dbar_pre=dbar_pre, drift=drift,
                      recent_gap=recent_gap, pool_size=len(gated), gate_m=m_t,
                      wmax=float(w.max()))
        if self.adaptive and not self.clip:
            fi = self._free_interval(level_alpha, pred_t)
            if fi is not None:
                lo, hi, extra = fi
                return lo, hi, self._diag(pi=0.0, **common, **extra)
        rg_vals = residuals[gated]
        q = weighted_quantile(rg_vals, w, 1.0 - level_alpha)
        return pred_t - q, pred_t + q, self._diag(q=q, pi=_mass_at(rg_vals, w, q), **common)

    def update(self, miss):
        if not self.adaptive:
            return self.alpha, np.nan, 0.0
        u = self.alpha_t + self.gamma * (self.alpha - float(miss))
        a = min(max(u, self.alpha_min), self.alpha_max) if self.clip else u
        self.alpha_t = float(a)
        return self.alpha_t, float(u), float(a - u)


# ----------------------------------------------------------------------------
# Method factory (names used in results files)
# ----------------------------------------------------------------------------

METHOD_NAMES = [
    "rolling", "aci_clip", "aci_noclip", "pid", "decay",
    "spectral_fixed", "spectral_aci", "dasc_fixed", "dasc_clip", "dasc_noclip",
]

METHOD_LABELS = {
    "rolling": "Rolling conformal",
    "aci_clip": "ACI (projected)",
    "aci_noclip": "ACI (free)",
    "pid": "PID-style",
    "decay": "Decay-weighted",
    "spectral_fixed": "Spectral (fixed level)",
    "spectral_aci": "Spectral ACI (no gate)",
    "dasc_fixed": "DASC (fixed level)",
    "dasc_clip": "DASC (projected)",
    "dasc_noclip": "DASC (free)",
}


def make_method(name, alpha=0.10, first_valid=1, **p):
    """Build a method from its name and a parameter dict (missing keys = defaults)."""
    w = int(p.get("window", 360))
    if name == "rolling":
        return RollingConformal(alpha, w, first_valid)
    if name == "aci_clip":
        return ACI(alpha, w, first_valid, gamma=p.get("gamma", 0.015), clip=True)
    if name == "aci_noclip":
        return ACI(alpha, w, first_valid, gamma=p.get("gamma", 0.015), clip=False)
    if name == "pid":
        return PIDConformal(alpha, w, first_valid, kp=p.get("kp", 0.025),
                            ki=p.get("ki", 0.003), kd=p.get("kd", 0.01))
    if name == "decay":
        return DecayWeighted(alpha, w, first_valid, decay=p.get("decay", 0.99))
    common = dict(h=p.get("h", 0.55), recent_len=p.get("recent_len", RECENT_LEN),
                  drift_lambda=p.get("drift_lambda", 0.45), m_min=p.get("m_min", 80),
                  gamma=p.get("gamma", 0.015))
    if name == "spectral_fixed":
        return SpectralConformal(alpha, w, first_valid, adaptive=False, gate=False, **common)
    if name == "dasc_fixed":
        return SpectralConformal(alpha, w, first_valid, adaptive=False, gate=True, **common)
    if name == "spectral_aci":
        return SpectralConformal(alpha, w, first_valid, adaptive=True, gate=False, clip=True, **common)
    if name == "dasc_clip":
        return SpectralConformal(alpha, w, first_valid, adaptive=True, gate=True, clip=True, **common)
    if name == "dasc_noclip":
        return SpectralConformal(alpha, w, first_valid, adaptive=True, gate=True, clip=False, **common)
    raise ValueError(f"unknown method {name}")


# ----------------------------------------------------------------------------
# Stream runner
# ----------------------------------------------------------------------------

STEP_KEYS = ["t", "lo", "hi", "miss", "width", "score", "U", "proj_resid"]


def run_stream(y, pred, method, features=None, start=0, end=None, clip_zero=False,
               alpha=0.10, record=True):
    """Run one method over y[start:end] and return per-step arrays.

    y, pred   full arrays; pred[t] must depend on y[:t] only (caller's duty)
    features  full feature matrix (features[t] from y[:t]); required for spectral
    clip_zero if True the lower limit is replaced by max(lo, 0) for the miss
              indicator and the width (used for non-negative targets)
    Steps with a non-finite pred[t] are skipped.  The adaptive state is updated
    after every evaluated step with the observed miss.
    Returns a dict of numpy arrays (STEP_KEYS and DIAG_KEYS) plus 'gamma' and
    'alpha_final' (the state after the last update).
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(y)
    end = n if end is None else min(int(end), n)
    residuals = np.abs(y - pred)      # residuals[i] is read only for i < t
    method.reset()
    cols = {k: [] for k in STEP_KEYS + DIAG_KEYS}
    for t in range(start, end):
        if not np.isfinite(pred[t]):
            continue
        lo, hi, diag = method.interval(t, pred[t], residuals, features)
        yt = y[t]
        if diag["empty"]:
            miss, width = 1, 0.0
            score = (2.0 / alpha) * abs(yt - pred[t])
        else:
            if clip_zero:
                lo = max(lo, 0.0)
            miss = int(yt < lo or yt > hi)
            width = hi - lo
            score = width + (2.0 / alpha) * max(lo - yt, 0.0) + (2.0 / alpha) * max(yt - hi, 0.0)
        _, u, resid = method.update(miss)
        if record:
            cols["t"].append(t)
            cols["lo"].append(lo)
            cols["hi"].append(hi)
            cols["miss"].append(miss)
            cols["width"].append(width)
            cols["score"].append(score)
            cols["U"].append(u)
            cols["proj_resid"].append(resid)
            for k in DIAG_KEYS:
                cols[k].append(diag[k])
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["gamma"] = float(getattr(method, "gamma", np.nan))
    out["alpha_final"] = float(method.alpha_t)
    return out
