"""
Causal reference implementation of the DASC step used by the theory checks.

This mirrors dasc.py and the dasc() function of run_first_simulation.py of the earlier project
(pool = indices t-window, ..., t-1; feature at t computed from y[t-l], ..., y[t-1];
recent mean over features t-l, ..., t-1; gate; floor of 20; weighted quantile with
searchsorted(side="left")).  It adds the quantities the revised theory needs:

  dbar      weight-averaged spectral distance over the gated pool (final weights)
  dbar_pre  weight-averaged spectral distance under the pre-gate weights
  mass_q    total final weight on pool residuals equal to the returned quantile
  wmax      largest final weight
  recent_gap  || mean of recent features - S_t ||_2

and an adaptive update with two modes:

  clipped   alpha_{t+1} = clip(alpha_t + gamma (alpha - E_t), alpha_min, alpha_max)
  free      alpha_{t+1} = alpha_t + gamma (alpha - E_t) with the conventions
            alpha_t <= 0 -> interval is the whole line (E_t = 0)
            alpha_t >= 1 -> interval is empty (E_t = 1)
            (the weighted quantile at level 1 - alpha_t is used for alpha_t in (0, 1);
            these are the conventions stated in the revised theory section)

Nothing here is used by the experiments; the experiments have their own library.
"""
import numpy as np


def spectral_feature(y, end, window=64, n_bands=6):
    """Normalised banded periodogram of y[end-window], ..., y[end-1] (exclusive end)."""
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


def all_features(y, window=64, n_bands=6):
    y = np.asarray(y, dtype=float)
    return np.vstack([spectral_feature(y, i, window, n_bands) for i in range(len(y))])


def weighted_quantile(values, weights, q):
    """Smallest value v with sum of weights of {values <= v} >= q (weights normalised)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    k = np.searchsorted(cdf, q, side="left")
    k = min(k, len(values) - 1)
    return values[k]


def spectral_weights(features, current_feature, bandwidth):
    distances = np.linalg.norm(features - current_feature, axis=1)
    weights = np.exp(-(distances ** 2) / (bandwidth ** 2))
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    return weights / weights.sum(), distances


def dasc_step(residuals, features, t, alpha_t, window=360, h=0.55, drift_lambda=0.45,
              m_min=80, recent_len=64, gate=True):
    """One DASC prediction step at time t.  Returns q_t and diagnostics.

    residuals[i] must be the residual R_i for i < t (anything at index >= t is not read).
    features[i] must be the causal feature S_i (computed from observations before i).
    """
    current = features[t]
    pool = np.arange(max(1, t - window), t)
    pre_w, pre_d = spectral_weights(features[pool], current, h)

    recent = np.arange(max(1, t - recent_len), t)
    recent_feature = features[recent].mean(axis=0)
    weighted_feature = np.average(features[pool], axis=0, weights=pre_w)
    drift = float(np.linalg.norm(recent_feature - weighted_feature))

    if gate:
        m_t = int(window - (window - m_min) * min(1.0, drift / drift_lambda))
        gated = pool[pool >= t - m_t]
        if len(gated) < 20:
            gated = pool[-min(len(pool), 20):]
    else:
        m_t = int(window)
        gated = pool

    w, d = spectral_weights(features[gated], current, h)
    neff = float(1.0 / np.sum(w ** 2))
    level = 1.0 - alpha_t
    if level >= 1.0:
        q = np.inf
    elif level <= 0.0:
        q = -np.inf
    else:
        q = float(weighted_quantile(residuals[gated], w, level))
    mass_q = float(np.sum(w[residuals[gated] == q])) if np.isfinite(q) else 0.0
    return {
        "q": q,
        "alpha_t": float(alpha_t),
        "drift": drift,
        "neff": neff,
        "m_t": m_t,
        "n_pool": int(len(gated)),
        "dbar": float(np.sum(w * d)),
        "dbar_pre": float(np.sum(pre_w * pre_d)),
        "recent_gap": float(np.linalg.norm(recent_feature - current)),
        "mass_q": mass_q,
        "wmax": float(w.max()),
    }


def update_alpha(alpha_t, miss, alpha, gamma, mode="clipped", alpha_min=0.01, alpha_max=0.35):
    """Return (alpha_{t+1}, projection residual eps_t)."""
    u = alpha_t + gamma * (alpha - miss)
    if mode == "free":
        return u, 0.0
    a = min(max(u, alpha_min), alpha_max)
    return a, a - u


def grid_term(neff, eta, K=None):
    """Uniform concentration term of the grid lemma: sqrt(log(4K/eta)/(2 neff)) + 1/K."""
    if K is None:
        K = int(np.ceil(neff))
    K = max(int(K), 2)   # the theorem's default is max{2, ceil(n_eff)}
    return float(np.sqrt(np.log(4.0 * K / eta) / (2.0 * neff)) + 1.0 / K)


def grid_term_best(neff, eta, kmax=100000):
    """Minimum over integer K of the grid term (computable, used for reporting)."""
    ks = np.unique(np.concatenate([np.arange(1, 200), np.geomspace(200, kmax, 60).astype(int)]))
    vals = np.sqrt(np.log(4.0 * ks / eta) / (2.0 * neff)) + 1.0 / ks
    j = int(np.argmin(vals))
    return float(vals[j]), int(ks[j])
