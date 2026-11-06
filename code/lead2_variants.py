"""
lead2_variants.py: phase-aware and calendar-hour DASC variants (lead 2 kill test).

Nothing in dasc_lib.py is edited.  The variants subclass SpectralConformal and
replace only the kernel weights:

    weight(i, t) = exp(-(d_mag(i, t)^2 + w^2 c(i, t)^2) / h^2)

d_mag is the unchanged six-band magnitude distance; c is the chord distance
2 |sin((a_t - a_i) / 2)| between the angles a_t and a_i of the auxiliary
unit vectors (cos a, sin a).  For the phase variant a_t is the phase of the
daily-frequency DFT coefficient of y[t-m .. t-1]; for the calendar variant a_t
is 2 pi hour(t) / 24 (the clock hour is deterministic and known before t).

The drift score D_t, its gate and the ACI update use the six magnitude columns
and the magnitude-only kernel exactly as in DASC (the mean of cos and sin over
64 steps is not meaningful because 64 is not a multiple of 24).  With w = 0 the
class reproduces SpectralConformal bit for bit.

Causality: aux[t] is built from y[:t] (phase) or from the clock (calendar), as
features[t] is in dasc_lib.
"""

from __future__ import annotations

import numpy as np

from dasc_lib import SpectralConformal, weighted_quantile, _mass_at

PERIOD = 24


def phase_angles(y, window):
    """Angle a_t of the daily-frequency coefficient Z_t = sum_{j=1..m} (y[t-j] - mean) exp(2 pi i j / 24),
    m = min(t, window), for every t.  Uses y[t-m .. t-1] only.  Zero angle (and a flag) for m < 24.

    For m = window (a multiple of 24) Z_t equals the real-FFT coefficient at bin window/24 of the
    mean-removed window y[t-window .. t-1].  For a pure daily harmonic A cos(2 pi u / 24 - theta),
    arg Z_t = 2 pi t / 24 - theta: the phase at the forecast time t, advancing by 2 pi / 24 per step.
    Returns (angle, amplitude, valid) arrays of length len(y).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    ang = np.zeros(n)
    amp = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    # full windows: Z_t = sum_{n'=0}^{L-1} y[t-L+n'] exp(2 pi i (L-n')/24); the sum of exp over whole cycles is 0
    # so the mean removal is not needed when L is a multiple of 24
    L = int(window)
    if L % PERIOD == 0 and n > L:
        e = np.exp(2j * np.pi * (L - np.arange(L)) / PERIOD)
        win = np.lib.stride_tricks.sliding_window_view(y, L)       # win[s] = y[s .. s+L-1]; t = s + L
        z = win[: n - L] @ e                                         # t = L .. n-1
        ang[L:] = np.angle(z)
        amp[L:] = np.abs(z)
        valid[L:] = True
        lo = PERIOD
        hi = L
    else:
        lo, hi = PERIOD, n
    for t in range(lo, min(hi, n)):
        m = min(t, L)
        chunk = y[t - m:t]
        j = np.arange(m, 0, -1)                                      # chunk[0] has lag m, chunk[-1] has lag 1
        z = np.sum((chunk - chunk.mean()) * np.exp(2j * np.pi * j / PERIOD))
        ang[t] = np.angle(z)
        amp[t] = np.abs(z)
        valid[t] = True
    return ang, amp, valid


def unit_vectors(angle):
    return np.column_stack([np.cos(angle), np.sin(angle)])


def calendar_angles(hour):
    return 2.0 * np.pi * np.asarray(hour, dtype=float) / PERIOD


def chord_sq(a, b):
    """Squared chord distance between unit vectors of angles a and b (arrays broadcast)."""
    return 4.0 * np.sin((a - b) / 2.0) ** 2


class PhaseKernelConformal(SpectralConformal):
    """DASC with the product kernel above.  adaptive=True, gate=True, clip=True (the projected DASC)."""

    name = "dasc_aux"

    def __init__(self, alpha=0.10, window=360, first_valid=1, h=0.55, w=1.0, aux_angle=None,
                 aux_valid=None, drift_lambda=0.45, m_min=80, gamma=0.015, **kw):
        super().__init__(alpha=alpha, window=window, first_valid=first_valid, h=h,
                         drift_lambda=drift_lambda, m_min=m_min, gamma=gamma,
                         adaptive=True, gate=True, clip=True, **kw)
        self.w = float(w)
        self.aux_angle = None if aux_angle is None else np.asarray(aux_angle, dtype=float)
        self.aux_valid = None if aux_valid is None else np.asarray(aux_valid, dtype=bool)

    def interval(self, t, pred_t, residuals, features):
        level_alpha = self.alpha_t
        pool = self.pool_indices(t, residuals)
        current = features[t]
        fp = features[pool]
        diff = fp - current
        d = np.sqrt((diff * diff).sum(axis=1))
        wu = np.exp(-(d ** 2) / (self.h ** 2))
        if wu.sum() == 0:
            wu = np.ones_like(wu)
        prelim_w = wu / wu.sum()                       # magnitude-only weights, used by the drift score (as DASC)
        dbar_pre = float(np.sum(prelim_w * d))

        recent = np.arange(max(self.first_valid, t - self.recent_len), t)
        recent_feature = features[recent].mean(axis=0)
        weighted_feature = np.multiply(fp, prelim_w[:, None]).sum(axis=0) / prelim_w.sum()
        dd = recent_feature - weighted_feature
        drift = float(np.sqrt(np.dot(dd, dd)))
        rg = recent_feature - current
        recent_gap = float(np.sqrt(np.dot(rg, rg)))

        m_max = self.window
        m_t = int(m_max - (m_max - self.m_min) * min(1.0, drift / self.drift_lambda))
        k = int(np.searchsorted(pool, t - m_t, side="left"))
        if len(pool) - k < self.min_gated:
            k = max(0, len(pool) - self.min_gated)
        gated = pool[k:]
        wg = wu[k:]
        dg = d[k:]
        if self.w != 0.0 and self.aux_angle is not None:
            c2 = chord_sq(self.aux_angle[gated], self.aux_angle[t])
            log_extra = -(self.w ** 2) * c2 / (self.h ** 2)
            if self.aux_valid is not None:
                # members whose phase is undefined (window shorter than one day) get the largest chord, never a boost
                bad = ~self.aux_valid[gated]
                if bad.any():
                    log_extra = np.where(bad, -(self.w ** 2) * 4.0 / (self.h ** 2), log_extra)
            wg2 = wg * np.exp(log_extra)
            if not np.isfinite(wg2.sum()) or wg2.sum() <= 0:
                # underflow guard: renormalise in the log domain
                lg = -(dg ** 2) / (self.h ** 2) + log_extra
                lg = lg - lg.max()
                wg2 = np.exp(lg)
            wg = wg2
        w = wg / wg.sum()
        neff = float(1.0 / np.sum(w ** 2))
        common = dict(alpha_t=level_alpha, alpha_eff=level_alpha, neff=neff,
                      dbar=float(np.sum(w * dg)), dbar_pre=dbar_pre, drift=drift,
                      recent_gap=recent_gap, pool_size=len(gated), gate_m=m_t,
                      wmax=float(w.max()))
        rg_vals = residuals[gated]
        q = weighted_quantile(rg_vals, w, 1.0 - level_alpha)
        return pred_t - q, pred_t + q, self._diag(q=q, pi=_mass_at(rg_vals, w, q), **common)
