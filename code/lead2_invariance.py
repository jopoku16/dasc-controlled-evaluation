"""
lead2_invariance.py: Task 1 of the phase-blindness kill test.

(a) exact periodic signal: |X_k| of a window of length l is unchanged when the window start moves by whole
    samples (period divides l), while arg X_k at k = l/24 advances by 2 pi / 24 per sample;
(b) real series (electricity, weather): how much of the magnitude features and of the daily-bin phase is a
    function of the clock hour; per-step phase increment; windows that span a gap in the electricity rows;
(c) Spearman correlation of feature distance with circular hour difference, with a day-block bootstrap;
(d) how much weight the tuned DASC kernel puts on pool members within one hour of the current hour.

Outputs (results/): lead2_invariance_synthetic.csv, lead2_invariance_real.csv,
lead2_feature_hour_spearman.csv, lead2_dasc_weight_hour_mass.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DASC_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data" / "real"))   # same default as datasets.py

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import datasets as ds
from dasc_lib import spectral_feature, spectral_feature_matrix
from lead2_variants import calendar_angles, chord_sq, phase_angles

RESULTS = Path(__file__).resolve().parent.parent / "results"


# ----------------------------------------------------------------------------
# (a) exact periodic signal
# ----------------------------------------------------------------------------

def synthetic_check():
    rng = np.random.default_rng(1)
    rows = []
    for period in (24, 25):
        pattern = rng.normal(size=period)
        pattern = pattern - pattern.mean()
        x = np.tile(pattern, 40)
        for L in (168, 336):
            ks = np.arange(1, L // 2 + 1)
            starts = np.arange(0, 200)
            mags = np.array([np.abs(np.fft.rfft(x[s:s + L] - x[s:s + L].mean()))[ks] for s in starts])
            max_dev = float(np.max(np.abs(mags - mags[0])))
            feats = np.array([spectral_feature(x, s + L, L) for s in starts])
            feat_dev = float(np.max(np.abs(feats - feats[0])))
            k = L // 24
            ph = np.array([np.angle(np.fft.rfft(x[s:s + L] - x[s:s + L].mean())[k]) for s in starts])
            inc = np.angle(np.exp(1j * np.diff(ph)))
            rows.append({"period": period, "window": L, "divides": bool(L % period == 0),
                         "max_abs_dev_of_|X_k|_over_200_shifts_all_k": max_dev,
                         "max_abs_dev_of_banded_feature": feat_dev,
                         "daily_bin_k": k,
                         "phase_increment_median": float(np.median(inc)),
                         "phase_increment_min": float(inc.min()), "phase_increment_max": float(inc.max()),
                         "target_2pi_over_24": 2 * np.pi / 24})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "lead2_invariance_synthetic.csv", index=False)
    print(df.to_string(), flush=True)


# ----------------------------------------------------------------------------
# helpers for real series
# ----------------------------------------------------------------------------

def circ_mean_resultant(a):
    return float(np.abs(np.mean(np.exp(1j * a))))


def anova_r2(values, groups):
    """Share of variance of `values` explained by the group means."""
    v = np.asarray(values, dtype=float)
    g = pd.Series(groups)
    gm = pd.Series(v).groupby(g.values).transform("mean").to_numpy()
    tot = np.sum((v - v.mean()) ** 2)
    return float(np.sum((gm - v.mean()) ** 2) / tot) if tot > 0 else np.nan


def real_checks(name, meta, dev_end):
    y = meta["y"]
    n = len(y)
    hour = meta["hour"]
    dt = pd.to_datetime(meta["datetime"])
    out = []
    for fw in (168, 336):
        ang, amp, valid = phase_angles(y, fw)
        F = spectral_feature_matrix(y, fw)
        ev = np.arange(dev_end, n)
        # windows y[t-fw..t-1] that contain a missing-hour gap (rows are consecutive non-missing hours)
        span = (dt[ev - 1].to_numpy() - dt[ev - fw].to_numpy()) / np.timedelta64(1, "h")
        gap_free = np.isclose(span, fw - 1)
        rel = np.angle(np.exp(1j * (ang[ev] - calendar_angles(hour[ev]))))
        R_all = circ_mean_resultant(rel)
        R_gapfree = circ_mean_resultant(rel[gap_free])
        # circular-linear association: offset relative to clock hour, by hour
        inc = np.angle(np.exp(1j * np.diff(ang[ev])))
        ok = gap_free[1:] & gap_free[:-1]
        # magnitude features: between-hour share of variance (by hour of t) and within-day versus across-day
        row = {"dataset": name, "window": fw, "n_eval": len(ev), "frac_windows_with_gap": float(1 - gap_free.mean()),
               "phase_minus_clock_mean_resultant_all": R_all, "phase_minus_clock_mean_resultant_gapfree": R_gapfree,
               "phase_minus_clock_circ_sd_gapfree": float(np.sqrt(-2 * np.log(max(R_gapfree, 1e-12)))),
               "phase_minus_clock_mean_offset_rad": float(np.angle(np.mean(np.exp(1j * rel[gap_free])))),
               "phase_increment_median": float(np.median(inc[ok])),
               "phase_increment_q25": float(np.quantile(inc[ok], 0.25)),
               "phase_increment_q75": float(np.quantile(inc[ok], 0.75)),
               "target_increment": 2 * np.pi / 24,
               "amp_R2_by_hour": anova_r2(amp[ev], hour[ev])}
        for b in range(F.shape[1]):
            row[f"band{b + 1}_R2_by_hour"] = anova_r2(F[ev, b], hour[ev])
        row["band_R2_by_hour_mean"] = float(np.mean([row[f"band{b + 1}_R2_by_hour"] for b in range(F.shape[1])]))
        # R2 by day (24-step blocks) as a comparison: how much of the feature variance is across-day
        day = (ev // 24)
        row["band_R2_by_day_mean"] = float(np.mean([anova_r2(F[ev, b], day) for b in range(F.shape[1])]))
        out.append(row)
    return out


def spearman_with_day_bootstrap(dist, dh, day, n_boot=2000, seed=0):
    """Spearman correlation of dist with dh, 95% interval by resampling whole days with global ranks fixed."""
    dist = np.round(np.asarray(dist, dtype=float), 10)      # exact ties for equal circular hour differences
    rx, ry = rankdata(dist), rankdata(dh)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    df = pd.DataFrame({"day": day, "x": rx, "y": ry})
    df["xx"], df["yy"], df["xy"] = df.x ** 2, df.y ** 2, df.x * df.y
    g = df.groupby("day").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxx=("xx", "sum"),
                              syy=("yy", "sum"), sxy=("xy", "sum"))
    G = g.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    nb = len(G)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        S = G[rng.integers(0, nb, size=nb)].sum(axis=0)
        N, sx, sy, sxx, syy, sxy = S
        cov = sxy / N - sx * sy / N ** 2
        vx = sxx / N - (sx / N) ** 2
        vy = syy / N - (sy / N) ** 2
        boots[b] = cov / np.sqrt(vx * vy)
    return rho, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def feature_hour_spearman(name, meta, dev_end, n_partner=30, lag_max=1440, seed=0):
    y = meta["y"]
    n = len(y)
    hour = meta["hour"]
    rng = np.random.default_rng(seed)
    t_all = np.arange(dev_end, n)
    t = np.repeat(t_all, n_partner)
    i = t - rng.integers(1, lag_max + 1, size=len(t))
    dh = np.abs(hour[t] - hour[i])
    dh = np.minimum(dh, 24 - dh)
    lag = t - i
    day = t // 24
    cal_chord = np.sqrt(chord_sq(calendar_angles(hour[t]), calendar_angles(hour[i])))
    rows = []
    for fw in (168, 336):
        F = spectral_feature_matrix(y, fw)
        ang, amp, valid = phase_angles(y, fw)
        d_mag = np.sqrt(((F[t] - F[i]) ** 2).sum(axis=1))
        c_ph = np.sqrt(chord_sq(ang[t], ang[i]))
        d_comb = np.sqrt(d_mag ** 2 + c_ph ** 2)
        for label, dist in (("magnitude (6 bands)", d_mag), ("phase chord", c_ph),
                            ("magnitude + phase (w = 1)", d_comb), ("clock-hour chord (benchmark)", cal_chord),
                            ("recency |t - i| (benchmark)", lag.astype(float))):
            # sign convention: positive means larger distance goes with larger hour difference
            rho, lo, hi = spearman_with_day_bootstrap(dist, dh, day)
            rho_lag, lo_l, hi_l = spearman_with_day_bootstrap(dist, lag.astype(float), day)
            rows.append({"dataset": name, "window": fw, "distance": label, "n_pairs": len(t),
                         "spearman_with_circular_hour_diff": rho, "ci_lo": lo, "ci_hi": hi,
                         "spearman_with_lag": rho_lag, "lag_ci_lo": lo_l, "lag_ci_hi": hi_l})
    return rows


def weight_hour_mass(name, meta, dev_end, cfg, n_t=600, seed=0):
    """Mass of the tuned DASC kernel (pre-gate, magnitude-only) on pool members within one clock hour of the current hour."""
    y = meta["y"]
    n = len(y)
    hour = meta["hour"]
    F = spectral_feature_matrix(y, int(cfg["fw"]))
    rng = np.random.default_rng(seed)
    ts = rng.choice(np.arange(dev_end, n), size=n_t, replace=False)
    rows = []
    for t in ts:
        pool = np.arange(max(1, t - int(cfg["window"])), t)
        d = np.sqrt(((F[pool] - F[t]) ** 2).sum(axis=1))
        wu = np.exp(-(d ** 2) / cfg["h"] ** 2)
        w = wu / wu.sum()
        dh = np.abs(hour[pool] - hour[t])
        dh = np.minimum(dh, 24 - dh)
        near = dh <= 1
        lag = t - pool
        rows.append({"mass_within_1h": float(w[near].sum()), "pool_share_within_1h": float(near.mean()),
                     "neff_over_pool": float(1.0 / np.sum(w ** 2) / len(pool)),
                     "weight_lag_spearman": float(np.corrcoef(rankdata(w), rankdata(-lag))[0, 1])})
    r = pd.DataFrame(rows).mean()
    return {"dataset": name, "window": int(cfg["window"]), "h": float(cfg["h"]), "fw": int(cfg["fw"]),
            "n_steps": n_t, **{k: float(v) for k, v in r.items()}}


def main():
    synthetic_check()
    real_rows, sp_rows, mass_rows = [], [], []
    for name in ("electricity", "weather"):
        cfgd = ds.REAL_DATASETS[name]
        meta = cfgd["load"]()
        dev_end = int(cfgd["dev_frac"] * len(meta["y"]))
        real_rows += real_checks(name, meta, dev_end)
        print(pd.DataFrame(real_rows[-2:]).T.to_string(), flush=True)
        sp_rows += feature_hour_spearman(name, meta, dev_end)
        print(pd.DataFrame(sp_rows[-10:]).to_string(), flush=True)
        for fname in ("naive24", "ridge"):
            ch = pd.read_csv(RESULTS / "tuning" / f"{name}_{fname}_dascgamma_chosen.csv")
            cfg = ch[ch["variant"] == "dasc_clip"].iloc[0]
            m = weight_hour_mass(name, meta, dev_end, cfg)
            m["forecaster"] = fname
            mass_rows.append(m)
            print(m, flush=True)
    pd.DataFrame(real_rows).to_csv(RESULTS / "lead2_invariance_real.csv", index=False)
    pd.DataFrame(sp_rows).to_csv(RESULTS / "lead2_feature_hour_spearman.csv", index=False)
    pd.DataFrame(mass_rows).to_csv(RESULTS / "lead2_dasc_weight_hour_mass.csv", index=False)


if __name__ == "__main__":
    main()
