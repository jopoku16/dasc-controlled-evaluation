"""
check_reproduce_original.py: regression anchor for the new framework.

Shows that dasc_lib reproduces the DASC numbers published in the original
manuscript (Tables 4, 10, 11, 12) when run under the original settings, and
reports the same runs with the interval centred on the forecaster that the
baselines used.  The published hourly DASC rows were produced with the interval
centred on y[t-1] while the residual pool came from the 24-hour lag; the
"recentred" rows centre on y[t-24] like every baseline.

Published values (from the original manuscript):
    synthetic (10 seeds)  DASC 0.8995 / 2.7517
    electricity           DASC 0.9033 / 2.5508   ACI 0.9001 / 3.5398
    weather               DASC 0.9656 / 6.4129   ACI 0.8938 / 11.0270
    finance (lo clipped)  DASC 0.8988 / 2.2791   ACI 0.8954 / 2.2615
Writes ../results/reproduction_check.csv.

What agrees and what does not, and why (verified step by step):
  * synthetic and finance DASC rows agree to the published precision when the
    original's early-window feature quirk is reproduced (legacy rows below);
    with dasc_lib's features the synthetic width moves by 0.0007.
  * hourly DASC rows differ by 0.001 (electricity) and 0.003 (weather) in
    width because the original dasc() kept the 23 NaN residuals of the lag-24
    forecaster inside the pool for the first m_max steps (t < 1464 for
    electricity, t < 2184 for weather); dasc_lib excludes non-finite
    residuals.  After those steps the two implementations agree at every
    step (0 of 10,536 electricity steps differ).
  * ACI rows differ by at most one order statistic because the original
    baselines used numpy's 'higher' quantile and the new framework uses one
    weighted quantile for every method.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
from dasc_lib import ACI, SpectralConformal, spectral_feature_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"


def legacy_spectral_feature_matrix(y, window):
    """The original run_first_simulation.spectral_feature, kept only for this check.

    It differs from dasc_lib.spectral_feature for windows of 8 to 12 points
    (t = 8, ..., 11): there it returns the raw, zero-padded, un-normalised
    periodogram instead of banded normalised features.  Those four early vectors
    sit in the calibration pool for the first m_max steps and move the drift
    score enough to change the gated pool by one index at a few steps, which is
    the whole difference between the published numbers and dasc_lib.
    """
    y = np.asarray(y, dtype=float)

    def one(end):
        start = max(0, end - window)
        chunk = y[start:end]
        if len(chunk) < 8:
            return np.zeros(6)
        chunk = chunk - np.mean(chunk)
        power = np.abs(np.fft.rfft(chunk)) ** 2
        if len(power) < 7:
            return np.pad(power, (0, 7 - len(power)))[:6]
        bands = np.array_split(power[1:], 6)
        feature = np.array([b.mean() for b in bands])
        denom = np.linalg.norm(feature)
        return feature / denom if denom > 0 else feature

    return np.vstack([one(t) for t in range(len(y))])

PUBLISHED = {
    ("synthetic", "DASC"): (0.8995, 2.7517), ("synthetic", "ACI"): (0.8990, 2.7381),
    ("electricity", "DASC"): (0.9033, 2.5508), ("electricity", "ACI"): (0.9001, 3.5398),
    ("weather", "DASC"): (0.9656, 6.4129), ("weather", "ACI"): (0.8938, 11.0270),
    ("finance", "DASC"): (0.8988, 2.2791), ("finance", "ACI"): (0.8954, 2.2615),
}


def run_original_style(y, pred_resid, pred_centre, features, start, window, m_min,
                       aci_window, clip_zero, gamma=0.015):
    """Mimic the original drivers: residual pool from pred_resid, DASC centred on
    pred_centre, ACI centred on pred_resid with its own (smaller) window."""
    residuals = np.abs(y - pred_resid)
    dasc = SpectralConformal(0.10, window, 1, h=0.55, recent_len=64, drift_lambda=0.45,
                             m_min=m_min, gamma=gamma, adaptive=True, gate=True, clip=True)
    aci = ACI(0.10, aci_window, 1, gamma=gamma, clip=True)
    d_miss, d_w, a_miss, a_w = [], [], [], []
    for t in range(start, len(y)):
        if not np.isfinite(pred_resid[t]) or not np.isfinite(pred_centre[t]):
            continue
        lo, hi, _ = dasc.interval(t, pred_centre[t], residuals, features)
        if clip_zero:
            lo = max(lo, 0.0)
        miss = int(y[t] < lo or y[t] > hi)
        dasc.update(miss)
        d_miss.append(miss)
        d_w.append(hi - lo)
        lo, hi, _ = aci.interval(t, pred_resid[t], residuals, features)
        if clip_zero:
            lo = max(lo, 0.0)
        miss = int(y[t] < lo or y[t] > hi)
        aci.update(miss)
        a_miss.append(miss)
        a_w.append(hi - lo)
    return (1 - np.mean(d_miss), np.mean(d_w)), (1 - np.mean(a_miss), np.mean(a_w))


def main():
    t0 = time.time()
    rows = []

    def add(dataset, variant, method, cov, width):
        pc, pw = PUBLISHED.get((dataset, method), (np.nan, np.nan))
        rows.append({"dataset": dataset, "variant": variant, "method": method,
                     "coverage": cov, "avg_width": width,
                     "published_coverage": pc, "published_width": pw})

    # synthetic, ten seeds, lag-1 everywhere (no centring issue)
    for label, fmat in [("as published, legacy early-window features", legacy_spectral_feature_matrix),
                        ("as published, dasc_lib features", spectral_feature_matrix)]:
        dm, dw, am, aw = [], [], [], []
        for seed in range(10):
            y = ds.sim_original_three_phase(seed=seed)[0]
            pred = np.r_[np.nan, y[:-1]]
            feats = fmat(y, 64)
            (dc, dwid), (ac, awid) = run_original_style(y, pred, pred, feats, 220, 360, 80, 180, False)
            dm.append(dc); dw.append(dwid); am.append(ac); aw.append(awid)
        add("synthetic", label, "DASC", np.mean(dm), np.mean(dw))
        if "legacy" in label:
            add("synthetic", "as published", "ACI", np.mean(am), np.mean(aw))

    specs = {
        "electricity": (ds.load_electricity, 24, 168, 1440, 336, 720, 1080, False),
        "weather": (ds.load_weather, 24, 336, 2160, 504, 1080, 1440, False),
        "finance": (ds.load_finance, 1, 64, 504, 126, 252, 420, True),
    }
    for name, (loader, lag, fw, window, m_min, aci_w, start, clip_zero) in specs.items():
        meta = loader()
        y = meta["y"]
        pred_lag = np.full(len(y), np.nan); pred_lag[lag:] = y[:-lag]
        pred_1 = np.r_[np.nan, y[:-1]]
        feats_legacy = legacy_spectral_feature_matrix(y, fw)
        feats = spectral_feature_matrix(y, fw)
        # as published: DASC centred on y[t-1], residuals from the dataset lag
        (dc, dwid), (ac, awid) = run_original_style(y, pred_lag, pred_1, feats_legacy, start, window,
                                                     m_min, aci_w, clip_zero)
        add(name, "as published (DASC centred on y[t-1]), legacy features", "DASC", dc, dwid)
        add(name, "as published", "ACI", ac, awid)
        (dc, dwid), _ = run_original_style(y, pred_lag, pred_1, feats, start, window,
                                            m_min, aci_w, clip_zero)
        add(name, "as published (DASC centred on y[t-1]), dasc_lib features", "DASC", dc, dwid)
        if lag != 1:
            (dc2, dwid2), _ = run_original_style(y, pred_lag, pred_lag, feats, start, window,
                                                  m_min, aci_w, clip_zero)
            add(name, f"recentred on y[t-{lag}] (same forecaster as baselines)", "DASC", dc2, dwid2)
        print(f"{name}: done ({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    df["coverage_diff"] = df["coverage"] - df["published_coverage"]
    df["width_diff"] = df["avg_width"] - df["published_width"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "reproduction_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
