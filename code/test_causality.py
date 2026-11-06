"""
test_causality.py: every method and every forecaster must be causal.

For a stream y, a forecaster and a method, the interval at time t is computed
twice: once on y and once on y' where y'[t:] (including y'[t]) is replaced by
unrelated values.  Forecasts, residuals and spectral features are recomputed
from y'.  The test passes only if lo, hi, the forecast pred[t] and the feature
vector features[t] are bit-for-bit identical in the two runs, at every test
time, for every method and forecaster.

A deliberately leaky forecaster (pred[t] = y[t]) is included as a negative
control; the suite requires that it FAILS, which shows the test has power.

Run:  python -X utf8 test_causality.py
Exit code 0 on success.  Writes ../results/causality_test_report.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
import forecasters as fc
from dasc_lib import METHOD_NAMES, make_method, run_stream, spectral_feature_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"


def interval_at(y, pred, features, method_name, params, t, start, clip_zero):
    m = make_method(method_name, alpha=0.10, first_valid=1, **params)
    out = run_stream(y, pred, m, features=features, start=start, end=t + 1,
                    clip_zero=clip_zero, record=True)
    assert out["t"][-1] == t, "stream did not reach t"
    return out["lo"][-1], out["hi"][-1]


def perturb(y, t, rng):
    yp = y.copy()
    yp[t:] = 100.0 + 25.0 * rng.standard_normal(len(y) - t)
    return yp


def hourly_like_stream(n=2600, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 2.0 + np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 0.4, n)
    dt = pd.date_range("2020-01-01", periods=n, freq="h")
    return {"y": y, "hour": dt.hour.to_numpy(), "dow": dt.dayofweek.to_numpy()}


def leaky_forecaster(y, meta=None):
    return {"leaky": y.copy()}            # pred[t] = y[t]: must be caught


def main():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    rows = []
    failures = 0

    streams = []
    # 1. simulation stream, simulation forecasters, default simulation settings
    sim = ds.make_sim("original_three_phase", seed=0)
    streams.append(("sim_three_phase", sim["y"], None, ds.sim_forecasts, 220, False, 64, 360, 80))
    # 2. hourly-like stream with calendar terms, ridge and xgboost refits
    hl = hourly_like_stream()
    streams.append(("hourly_like", hl["y"], hl, lambda y, meta: ds.hourly_forecasts(y, meta, with_xgb=True),
                    1080, False, 168, 1440, 336))
    # 3. non-negative daily-like stream with the finance design and clip_zero
    fin = np.abs(ds.sim_stationary_ar1(n=900, seed=5)[0]) + 0.1
    streams.append(("daily_like_clipzero", fin, None, lambda y, meta: ds.daily_forecasts(y, meta, with_xgb=True),
                    252, True, 64, 504, 126))

    for sname, y, meta, fbuild, start, clip_zero, fw, pool, m_min in streams:
        n = len(y)
        preds = fbuild(y, meta)
        feats = spectral_feature_matrix(y, window=fw)
        # test times: spread over the evaluated range, plus refit boundaries
        test_ts = sorted(set(np.linspace(start, n - 1, 5).astype(int).tolist()
                             + [start + 1, n - 1]))
        if sname == "hourly_like":
            test_ts += [900 + 168 * 3, 900 + 168 * 3 - 1, 900 + 168 * 3 + 1]
        if sname == "daily_like_clipzero":
            test_ts += [126 + 21 * 10, 126 + 21 * 10 - 1]
        # the perturbation does not depend on the method: build the perturbed
        # forecasts and features once per test time and reuse them
        perturbed = {}
        for t in test_ts:
            yp = perturb(y, t, rng)
            perturbed[t] = (yp, fbuild(yp, meta), spectral_feature_matrix(yp, window=fw))
        for fname, pred in preds.items():
            for mname in METHOD_NAMES:
                params = dict(window=pool, h=0.55, recent_len=fw, drift_lambda=0.45,
                              m_min=m_min, gamma=0.015, decay=0.99)
                ok_all = True
                for t in test_ts:
                    if not np.isfinite(pred[t]):
                        continue
                    lo, hi = interval_at(y, pred, feats, mname, params, t, start, clip_zero)
                    yp, preds_p, feats_p = perturbed[t]
                    pred_p = preds_p[fname]
                    lo_p, hi_p = interval_at(yp, pred_p, feats_p, mname, params, t, start, clip_zero)
                    same = (lo == lo_p) and (hi == hi_p) and (pred[t] == pred_p[t]) \
                        and np.array_equal(feats[t], feats_p[t])
                    rows.append({"stream": sname, "forecaster": fname, "method": mname, "t": t,
                                 "lo": lo, "lo_perturbed": lo_p, "hi": hi, "hi_perturbed": hi_p,
                                 "pred_same": pred[t] == pred_p[t],
                                 "features_same": bool(np.array_equal(feats[t], feats_p[t])),
                                 "interval_same": bool((lo == lo_p) and (hi == hi_p)),
                                 "pass": bool(same)})
                    ok_all &= same
                if not ok_all:
                    failures += 1
                print(f"{sname:20s} {fname:8s} {mname:15s} {'PASS' if ok_all else 'FAIL'}", flush=True)

    # negative control: a leaky forecaster must be detected
    y = sim["y"]
    feats = spectral_feature_matrix(y, window=64)
    pred = leaky_forecaster(y)["leaky"]
    detected = False
    for t in [300, 700, 1100]:
        lo, hi = interval_at(y, pred, feats, "aci_clip", dict(window=360), t, 220, False)
        yp = perturb(y, t, rng)
        pred_p = leaky_forecaster(yp)["leaky"]
        feats_p = spectral_feature_matrix(yp, window=64)
        lo_p, hi_p = interval_at(yp, pred_p, feats_p, "aci_clip", dict(window=360), t, 220, False)
        if (lo != lo_p) or (hi != hi_p):
            detected = True
    rows.append({"stream": "negative_control_leaky", "forecaster": "leaky", "method": "aci_clip",
                 "t": -1, "pass": detected, "interval_same": not detected})
    print(f"negative control (leaky forecaster): {'DETECTED (good)' if detected else 'NOT DETECTED (bad)'}")
    if not detected:
        failures += 1

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "causality_test_report.csv", index=False)
    n_checks = sum(1 for r in rows if r["stream"] != "negative_control_leaky")
    print(f"\n{n_checks} interval checks, {failures} failing method/forecaster combinations, "
          f"{time.time() - t0:.0f} s")
    if failures:
        print("CAUSALITY TEST FAILED")
        sys.exit(1)
    print("CAUSALITY TEST PASSED")


if __name__ == "__main__":
    main()
