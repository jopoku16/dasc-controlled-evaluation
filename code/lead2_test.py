"""
lead2_test.py: correctness and causality checks for the phase-aware and calendar-hour variants.

1. phase_angles equals the real-FFT bin angle of the mean-removed window, and advances by 2 pi / 24 per step on a pure daily harmonic;
2. w = 0 reproduces dasc_lib's SpectralConformal (projected DASC) bit for bit;
3. causality, same perturbation idea as test_causality.py: replacing y[t:] (including y[t]) by unrelated values must leave
   angle[t] (and every angle at or before t), features[t], pred[t] and the interval at t bit-for-bit unchanged, for the phase
   variant and the calendar variant, with the naive-24 and ridge forecasters, on a synthetic hourly-like stream and on slices
   of the real electricity and weather series;
4. negative control: a deliberately leaky phase feature (angle computed from a window that includes y[t]) must be detected.

Writes results/lead2_causality_report.csv.  Exit code 0 on success.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DASC_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data" / "real"))   # same default as datasets.py

import numpy as np
import pandas as pd

import datasets as ds
import forecasters as fc
from dasc_lib import SpectralConformal, run_stream, spectral_feature_matrix
from lead2_variants import PhaseKernelConformal, calendar_angles, phase_angles
from test_causality import hourly_like_stream, perturb

RESULTS = Path(__file__).resolve().parent.parent / "results"


def test_phase_definition():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1500) + np.cos(2 * np.pi * np.arange(1500) / 24 - 1.0)
    ok = True
    for L in (168, 336):
        ang, amp, valid = phase_angles(y, L)
        for t in (L, L + 17, 700, 1200, 1499):
            z = np.fft.rfft(y[t - L:t] - y[t - L:t].mean())[L // 24]
            ok &= abs(np.angle(np.exp(1j * (np.angle(z) - ang[t])))) < 1e-9 and abs(abs(z) - amp[t]) < 1e-8
    # pure harmonic: arg Z_t = 2 pi t / 24 - theta
    theta = 1.3
    x = np.cos(2 * np.pi * np.arange(1000) / 24 - theta)
    ang, _, _ = phase_angles(x, 168)
    for t in (30, 100, 168, 400, 999):          # includes partial windows (t < 168, m not a multiple of 24)
        pred = np.angle(np.exp(1j * (2 * np.pi * t / 24 - theta)))
        err = abs(np.angle(np.exp(1j * (ang[t] - pred))))
        if t >= 48:
            ok &= err < 0.15       # partial windows leak a little (non-integer number of cycles); full windows are exact
        if t >= 168:
            ok &= err < 1e-9
    return bool(ok)


def test_w0_identity():
    meta = ds.REAL_DATASETS["weather"]["load"]()
    y = meta["y"][:3200]
    pred = fc.seasonal_naive(y, 24)
    F = spectral_feature_matrix(y, 168)
    ang, _, valid = phase_angles(y, 168)
    kw = dict(alpha=0.10, window=1080, first_valid=1, h=0.55, drift_lambda=0.15, m_min=540, gamma=0.03)
    base = SpectralConformal(adaptive=True, gate=True, clip=True, **kw)
    new = PhaseKernelConformal(w=0.0, aux_angle=ang, aux_valid=valid, **kw)
    a = run_stream(y, pred, base, features=F, start=1440)
    b = run_stream(y, pred, new, features=F, start=1440)
    return all(np.array_equal(a[k], b[k]) for k in ("lo", "hi", "miss", "width", "score", "alpha_t"))


def make_method(kind, y, meta, fw, params, w):
    if kind == "phase":
        ang, _, valid = phase_angles(y, fw)
    else:
        ang, valid = calendar_angles(meta["hour"]), np.ones(len(y), dtype=bool)
    return PhaseKernelConformal(alpha=0.10, window=params["pool"], first_valid=1, h=0.55, w=w, aux_angle=ang,
                                aux_valid=valid, drift_lambda=0.45, m_min=params["m_min"], gamma=0.015), ang


def interval_at(y, pred, F, kind, meta, fw, params, w, t, start, leaky=False):
    m, ang = make_method(kind, y, meta, fw, params, w)
    if leaky:
        # leaky aux: the angle used at time s is the angle of the window ending at s (inclusive of y[s])
        shifted = np.concatenate([ang[1:], ang[-1:]])
        m.aux_angle = shifted
    out = run_stream(y, pred, m, features=F, start=start, end=t + 1)
    assert out["t"][-1] == t
    return out["lo"][-1], out["hi"][-1]


def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)
    rows, failures = [], 0
    p1 = test_phase_definition()
    print("phase definition vs real FFT bin and pure harmonic:", "PASS" if p1 else "FAIL", flush=True)
    p2 = test_w0_identity()
    print("w = 0 reproduces projected DASC bit for bit:", "PASS" if p2 else "FAIL", flush=True)
    rows.append({"check": "phase_definition", "pass": p1})
    rows.append({"check": "w0_identity", "pass": p2})
    failures += (not p1) + (not p2)

    streams = []
    hl = hourly_like_stream(n=2400)
    streams.append(("hourly_like", hl["y"], hl, 1080, 168, {"pool": 720, "m_min": 360}))
    for name in ("electricity", "weather"):
        m = ds.REAL_DATASETS[name]["load"]()
        n = 2600
        meta = {k: (v[:n] if hasattr(v, "__len__") and len(v) >= n else v) for k, v in m.items()}
        streams.append((name + "_slice", meta["y"], meta, 1080, 168, {"pool": 720, "m_min": 360}))

    for sname, y, meta, start, fw, params in streams:
        n = len(y)
        preds = ds.hourly_forecasts(y, meta, with_xgb=False)
        F = spectral_feature_matrix(y, fw)
        test_ts = sorted(set([start + 1, start + 300, 1700, 2000, n - 1, 900 + 168 * 3, 900 + 168 * 3 + 1]))
        test_ts = [t for t in test_ts if t < n and t >= start]
        perturbed = {}
        for t in test_ts:
            yp = perturb(y, t, rng)
            perturbed[t] = (yp, ds.hourly_forecasts(yp, meta, with_xgb=False), spectral_feature_matrix(yp, fw))
        for fname, pred in preds.items():
            for kind, w in (("phase", 1.0), ("phase", 3.0), ("cal", 1.0)):
                ok_all = True
                for t in test_ts:
                    if not np.isfinite(pred[t]):
                        continue
                    lo, hi = interval_at(y, pred, F, kind, meta, fw, params, w, t, start)
                    yp, preds_p, F_p = perturbed[t]
                    pred_p = preds_p[fname]
                    lo_p, hi_p = interval_at(yp, pred_p, F_p, kind, meta, fw, params, w, t, start)
                    ang, amp, _ = phase_angles(y, fw)
                    ang_p, amp_p, _ = phase_angles(yp, fw)
                    angle_same = bool(np.array_equal(ang[:t + 1], ang_p[:t + 1]) and np.array_equal(amp[:t + 1], amp_p[:t + 1]))
                    same = (lo == lo_p) and (hi == hi_p) and (pred[t] == pred_p[t]) and np.array_equal(F[t], F_p[t]) and angle_same
                    rows.append({"stream": sname, "forecaster": fname, "method": f"{kind}_w{w:g}", "t": t, "lo": lo, "lo_perturbed": lo_p,
                                 "hi": hi, "hi_perturbed": hi_p, "angle_same": angle_same, "interval_same": bool((lo == lo_p) and (hi == hi_p)),
                                 "pass": bool(same)})
                    ok_all &= bool(same)
                failures += (not ok_all)
                print(f"{sname:18s} {fname:8s} {kind:6s} w={w:g}  {'PASS' if ok_all else 'FAIL'}", flush=True)

    # negative control: leaky aux angle must be caught
    sname, y, meta, start, fw, params = streams[0]
    preds = ds.hourly_forecasts(y, meta, with_xgb=False)
    pred = preds["naive24"]
    F = spectral_feature_matrix(y, fw)
    detected = False
    for t in (1500, 1800, 2100):
        lo, hi = interval_at(y, pred, F, "phase", meta, fw, params, 3.0, t, start, leaky=True)
        yp = perturb(y, t, rng)
        predp = ds.hourly_forecasts(yp, meta, with_xgb=False)["naive24"]
        Fp = spectral_feature_matrix(yp, fw)
        lo_p, hi_p = interval_at(yp, predp, Fp, "phase", meta, fw, params, 3.0, t, start, leaky=True)
        if (lo != lo_p) or (hi != hi_p):
            detected = True
    rows.append({"stream": "negative_control_leaky_phase", "method": "phase_w3_leaky", "pass": detected})
    print("negative control (leaky phase angle):", "DETECTED (good)" if detected else "NOT DETECTED (bad)", flush=True)
    failures += (not detected)
    pd.DataFrame(rows).to_csv(RESULTS / "lead2_causality_report.csv", index=False)
    n_checks = sum(1 for r in rows if "t" in r and not (isinstance(r.get("t"), float) and np.isnan(r["t"])))
    print(f"{n_checks} interval checks, {failures} failures, {time.time() - t0:.0f} s")
    print("LEAD2 TESTS PASSED" if failures == 0 else "LEAD2 TESTS FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
