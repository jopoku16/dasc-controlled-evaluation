"""
run_timing.py: per-step computation time of every method at each dataset's
maximum pool size, on a synthetic stream of the same length scale.

Writes ../results/timing.csv (method, pool, feature window, microseconds per
step, steps timed).  Run on an otherwise idle machine; wall-clock numbers.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as ds
from dasc_lib import METHOD_NAMES, make_method, run_stream, spectral_feature_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"

SETTINGS = [("simulation", 360, 64, 80), ("finance", 504, 64, 126),
            ("electricity", 1440, 168, 336), ("weather", 2160, 336, 504)]


def main():
    rows = []
    y, _ = ds.sim_original_three_phase(n=6000, seed=0)
    pred = np.r_[np.nan, y[:-1]]
    for label, pool, fw, m_min in SETTINGS:
        feats = spectral_feature_matrix(y, fw)
        start = pool + fw
        for name in METHOD_NAMES:
            m = make_method(name, window=pool, h=0.55, recent_len=fw, drift_lambda=0.45,
                            m_min=m_min, gamma=0.015, decay=0.99)
            run_stream(y, pred, m, features=feats, start=start, end=start + 200)   # warm-up
            t0 = time.perf_counter()
            out = run_stream(y, pred, m, features=feats, start=start)
            dt = time.perf_counter() - t0
            rows.append({"setting": label, "pool": pool, "feature_window": fw, "method": name,
                         "us_per_step": 1e6 * dt / len(out["t"]), "steps": len(out["t"])})
            print(f"{label:12s} pool={pool:5d} {name:15s} {rows[-1]['us_per_step']:7.0f} us/step", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "timing.csv", index=False)


if __name__ == "__main__":
    main()
