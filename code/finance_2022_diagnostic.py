"""
finance_2022_diagnostic.py: month-by-month descriptive check of the drift score on the finance stream.

Reads  ../results/diagnostics/finance_naive1_dasc_clip.parquet and finance_naive1_rolling.parquet
Writes ../results/finance_2022_drift_diagnostic.csv

For every month of the evaluation period it reports the mean drift score D_t of DASC (projected, naive forecaster),
the mean rolling 50-day coverage deficit of rolling conformal prediction (rolling mean of the miss indicator minus
0.10), DASC's monthly miss rate, the number of steps, and the rank of the month among all evaluated months by each of
the two means (1 = largest).  Descriptive only; it tests nothing.

Usage (from code/):  python -X utf8 finance_2022_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"


def finance_2022():
    p_d = RES / "diagnostics" / "finance_naive1_dasc_clip.parquet"
    p_r = RES / "diagnostics" / "finance_naive1_rolling.parquet"
    if not (p_d.exists() and p_r.exists()):
        return
    d = pd.read_parquet(p_d)
    r = pd.read_parquet(p_r)
    d["month"] = pd.to_datetime(d["datetime"]).dt.to_period("M").astype(str)
    miss = r["miss"].to_numpy(dtype=float)
    deficit = pd.Series(miss).rolling(50).mean().to_numpy() - 0.10
    d["rolling_deficit"] = deficit
    d = d[d["in_eval"]]                       # evaluation period only
    m = d.groupby("month").agg(drift_mean=("drift", "mean"), deficit_mean=("rolling_deficit", "mean"),
                               miss_rate_dasc=("miss", "mean"), n=("miss", "size")).reset_index()
    m = m.dropna(subset=["drift_mean", "deficit_mean"]).reset_index(drop=True)
    m["drift_rank"] = m["drift_mean"].rank(ascending=False, method="min").astype(int)
    m["deficit_rank"] = m["deficit_mean"].rank(ascending=False, method="min").astype(int)
    m.to_csv(RES / "finance_2022_drift_diagnostic.csv", index=False)
    print(f"wrote {RES / 'finance_2022_drift_diagnostic.csv'} with {len(m)} months")


if __name__ == "__main__":
    finance_2022()
