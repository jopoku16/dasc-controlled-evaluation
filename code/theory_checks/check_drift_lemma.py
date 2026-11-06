"""
Check of the drift-score inequality and search for counterexamples to the old remark.

New lemma:  D_t <= || Sbar_t^recent - S_t ||_2 + dbar_t^pre,
            where dbar_t^pre is the weight-averaged spectral distance under the PRE-GATE weights.
Old remark: D_t <= dbar_t (final, gated weights) "by Jensen when the recent mean is close to S_t".

The script evaluates both on the synthetic stream (several seeds) and on random feature
configurations, and reports the number of violations of each.

Run:  python -X utf8 check_drift_lemma.py [--out outputs]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dasc_causal import all_features, dasc_step


def simulate_stream(n=1400, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    freq = np.piecewise(t.astype(float), [t < 450, (t >= 450) & (t < 900), t >= 900],
                        [0.035, 0.055, lambda z: 0.055 + 0.00006 * (z - 900)])
    amp = np.where((t // 180) % 2 == 0, 1.0, 1.7)
    level = np.where(t >= 900, 1.2, 0.0)
    sigma = np.where(t >= 900, 0.75, 0.45)
    return level + amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / args.out
    out.mkdir(exist_ok=True)
    rows = []
    for seed in range(10):
        y = simulate_stream(seed=seed)
        resid = np.abs(y - np.r_[np.nan, y[:-1]]); resid[0] = 0
        feats = all_features(y, 64)
        for t in range(220, 1400):
            d = dasc_step(resid, feats, t, 0.1, 360, 0.55, 0.45, 80, 64)
            rows.append({"source": "synthetic", "seed": seed, "t": t, "D_t": d["drift"], "dbar": d["dbar"],
                         "dbar_pre": d["dbar_pre"], "recent_gap": d["recent_gap"],
                         "new_rhs": d["recent_gap"] + d["dbar_pre"]})
    rng = np.random.default_rng(0)
    for k in range(5000):
        n = 360
        F = rng.random((n + 1, 6)); F = F / np.linalg.norm(F, axis=1, keepdims=True)
        feats = np.vstack([np.zeros((1, 6)), F])          # index 0 unused
        resid = rng.random(n + 2)
        t = n + 1
        h = rng.choice([0.2, 0.55, 1.0])
        d = dasc_step(resid, feats, t, 0.1, n, h, 0.45, 80, 64)
        rows.append({"source": "random", "seed": k, "t": t, "D_t": d["drift"], "dbar": d["dbar"],
                     "dbar_pre": d["dbar_pre"], "recent_gap": d["recent_gap"],
                     "new_rhs": d["recent_gap"] + d["dbar_pre"]})
    df = pd.DataFrame(rows)
    df["viol_new"] = df["D_t"] > df["new_rhs"] + 1e-12
    df["viol_old"] = df["D_t"] > df["dbar"] + 1e-12
    df["excess_old"] = df["D_t"] - df["dbar"]
    summ = df.groupby("source").agg(cases=("t", "size"), viol_new=("viol_new", "sum"), viol_old=("viol_old", "sum"),
                                    max_excess_old=("excess_old", "max")).reset_index()
    text = ("Drift-score lemma check.\n"
            "viol_new: violations of D_t <= recent_gap + dbar_pre (must be 0).\n"
            "viol_old: violations of the old remark D_t <= dbar_t (final weights).\n\n"
            + summ.to_string(index=False) + "\n")
    df.to_csv(out / "drift_lemma_cases.csv.gz", index=False, compression="gzip")
    (out / "drift_lemma_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
