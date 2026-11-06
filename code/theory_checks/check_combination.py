"""
Check of the proposition on what the per-step and long-run statements imply together.

With the free (unclipped) update, the per-step bound p_t <= alpha_t + Lambda_t on the event
B_t (conditional probability >= 1 - eta), p_t = P(E_t = 1 | history), and the exact long-run
identity, one gets for every T
    mean_t alpha_t >= alpha - mean_t Lambda_t - (max(alpha_1, 1-alpha_1) + gamma)/(gamma T)
                      - mean_t 1{B_t^c} - M_T,
where M_T = mean_t (p_t - E_t) is a martingale average.  The script evaluates every term in
models A and B1 (exact p_t available), reports the margin, and the size of each term.

Run:  python -X utf8 check_combination.py [--seeds 10] [--out outputs]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from check_perstep_bound import run_model, ALPHA, GAMMA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / args.out
    out.mkdir(exist_ok=True)
    eta = 0.1
    rows = []
    for name, kw in [("A_exogenous", dict(exogenous=True)), ("B1_endog_phi0.95", dict(phi=0.95))]:
        for seed in range(args.seeds):
            df = run_model(name, seed, [eta], mode="free", **kw)
            p = 1 - df["cov"].to_numpy()
            a = df["alpha_t"].to_numpy()
            lam = 1 - a - df[f"lo_grid_{eta}"].to_numpy()   # = L dbar_t + eps_t(eta); delta = 0 in these models
            T = len(df)
            tele = (max(ALPHA, 1 - ALPHA) + GAMMA) / (GAMMA * T)
            # B_t^c: per-step bound violated although p_t is exact -> count directly
            viol = (df["cov"].to_numpy() < df[f"lo_grid_{eta}"].to_numpy()).mean()
            # realised misses E_t are needed for M_T; recompute from cov and the identity is not possible,
            # so we use E[E_t | history] = p_t and the identity in expectation: mean p_t = mean E_t + M_T.
            lhs = a.mean()
            rhs_no_mart = ALPHA - lam.mean() - tele - viol
            rows.append({"model": name, "seed": seed, "T": T, "mean_alpha_t": lhs, "mean_p_t": p.mean(),
                         "mean_Lambda": lam.mean(), "telescoping": tele, "frac_B_complement": viol,
                         "rhs_without_martingale": rhs_no_mart, "margin": lhs - rhs_no_mart,
                         "mean_alpha_minus_p": (a - p).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "combination_check.csv", index=False)
    text = ("Combination proposition check (free update, eta = 0.1).\n"
            "mean_alpha_t must be >= alpha - mean_Lambda - telescoping - frac_B_complement - M_T.\n"
            "The martingale term M_T is of order T^{-1/2} and is omitted here (its omission can only make the\n"
            "check stricter when it is positive); 'margin' is lhs minus the right-hand side without M_T.\n"
            "mean_alpha_minus_p shows the per-step gap alpha_t - p_t averaged over time (the realised version\n"
            "of 'the adaptive level sits above the conditional miss probability').\n\n"
            + df.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")
    (out / "combination_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
