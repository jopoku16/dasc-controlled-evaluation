"""
lead4_match.py: parts (ii) and (iii) of the lead.

(ii) Weighted quantile at the fixed level alpha_0 with mass w on the same regime:
     q_r solves  w S_r(q) + (1-w) S_other(q) = alpha_0,  miss_r = S_r(q_r).
(iii) The step size that makes ACI's regime-wise coverage equal to that of the weighted
     quantile.  At the level of the averaged ODE the ACI miss rates depend on gamma and m
     only through x = gamma m, so gamma*(m) = x*(w, rho)/m.  ACI's two miss rates are
     symmetric about alpha_0 (miss_1 = alpha_0 - d, miss_2 = alpha_0 + d); the weighted
     quantile's are alpha_0 + c -/+ d_W.  The best sup-norm match sets d = d_W and leaves
     the common-mode error |c|, c = (miss_1^W + miss_2^W)/2 - alpha_0, which no step size
     can remove.

Writes results/lead4_match.csv (one row per (rho, w)) and prints derivation checks:
  * ODE scaling: F2 depends on (gamma, m) only through gamma*m;
  * F1 closed-form matching rule versus the ODE rule;
  * exact chain at gamma* = x*/m for m in {50, 150, 450} versus the weighted quantile.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(HERE), "results")
import lead4_model as M  # noqa: E402

RHOS = [1.5, 2.0, 3.0]
WS = [0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]
A0 = 0.1


def ode_gap_curve(pool, xs, m0=100):
    """ACI gap g(x) = miss_2 - miss_1 of the averaged ODE on a grid of x = gamma m."""
    g = []
    for x in xs:
        mi1, mi2, A, B = M.f2_ode(pool, x / m0, m0, hscale=0.1)
        g.append(mi2 - mi1)
    return np.array(g)


def f1_gap(pool, x):
    q0, p0, s0 = pool.zero_order()
    c = (A0 - p0) / s0
    a = np.exp(-s0 * x)
    delta = (c[0] - c[1]) * (1 - a[0]) * (1 - a[1]) / (1 - a[0] * a[1])
    return 2 * delta / x


def main():
    # ---- check 1: ODE depends on gamma*m only
    pool = M.Pool((1.0, 3.0))
    a = M.f2_ode(pool, 0.02, 150)
    b = M.f2_ode(pool, 0.06, 50)
    c = M.f2_ode(pool, 0.01, 300)
    print("scaling check, x = 3.0 reached by (0.02,150), (0.06,50), (0.01,300):")
    print("  miss_1:", a[0], b[0], c[0], " max spread:", max(abs(a[0] - b[0]), abs(a[0] - c[0])))

    xs = np.exp(np.linspace(np.log(0.05), np.log(60.0), 28))
    rows = []
    curves = {}
    for rho in RHOS:
        pool = M.Pool((1.0, rho))
        q0, p0, s0 = pool.zero_order()
        g = ode_gap_curve(pool, xs)
        curves[rho] = (xs, g)
        interp = PchipInterpolator(np.log(xs), g)
        g0 = p0[1] - p0[0]
        for w in WS:
            m1w, m2w = M.weighted_miss(pool, w)
            m1l, m2l = M.weighted_miss_linear(pool, w)
            gW = m2w - m1w
            cW = 0.5 * (m1w + m2w) - A0
            row = {"rho": rho, "w": w, "p1_frozen": p0[0], "p2_frozen": p0[1], "s1": s0[0], "s2": s0[1],
                   "W_miss1": m1w, "W_miss2": m2w, "W_lin_miss1": m1l, "W_lin_miss2": m2l,
                   "W_gap": gW, "W_common_mode": cW}
            if w >= 1.0 - 1e-12 or gW <= 1e-9:
                row.update({"x_star_ode": np.inf, "x_star_F1": np.inf})
            else:
                if interp(np.log(xs[0])) < gW:
                    xo = np.nan
                elif interp(np.log(xs[-1])) > gW:
                    xo = np.inf
                else:
                    xo = float(np.exp(brentq(lambda lx: float(interp(lx)) - gW, np.log(xs[0]), np.log(xs[-1]))))
                try:
                    x1 = float(brentq(lambda x: f1_gap(pool, x) - gW, 1e-3, 400.0))
                except ValueError:
                    x1 = np.nan
                row.update({"x_star_ode": xo, "x_star_F1": x1})
            rows.append(row)
    df = pd.DataFrame(rows)

    # ---- exact-chain check at gamma* = x*/m: error of the best match
    chk = []
    for _, r in df.iterrows():
        if not np.isfinite(r["x_star_ode"]) or r["x_star_ode"] > 60:
            continue
        pool = M.Pool((1.0, r["rho"]))
        for m in (50, 150, 450):
            gam = r["x_star_ode"] / m
            if gam > 0.2:
                continue
            f4 = M.f4_exact_chain(pool, gam, m)
            e1 = f4[0] - r["W_miss1"]
            e2 = f4[1] - r["W_miss2"]
            chk.append({"rho": r["rho"], "w": r["w"], "m": m, "gamma_star": gam, "x_star": r["x_star_ode"],
                        "chain_miss1": f4[0], "chain_miss2": f4[1], "W_miss1": r["W_miss1"], "W_miss2": r["W_miss2"],
                        "err_regime1": e1, "err_regime2": e2, "max_abs_err": max(abs(e1), abs(e2)),
                        "predicted_err_abs_cW": abs(r["W_common_mode"])})
    ck = pd.DataFrame(chk)
    df.to_csv(os.path.join(RES, "lead4_match.csv"), index=False)
    ck.to_csv(os.path.join(RES, "lead4_match_chain_check.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df[["rho", "w", "W_miss1", "W_miss2", "W_lin_miss1", "W_lin_miss2", "W_gap", "W_common_mode", "x_star_ode", "x_star_F1"]].round(5).to_string(index=False))
    print()
    print(ck.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
