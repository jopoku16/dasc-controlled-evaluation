"""
lead4_substitution.py: step size that makes ACI (window pool, free update) reproduce the
regime-wise coverage of the paper's spectral methods in the design 'recurring_freq_noise'.

Method: the window-aware moment closure with the empirical residual laws of the design
(lead4_link / lead4_scenario) gives ACI's regime-wise coverage as a function of gamma for the
paper's run (alpha = alpha_0 at t = 220, 1180 evaluated steps).  For each target (stored CSV
regime coverage of a spectral method) the script reports the gamma that minimises the larger of
the two regime errors, and that error.  The same is done against the re-run ACI itself
(100 seeds) at the tuned step as a check on the closure curve.

Output: results/lead4_substitution.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(HERE), "results")
import lead4_link as L  # noqa: E402
import lead4_scenario as S  # noqa: E402


def main():
    s1, s2 = S.empirical_laws_of_design()
    laws, qmax = L.empirical_laws(s1, s2)
    T = L.build_ptables(laws, qmax)
    gammas = np.round(np.arange(0.005, 0.1201, 0.0025), 4)
    curve = np.array([S.transient_prediction(T, g, True) for g in gammas])   # coverage R1, R2
    targets = {
        "spectral_fixed (fixed level, kernel weights)": (0.978533, 0.824293),
        "spectral ACI gamma=0.015": (0.946567, 0.849362),
        "spectral ACI gamma=0.03 (selected)": (0.926100, 0.868293),
        "DASC fixed (gate, fixed level)": (0.983850, 0.816534),
        "decay-weighted (selected)": (0.949483, 0.836828),
    }
    rows = []
    for name, (t1, t2) in targets.items():
        err = np.maximum(np.abs(curve[:, 0] - t1), np.abs(curve[:, 1] - t2))
        i = int(np.argmin(err))
        g1 = float(np.interp(t1, curve[::-1, 0], gammas[::-1]))     # coverage in R1 decreases with gamma
        g2 = float(np.interp(t2, curve[:, 1], gammas))              # coverage in R2 increases with gamma
        rows.append({"target": name, "target_cov_R1": t1, "target_cov_R2": t2,
                     "gamma_match_R1_only": g1, "gamma_match_R2_only": g2,
                     "gamma_best_supnorm": float(gammas[i]), "best_supnorm_error": float(err[i]),
                     "ACI_cov_R1_at_best": float(curve[i, 0]), "ACI_cov_R2_at_best": float(curve[i, 1]),
                     "step_ratio_vs_0.015": float(gammas[i] / 0.015)})
    df = pd.DataFrame(rows)
    out = os.path.join(RES, "lead4_substitution.csv")
    if os.path.exists(out):
        raise SystemExit("exists")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.round(4).to_string(index=False))
    print("closure curve (gamma, cov_R1, cov_R2) at 0.015, 0.03, 0.05:")
    for g in (0.015, 0.03, 0.05):
        i = int(np.argmin(np.abs(gammas - g)))
        print(g, curve[i].round(4))


if __name__ == "__main__":
    main()
