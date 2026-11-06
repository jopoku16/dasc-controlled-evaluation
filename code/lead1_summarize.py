"""
lead1_summarize.py: aggregate lead1_reps_kappa*.csv into ../results/lead1_cells.csv (one row per cell and estimator)
and print the pre-stated criteria.

Size of the test of H0: L <= L0 at the true L0 is P(Lhat > L0) with L0 the true constant (dual of the coverage
of L by the lower bound: coverage = 1 - size).  Power is P(Lhat > L0) when the true constant is 3 * L0, which on the same
data means testing L0 = L / 3.  Two targets: L_path (global Lipschitz constant on the realised path, the task's "true L")
and L_cl (the sharp cluster-pair population target; L_cl <= L_path).

engine_cov = share of replications in which sup_r |(Fhat_A - F_A) - (Fhat_B - F_B)| <= the method's margin for EVERY
retained pair (the true F_A is known); this is the event that makes Lhat a valid lower bound for L_cl (nominal 0.95).

Criteria (fixed in advance): (a) size within 0.05 +/- 0.02 (strict two-sided reading) or coverage >= 0.93 (one-sided
reading, size <= 0.07) in at least 80% of cells; (b) power >= 0.8 at 3 L; KILL if size > 0.10 in more than a third of cells.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
METHODS = ["C0", "C0b", "C1", "C1b", "B1", "B2", "B3"]
LABEL = {"C0": "existing DKW term, eta per set", "C0b": "existing DKW term, eta/J",
         "C1": "blocked effective-size DKW term, eta per set", "C1b": "blocked effective-size DKW term, eta/J",
         "B1": "block bootstrap, fixed b", "B2": "block bootstrap, Politis-White b",
         "B3": "studentised block bootstrap, Politis-White b"}


def cell_table(df):
    rows = []
    for (kappa, n, rho, d), g in df.groupby(["kappa", "n", "rho", "d"]):
        g = g[g["L_cl"].notna()]
        R = len(g)
        base = dict(kappa=kappa, n=n, rho=rho, d=d, reps=R, J_mean=g.J.mean(), L_path_mean=g.L_path.mean(),
                    L_cl_mean=g.L_cl.mean(), L_cl_over_L_path=(g.L_cl / g.L_path).mean(), tau_hat_mean=g.tau_hat.mean(),
                    lag1_R_mean=g.lag1_R.mean(), lag1_pit_mean=g.lag1_pit.mean(), b_fixed=g.b_fixed.mean(),
                    b_pw_mean=g.b_pw.mean())
        for m in METHODS:
            x = g[m].to_numpy()
            r = dict(base, method=m, description=LABEL[m], Lhat_mean=x.mean(), Lhat_pos=(x > 0).mean(),
                     Lhat_over_L_path=(x / g.L_path).mean())
            if "ev_" + m in g.columns:
                ev = float(g["ev_" + m].mean())
                r["engine_cov"] = ev
                r["engine_cov_se"] = float(np.sqrt(max(ev * (1 - ev), 1e-12) / R))
            for tag, tgt in (("path", g.L_path.to_numpy()), ("cl", g.L_cl.to_numpy())):
                size = float(np.mean(x > tgt))
                pw = float(np.mean(x > tgt / 3.0))
                r[f"size_{tag}"] = size
                r[f"size_{tag}_se"] = float(np.sqrt(max(size * (1 - size), 1e-12) / R))
                r[f"power3_{tag}"] = pw
                r[f"power3_{tag}_se"] = float(np.sqrt(max(pw * (1 - pw), 1e-12) / R))
            rows.append(r)
    return pd.DataFrame(rows)


def criteria(ct):
    out = []
    for (kappa, m), g in ct.groupby(["kappa", "method"]):
        n_cells = len(g)
        for tag in ("path", "cl"):
            s = g[f"size_{tag}"]
            p = g[f"power3_{tag}"]
            ec = g["engine_cov"] if "engine_cov" in g.columns else pd.Series([np.nan] * n_cells)
            out.append(dict(kappa=kappa, method=m, target=tag, cells=n_cells,
                            engine_cov_ge_0_93=int((ec >= 0.93).sum()), engine_cov_min=float(ec.min()),
                            a_cover_ok=int((s <= 0.07).sum()), a_twosided_ok=int(((s >= 0.03) & (s <= 0.07)).sum()),
                            kill_size_gt_0_10=int((s > 0.10).sum()), b_power_ge_0_8=int((p >= 0.8).sum()),
                            a_cover_frac=float((s <= 0.07).mean()), kill_frac=float((s > 0.10).mean()),
                            b_frac=float((p >= 0.8).mean())))
    return pd.DataFrame(out)


def main():
    files = sorted(RESULTS.glob("lead1_reps_kappa*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ct = cell_table(df)
    ct["exploratory"] = ct["rho"] > 0.9          # extra high-dependence cells, outside the pre-stated grid
    ct.to_csv(RESULTS / "lead1_cells.csv", index=False)
    cr = criteria(ct[~ct["exploratory"]])
    cr.to_csv(RESULTS / "lead1_criteria.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(cr.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
