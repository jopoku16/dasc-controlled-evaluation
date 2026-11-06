"""
Check of the uniform concentration lemma for a weighted empirical distribution.

Statement checked: with fixed weights w_i >= 0 summing to one, independent residuals with
CDFs F_i, Fhat_w(r) = sum_i w_i 1{R_i <= r}, Fbar_w(r) = sum_i w_i F_i(r), and
n_eff = 1 / sum_i w_i^2, for every integer K >= 2 and eta in (0,1),
    P( sup_r |Fhat_w(r) - Fbar_w(r)| > sqrt(log(4K/eta)/(2 n_eff)) + 1/K ) <= eta.

Cases: continuous laws (half-normal with scale depending on a feature), discrete laws
(atoms, so Fbar_w has jumps), several weight profiles (uniform, Gaussian-kernel weights of
varying sharpness).  The sup is computed exactly over the jump points of Fhat_w and Fbar_w,
including left limits.  Reports the exceedance frequency over replications for each eta and
two choices of K (K = ceil(n_eff) and the minimising K), and the old displayed term
sqrt(log(2/eta)/(2 n_eff)) without any uniformity factor.

Run:  python -X utf8 check_grid_lemma.py [--reps 4000] [--out outputs]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

from dasc_causal import grid_term, grid_term_best


def sup_dev_continuous(R, w, cdf_at):
    """sup_r |Fhat(r) - Fbar(r)| when Fbar is continuous: attained at jump points of Fhat."""
    order = np.argsort(R, kind="stable")
    Rs = R[order]
    ws = w[order]
    Fhat_right = np.cumsum(ws)
    Fhat_left = Fhat_right - ws
    Fbar = cdf_at(Rs)
    return float(max(np.max(np.abs(Fhat_right - Fbar)), np.max(np.abs(Fhat_left - Fbar))))


def sup_dev_discrete(R, w, support, pmf_matrix):
    """Exact sup over r for laws supported on a common finite grid; pmf_matrix[i, k] = P(R_i = support[k])."""
    Fbar = np.cumsum(w @ pmf_matrix)                       # at support points (right limits)
    Fhat = np.array([np.sum(w[R <= s]) for s in support])
    Fbar_left = np.r_[0.0, Fbar[:-1]]
    Fhat_left = np.r_[0.0, Fhat[:-1]]
    return float(max(np.max(np.abs(Fhat - Fbar)), np.max(np.abs(Fhat_left - Fbar_left))))


def make_weights(n, kind, rng):
    if kind == "uniform":
        w = np.ones(n)
    else:
        sharp = {"kernel_mild": 1.0, "kernel_sharp": 4.0, "kernel_very_sharp": 12.0}[kind]
        d = rng.random(n)
        w = np.exp(-sharp * d ** 2 / 0.1)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / args.out
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    etas = [0.02, 0.05, 0.1, 0.2]
    rows = []
    support = np.arange(0, 11, dtype=float)
    for law in ["continuous", "discrete"]:
        for n in [60, 360]:
            for kind in ["uniform", "kernel_mild", "kernel_sharp", "kernel_very_sharp"]:
                w = make_weights(n, kind, rng)
                neff = 1.0 / np.sum(w ** 2)
                sig = 0.6 + 0.8 * rng.random(n)            # scale of residual i
                if law == "discrete":
                    # geometric-like pmf on 0..10 with success parameter depending on sig
                    p = 1.0 / (1.0 + sig)
                    pmf = np.array([(1 - pk) ** support * pk for pk in p])
                    pmf = pmf / pmf.sum(axis=1, keepdims=True)
                    cum = np.cumsum(pmf, axis=1)
                devs = np.empty(args.reps)
                for r in range(args.reps):
                    if law == "continuous":
                        R = sig * np.abs(rng.normal(0, 1, n))
                        cdf_at = lambda x: np.sum(w[:, None] * erf(x[None, :] / (sig[:, None] * math.sqrt(2))), axis=0)
                        devs[r] = sup_dev_continuous(R, w, cdf_at)
                    else:
                        uu = rng.random(n)
                        R = np.array([support[np.searchsorted(cum[i], uu[i])] for i in range(n)])
                        devs[r] = sup_dev_discrete(R, w, support, pmf)
                rec = {"law": law, "n": n, "weights": kind, "neff": neff, "reps": args.reps,
                       "mean_sup_dev": devs.mean(), "q95_sup_dev": np.quantile(devs, 0.95)}
                for eta in etas:
                    b_ceil = grid_term(neff, eta)
                    b_best, kbest = grid_term_best(neff, eta)
                    b_old = math.sqrt(math.log(2 / eta) / (2 * neff))
                    rec[f"bound_Kceil_eta{eta}"] = b_ceil
                    rec[f"exceed_Kceil_eta{eta}"] = float(np.mean(devs > b_ceil))
                    rec[f"bound_Kbest_eta{eta}"] = b_best
                    rec[f"exceed_Kbest_eta{eta}"] = float(np.mean(devs > b_best))
                    rec[f"exceed_old_eta{eta}"] = float(np.mean(devs > b_old))
                rows.append(rec)
                print(f"{law:10s} n={n:4d} {kind:17s} neff={neff:7.1f} mean sup={devs.mean():.4f} "
                      + " ".join(f"ex(K*,{e})={rec[f'exceed_Kbest_eta{e}']:.4f}" for e in etas)
                      + " | old: " + " ".join(f"{rec[f'exceed_old_eta{e}']:.4f}" for e in etas))
    df = pd.DataFrame(rows)
    df.to_csv(out / "grid_lemma_summary.csv", index=False)
    with open(out / "grid_lemma_summary.txt", "w", encoding="utf-8") as f:
        f.write("Uniform concentration lemma check. Exceedance frequencies must be <= eta for the grid bound.\n")
        f.write("The 'old' columns use sqrt(log(2/eta)/(2 n_eff)) with no uniformity factor; they are reported only to show "
                "whether the old displayed term fails as a uniform bound.\n\n")
        f.write(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        f.write("\n")


if __name__ == "__main__":
    main()
