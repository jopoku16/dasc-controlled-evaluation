"""
Checks of the blocked concentration bound for fixed weights under beta-mixing, and the
arithmetic of the mixing remarks.

Statement checked (Proposition on blocked concentration): for a strictly stationary
beta-mixing sequence xi_1, ..., xi_n with |xi_i| <= 1, E xi_i = 0, fixed weights w_i >= 0
summing to one, consecutive blocks B_1, ..., B_{2 mu} of length a, block masses W_j,
    P(|sum_i w_i xi_i| >= u) <= 2 exp(-u^2 / (8 sum_{j odd} W_j^2))
                                + 2 exp(-u^2 / (8 sum_{j even} W_j^2)) + 2 (mu - 1) beta(a).

Part 1  Validity on an m-dependent process (beta(k) = 0 for k > m exactly), where the mixing
        penalty vanishes for a > m, with uniform and kernel-type weights.  Reports the
        empirical tail against the bound at several u.
Part 2  Arithmetic: the old remark's 4 floor(360/24) rho^23 with rho = exp(-1/10); the new
        bound's best achievable deviation u at confidence eta for n in {360, 2160, 20000}
        with equal weights, and with a DASC weight profile taken from the synthetic stream.
Part 3  Old lemma's claimed inequality compared with the true tail on a block-constant
        process (shows the gap-block step cannot be repaired by changing constants only).

Run:  python -X utf8 check_mixing_bound.py [--reps 20000] [--out outputs]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dasc_causal import all_features, dasc_step


def block_masses(w, a):
    n = len(w)
    nb = int(math.ceil(n / a))
    W = np.array([w[j * a:(j + 1) * a].sum() for j in range(nb)])
    return W


def blocked_bound(w, a, u, beta_a):
    W = block_masses(w, a)
    odd = W[0::2]
    even = W[1::2]
    mu = len(odd)
    term = 2 * math.exp(-u ** 2 / (8 * np.sum(odd ** 2)))
    if len(even):
        term += 2 * math.exp(-u ** 2 / (8 * np.sum(even ** 2)))
    return term + 2 * (mu - 1) * beta_a


def m_dependent_xi(n, m, rng, reps):
    """xi_i = 1{R_i <= r} - F(r), R_i = |Z_i + ... + Z_{i-m}| / sqrt(m+1), Z iid N(0,1), r = median."""
    Z = rng.normal(0, 1, (reps, n + m))
    S = np.zeros((reps, n))
    for k in range(m + 1):
        S += Z[:, k:k + n]
    R = np.abs(S) / math.sqrt(m + 1)
    r = math.sqrt(2) * 0.6744897501960817        # median of |N(0,1)| times... scale 1: median = 0.6745
    r = 0.6744897501960817
    return (R <= r).astype(float) - 0.5


def part1(rng, reps, lines, rows):
    n = 360
    for m in [3, 10]:
        xi = m_dependent_xi(n, m, rng, reps)
        for kind in ["uniform", "kernel"]:
            if kind == "uniform":
                w = np.ones(n) / n
            else:
                d = rng.random(n)
                w = np.exp(-d ** 2 / 0.3)
                w = w / w.sum()
            S = xi @ w
            a = m + 1                                   # gap > m so beta(a) = 0 exactly
            for u in [0.05, 0.1, 0.15, 0.2, 0.3]:
                emp = float(np.mean(np.abs(S) >= u))
                bnd = blocked_bound(w, a, u, 0.0)
                naive = 2 * math.exp(-2 * u ** 2 / np.sum(w ** 2))   # Hoeffding as if independent
                rows.append({"part": 1, "m": m, "weights": kind, "a": a, "u": u, "empirical": emp,
                             "blocked_bound": min(bnd, 1.0), "iid_hoeffding": min(naive, 1.0),
                             "neff": 1 / np.sum(w ** 2)})
                lines.append(f"[part 1] m={m} {kind:7s} a={a} u={u:.2f}: empirical={emp:.4f}, blocked bound={min(bnd,1):.4f}, "
                             f"iid Hoeffding (not valid here)={min(naive,1):.4f}")


def part2(lines, rows):
    rho = math.exp(-1 / 10)
    old = 4 * math.floor(360 / 24) * rho ** 23
    lines.append(f"[part 2] old remark arithmetic: 4*floor(360/24)*rho^23 with rho=e^(-1/10) = {old:.3f} (claimed below 1e-3)")
    lines.append(f"[part 2] rho^23 = {rho**23:.4f}; the remark would need rho <= {(1e-3/60)**(1/23):.3f}")
    rows.append({"part": 2, "item": "old_remark_value", "value": old})
    for n in [360, 2160, 20000]:
        w = np.ones(n) / n
        best = None
        for a in range(5, n // 4 + 1):
            W = block_masses(w, a)
            odd, even = W[0::2], W[1::2]
            mu = len(odd)
            pen = 2 * (mu - 1) * rho ** a
            for eta in [0.1]:
                if pen >= eta:
                    continue
                # solve 4 exp(-u^2/(8 nu)) = eta - pen
                nu = max(np.sum(odd ** 2), np.sum(even ** 2) if len(even) else 0)
                u = math.sqrt(8 * nu * math.log(4 / (eta - pen)))
                if best is None or u < best[0]:
                    best = (u, a, mu, pen)
        if best is None:
            lines.append(f"[part 2] n={n}: no block length makes the penalty below eta=0.1")
        else:
            u, a, mu, pen = best
            lines.append(f"[part 2] n={n}, equal weights, rho=e^(-1/10), eta=0.1: best block length a={a}, mu={mu} blocks per parity, "
                         f"penalty={pen:.4f}, deviation u={u:.3f} (compare alpha=0.1; iid term sqrt(log(2/eta)/(2n))={math.sqrt(math.log(20)/(2*n)):.4f})")
            rows.append({"part": 2, "item": f"best_u_n{n}", "value": u, "a": a, "mu": mu, "penalty": pen})
    # DASC weight profile from the synthetic stream
    rng = np.random.default_rng(7)
    t = np.arange(1400)
    freq = np.piecewise(t.astype(float), [t < 450, (t >= 450) & (t < 900), t >= 900],
                        [0.035, 0.055, lambda z: 0.055 + 0.00006 * (z - 900)])
    y = np.where((t // 180) % 2 == 0, 1.0, 1.7) * np.sin(2 * np.pi * freq * t) + np.where(t >= 900, 1.2, 0.0) + rng.normal(0, np.where(t >= 900, 0.75, 0.45))
    resid = np.abs(y - np.r_[np.nan, y[:-1]]); resid[0] = 0
    feats = all_features(y, 64)
    for tt in [600, 1000, 1300]:
        current = feats[tt]
        pool = np.arange(tt - 360, tt)
        d = np.linalg.norm(feats[pool] - current, axis=1)
        w = np.exp(-d ** 2 / 0.55 ** 2); w = w / w.sum()
        neff = 1 / np.sum(w ** 2)
        best = None
        for a in range(5, 91):
            W = block_masses(w, a)
            odd, even = W[0::2], W[1::2]
            mu = len(odd)
            pen = 2 * (mu - 1) * rho ** a
            if pen >= 0.1:
                continue
            nu = max(np.sum(odd ** 2), np.sum(even ** 2))
            u = math.sqrt(8 * nu * math.log(4 / (0.1 - pen)))
            if best is None or u < best[0]:
                best = (u, a, mu, pen)
        lines.append(f"[part 2] DASC weights at t={tt} (n=360, neff={neff:.1f}): best a={best[1]}, mu={best[2]}, penalty={best[3]:.4f}, deviation u={best[0]:.3f}")
        rows.append({"part": 2, "item": f"dasc_weights_t{tt}", "value": best[0], "a": best[1], "mu": best[2], "penalty": best[3], "neff": neff})


def part3(rng, reps, lines, rows):
    """Block-constant process: xi_i = Z_{floor(i/g)} with Z iid Rademacher, g = 64, n = 6400."""
    g, n = 64, 6400
    Z = rng.choice([-1.0, 1.0], size=(reps, n // g))
    S = Z.mean(axis=1)                      # equals sum_i w_i xi_i with uniform weights
    for u in [0.1, 0.15, 0.2]:
        emp = float(np.mean(np.abs(S) >= u))
        old_claim_conc = 4 * math.exp(-u ** 2 * n / 8)   # old lemma's exponential term with n_eff = n
        true_hoeff = 2 * math.exp(-2 * u ** 2 * n / g)   # Hoeffding on the n/g independent block means
        lines.append(f"[part 3] block-constant process g={g}, n={n}, u={u:.2f}: empirical tail={emp:.4f}, "
                     f"old lemma exponential term={old_claim_conc:.2e}, correct block Hoeffding={true_hoeff:.4f}")
        rows.append({"part": 3, "u": u, "empirical": emp, "old_exponential_term": old_claim_conc, "block_hoeffding": true_hoeff})
    lines.append("[part 3] note: this process has beta(k) = (1 - k/g)^+ under a uniformly random phase, which is not below "
                 "c0 rho^k for any rho with rho^g small, so it is NOT a counterexample to the old lemma's statement; it shows "
                 "only that the exponential term with n_eff = n cannot come from the gap blocks, which is the step the old proof asserted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / args.out
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    lines, rows = ["Mixing bound checks"], []
    part1(rng, args.reps, lines, rows)
    part2(lines, rows)
    part3(rng, args.reps, lines, rows)
    pd.DataFrame(rows).to_csv(out / "mixing_bound_checks.csv", index=False)
    text = "\n".join(lines) + "\n"
    (out / "mixing_bound_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
