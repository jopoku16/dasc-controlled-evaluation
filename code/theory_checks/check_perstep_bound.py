"""
Numerical check of the per-step coverage bound (Theorem on approximate coverage at the
adaptive level) against the EXACT conditional coverage P(Y_t in C_t | history), which is
available in closed form in the simulated models below.

Models
  A  exogenous regime path: features computed from an independent driver series, residuals
     R_t = sigma(S_t) |eps_t| with eps_t iid N(0,1).  The idealised assumption (conditional
     independence of residuals given the feature path, residual law Lipschitz in the feature)
     holds exactly.  The theorem predicts a failure frequency at most eta.
  B1 endogenous features (computed from the observed series itself), oracle-mean forecaster,
     AR coefficient 0.95: the Lipschitz part holds exactly, conditional independence fails.
  B2 as B1 with a sharp bandwidth (h = 0.15), so n_eff is small.
  B3 as B1 with AR coefficient 0 (weak serial dependence).
  B4 as B1 with an atom at zero in the residual law (ties at the quantile).
  B5 as B1 but with the lag-one forecaster: the residual law depends on more than S_t, so
     the Lipschitz assumption itself fails (reported as a violated-assumption case).

For each model and each eta the script reports the frequency over steps of
  cov_t < 1 - alpha_t - L dbar_t - eps_t(eta)           (lower-bound failure)
  cov_t > 1 - alpha_t + L dbar_t + eps_t(eta) + mass_t  (upper-bound failure)
with eps_t(eta) the grid term of the concentration lemma, and also with the old displayed
term sqrt(log(2/eta)/(2 n_eff)) (which has no uniformity factor) for comparison.

Run:  python -X utf8 check_perstep_bound.py [--seeds 20] [--out outputs]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm

from dasc_causal import all_features, dasc_step, grid_term, spectral_feature, update_alpha

ALPHA, GAMMA, WINDOW, LAM, MMIN, L_WIN = 0.10, 0.015, 360, 0.45, 80, 64
START, N = 220, 1400
U = np.array([1.0, 0, 0, 0, 0, 0])  # direction in feature space that drives the scale


def half_normal_cdf(r, sigma):
    r = np.asarray(r, dtype=float)
    return np.where(r >= 0, erf(r / (sigma * math.sqrt(2.0))), 0.0)


def ks_half_normal(s1, s2):
    """Exact Kolmogorov distance between |N(0,s1^2)| and |N(0,s2^2)| (vectorised)."""
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    lo = np.minimum(s1, s2)
    hi = np.maximum(s1, s2)
    c = hi / lo
    out = np.zeros_like(c)
    m = c > 1 + 1e-12
    cm = c[m]
    xs = np.sqrt(2.0 * np.log(cm) / (cm ** 2 - 1.0))  # argmax of G(c x) - G(x)
    out[m] = erf(cm * xs / math.sqrt(2.0)) - erf(xs / math.sqrt(2.0))
    return out


def sigma_of(S, sigma0, kappa):
    return sigma0 + kappa * (S @ U)


def driver_stream(n, rng):
    """Regime-switching driver, same shape as the paper's synthetic stream."""
    t = np.arange(n)
    freq = np.piecewise(t.astype(float), [t < 450, (t >= 450) & (t < 900), t >= 900],
                        [0.035, 0.055, lambda z: 0.055 + 0.00006 * (z - 900)])
    amp = np.where((t // 180) % 2 == 0, 1.0, 1.7)
    level = np.where(t >= 900, 1.2, 0.0)
    signal = level + amp * np.sin(2 * np.pi * freq * t)
    return signal, t


def lipschitz_on_path(features, sigma, t_range, window):
    """Largest KS/feature-distance ratio over all (pool index, t) pairs actually used."""
    best = 0.0
    for t in t_range:
        pool = np.arange(max(1, t - window), t)
        d = np.linalg.norm(features[pool] - features[t], axis=1)
        ks = ks_half_normal(sigma[pool], sigma[t])
        ok = d > 1e-9
        if ok.any():
            best = max(best, float(np.max(ks[ok] / d[ok])))
    return best


def run_model(name, seed, eta_list, phi=0.95, h=0.55, atom=0.0, exogenous=False,
              lag_forecaster=False, sigma0=0.6, kappa=0.8, mode="clipped"):
    rng = np.random.default_rng(seed)
    signal, t = driver_stream(N, rng)
    eps = rng.normal(0, 1, N)
    if atom > 0:
        eps = eps * (rng.random(N) >= atom)
    if exogenous:
        v = signal + rng.normal(0, 0.45, N)        # driver observed, features from v
        features = all_features(v, L_WIN)
        sigma = sigma_of(features, sigma0, kappa)
        resid = sigma * np.abs(eps)
        mu = np.zeros(N)
    else:
        y = np.zeros(N)
        features = np.zeros((N, 6))
        sigma = np.zeros(N)
        mu = np.zeros(N)
        for i in range(N):
            features[i] = spectral_feature(y, i, L_WIN)  # reads y[i-64], ..., y[i-1] only
            sigma[i] = sigma_of(features[i], sigma0, kappa)
            m = phi * y[i - 1] + signal[i] if i > 0 else signal[i]
            y[i] = m + sigma[i] * eps[i]
            if lag_forecaster and i > 0:
                mu[i] = m - y[i - 1]  # location of the folded-normal residual law
        if lag_forecaster:
            resid = np.abs(y - np.r_[np.nan, y[:-1]])
            resid[0] = 0.0
        else:
            resid = sigma * np.abs(eps)
    t_range = range(START, N)
    L = lipschitz_on_path(features, sigma, t_range, WINDOW)
    if atom > 0:
        L = L * (1 - atom)
    rows = []
    a = ALPHA
    for tt in t_range:
        d = dasc_step(resid, features, tt, a, WINDOW, h, LAM, MMIN, L_WIN)
        q = d["q"]
        if lag_forecaster:
            cov = norm.cdf((q - mu[tt]) / sigma[tt]) - norm.cdf((-q - mu[tt]) / sigma[tt])
        else:
            cov = atom + (1 - atom) * float(half_normal_cdf(q, sigma[tt])) if np.isfinite(q) else (1.0 if q > 0 else 0.0)
        miss = int(resid[tt] > q) if np.isfinite(q) else int(q < 0)
        row = {"model": name, "seed": seed, "t": tt, "alpha_t": a, "cov": cov,
               "dbar": d["dbar"], "neff": d["neff"], "mass_q": d["mass_q"], "wmax": d["wmax"],
               "L": L, "bias": L * d["dbar"]}
        for eta in eta_list:
            e_grid = grid_term(d["neff"], eta)
            e_old = math.sqrt(math.log(2 / eta) / (2 * d["neff"]))
            row[f"lo_grid_{eta}"] = 1 - a - L * d["dbar"] - e_grid
            row[f"hi_grid_{eta}"] = 1 - a + L * d["dbar"] + e_grid + d["mass_q"]
            row[f"lo_old_{eta}"] = 1 - a - L * d["dbar"] - e_old
        rows.append(row)
        a, _ = update_alpha(a, miss, ALPHA, GAMMA, mode)
    return pd.DataFrame(rows)


def summarise(df, eta_list):
    out = []
    for (model), g in df.groupby("model"):
        rec = {"model": model, "steps": len(g), "mean_cov": g["cov"].mean(),
               "mean_1_minus_alpha_t": (1 - g["alpha_t"]).mean(), "L_path_max": g["L"].max(),
               "mean_bias_Ldbar": g["bias"].mean(), "median_neff": g["neff"].median(),
               "mean_mass_q": g["mass_q"].mean()}
        for eta in eta_list:
            rec[f"fail_lo_grid_eta{eta}"] = (g["cov"] < g[f"lo_grid_{eta}"]).mean()
            rec[f"fail_hi_grid_eta{eta}"] = (g["cov"] > g[f"hi_grid_{eta}"]).mean()
            rec[f"fail_lo_old_eta{eta}"] = (g["cov"] < g[f"lo_old_{eta}"]).mean()
            rec[f"mean_slack_lo_eta{eta}"] = (g["cov"] - g[f"lo_grid_{eta}"]).mean()
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / args.out
    out.mkdir(exist_ok=True)
    eta_list = [0.05, 0.10, 0.20]

    # sanity: KS(c) <= 0.4839 (c - 1) for the half-normal scale family
    cs = np.linspace(1.0001, 6, 4000)
    ratio = ks_half_normal(np.ones_like(cs), cs) / (cs - 1)
    print(f"half-normal scale family: max KS/(c-1) over c in (1,6] = {ratio.max():.4f} (2*phi(1) = {2*norm.pdf(1):.4f})")

    specs = [
        ("A_exogenous", dict(exogenous=True)),
        ("B1_endog_phi0.95", dict(phi=0.95)),
        ("B2_endog_sharp_h0.15", dict(phi=0.95, h=0.15)),
        ("B3_endog_phi0", dict(phi=0.0)),
        ("B4_endog_atom0.3", dict(phi=0.95, atom=0.3)),
        ("B5_endog_lag_forecaster", dict(phi=0.95, lag_forecaster=True)),
    ]
    frames = []
    for name, kw in specs:
        for seed in range(args.seeds):
            frames.append(run_model(name, seed, eta_list, **kw))
        print("done", name)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "perstep_bound_steps.csv.gz", index=False, compression="gzip")
    summ = summarise(df, eta_list)
    summ.to_csv(out / "perstep_bound_summary.csv", index=False)
    with open(out / "perstep_bound_summary.txt", "w", encoding="utf-8") as f:
        f.write("Per-step bound check: failure frequencies over all steps and seeds.\n")
        f.write(f"alpha={ALPHA}, gamma={GAMMA}, pool={WINDOW}, lambda={LAM}, m_min={MMIN}, "
                f"feature window={L_WIN}, T={N}, start={START}, seeds={args.seeds}\n")
        f.write("Model A satisfies the idealised assumption exactly; the theorem requires "
                "fail_lo_grid_eta <= eta there.  B models violate conditional independence; "
                "B5 also violates the Lipschitz assumption.\n\n")
        f.write(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        f.write("\n")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
