"""
Checks of the long-run statements.

Part 1  Exact identity for the clipped update on the PUBLISHED DASC alpha_t paths
        (results/real_*_results.csv of the original project, read through --repo).
        The identity is
          mean(E_t) - alpha = (alpha_1 - alpha_{T+1})/(gamma T) + (1/(gamma T)) sum eps_t,
        with eps_t the projection residual, eps_t in [-gamma alpha, 0] at the cap and in
        [0, gamma (1-alpha)] at the floor.  The script recomputes alpha_{t+1} from alpha_t
        and E_t, counts the boundary steps, and compares with the old theorem's bound.
Part 2  Synthetic stream: clipped and free updates; exact identities; the range of alpha_t
        in the free update; the deterministic counterexample to the old theorem (constant
        stream, every interval covers, alpha_t pinned at the cap).
Part 3  Weather stream re-run with the causal implementation: as published (interval
        centred on y[t-1], residual quantiles from the 24-hour-lag residuals) and re-centred
        (y[t-24]); clipped and free updates.  Reports mean miss, cap/floor frequencies, the
        identity, and the free-update range of alpha_t.

Run:  python -X utf8 check_longrun.py [--repo PATH_TO_ORIGINAL_PROJECT] [--out outputs]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dasc_causal import all_features, dasc_step, update_alpha

ALPHA, GAMMA, AMIN, AMAX = 0.10, 0.015, 0.01, 0.35


def identity_report(alpha_path, miss, alpha=ALPHA, gamma=GAMMA, amin=AMIN, amax=AMAX, clipped=True):
    """alpha_path[k] is the level used at step k (k = 0..T-1); miss[k] in {0,1}."""
    a = np.asarray(alpha_path, dtype=float)
    e = np.asarray(miss, dtype=float)
    T = len(a)
    u = a + gamma * (alpha - e)                      # pre-projection values
    a_next = np.clip(u, amin, amax) if clipped else u
    recur_err = np.max(np.abs(a_next[:-1] - a[1:])) if T > 1 else 0.0
    eps = a_next - u
    n_hi = int(np.sum(u > amax + 1e-12))
    n_lo = int(np.sum(u < amin - 1e-12))
    aT1 = a_next[-1]
    lhs = e.mean() - alpha
    rhs = (a[0] - aT1) / (gamma * T) + eps.sum() / (gamma * T)
    return {
        "T": T, "mean_miss": e.mean(), "deviation": lhs, "identity_rhs": rhs,
        "identity_gap": lhs - rhs, "recursion_max_err": recur_err,
        "alpha_1": a[0], "alpha_T1": aT1, "n_hi": n_hi, "n_lo": n_lo,
        "frac_hi": n_hi / T, "frac_lo": n_lo / T,
        "telescoping_term": (a[0] - aT1) / (gamma * T),
        "projection_term": eps.sum() / (gamma * T),
        "old_theorem_bound": (amax - amin) / (gamma * T),
        "new_lower_bound_dev": -(amax - amin) / (gamma * T) - alpha * n_hi / T,
        "new_upper_bound_dev": (amax - amin) / (gamma * T) + (1 - alpha) * n_lo / T,
        "free_range_bound": (max(a[0], 1 - a[0]) + gamma) / (gamma * T),
        "alpha_min_seen": a.min(), "alpha_max_seen": a.max(),
    }


def part1(repo, lines):
    rows = []
    for name in ["real_weather_results.csv", "real_power_results.csv", "real_finance_results.csv"]:
        p = repo / "results" / name
        if not p.exists():
            lines.append(f"[part 1] {name}: not found, skipped")
            continue
        df = pd.read_csv(p, usecols=["t", "method", "miss", "alpha_t"])
        d = df[df["method"] == "DASC"].sort_values("t")
        rep = identity_report(d["alpha_t"].to_numpy(), d["miss"].to_numpy())
        rep["dataset"] = name
        rows.append(rep)
        lines.append(f"[part 1] {name}: T={rep['T']}, mean miss={rep['mean_miss']:.4f}, "
                     f"deviation from alpha={rep['deviation']:+.4f}, identity gap={rep['identity_gap']:.2e}, "
                     f"recursion max err={rep['recursion_max_err']:.2e}, cap steps={rep['frac_hi']:.3f}, "
                     f"floor steps={rep['frac_lo']:.3f}, telescoping={rep['telescoping_term']:+.5f}, "
                     f"projection={rep['projection_term']:+.5f}, old bound={rep['old_theorem_bound']:.5f}, "
                     f"new bounds on deviation=[{rep['new_lower_bound_dev']:+.4f}, {rep['new_upper_bound_dev']:+.4f}]")
    return pd.DataFrame(rows)


def simulate_stream(n=1400, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    freq = np.piecewise(t.astype(float), [t < 450, (t >= 450) & (t < 900), t >= 900],
                        [0.035, 0.055, lambda z: 0.055 + 0.00006 * (z - 900)])
    amp = np.where((t // 180) % 2 == 0, 1.0, 1.7)
    level = np.where(t >= 900, 1.2, 0.0)
    sigma = np.where(t >= 900, 0.75, 0.45)
    return level + amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, sigma)


def run_stream(y, resid, features, start, mode, window, h, lam, m_min, centre_offset):
    a = ALPHA
    alphas, misses, widths = [], [], []
    for t in range(start, len(y)):
        d = dasc_step(resid, features, t, a, window, h, lam, m_min, 64)
        q = d["q"]
        centre = y[t - centre_offset]
        if np.isinf(q):
            miss = 0 if q > 0 else 1
        else:
            miss = int(y[t] < centre - q or y[t] > centre + q)
        alphas.append(a)
        misses.append(miss)
        widths.append(2 * q if np.isfinite(q) else np.nan)
        a, _ = update_alpha(a, miss, ALPHA, GAMMA, mode, AMIN, AMAX)
    return np.array(alphas), np.array(misses), np.array(widths)


def part2(lines):
    rows = []
    for seed in range(5):
        y = simulate_stream(seed=seed)
        resid = np.abs(y - np.r_[np.nan, y[:-1]])
        resid[0] = 0.0
        feats = all_features(y, 64)
        for mode in ["clipped", "free"]:
            a, e, w = run_stream(y, resid, feats, 220, mode, 360, 0.55, 0.45, 80, 1)
            rep = identity_report(a, e, clipped=(mode == "clipped"))
            rep.update({"case": "synthetic", "seed": seed, "mode": mode, "mean_width": np.nanmean(w)})
            rows.append(rep)
            lines.append(f"[part 2] synthetic seed {seed} {mode}: mean miss={rep['mean_miss']:.4f}, "
                         f"identity gap={rep['identity_gap']:.2e}, |dev|={abs(rep['deviation']):.4f}, "
                         f"cap={rep['frac_hi']:.3f}, floor={rep['frac_lo']:.3f}, alpha range=[{rep['alpha_min_seen']:.3f},{rep['alpha_max_seen']:.3f}], "
                         f"free-range bound={rep['free_range_bound']:.4f}")
    # deterministic counterexample: constant stream, every interval covers
    T = 20000
    e = np.zeros(T)
    a = np.empty(T)
    a[0] = ALPHA
    for k in range(1, T):
        a[k], _ = update_alpha(a[k - 1], 0, ALPHA, GAMMA, "clipped", AMIN, AMAX)
    rep = identity_report(a, e)
    rep.update({"case": "constant_stream_counterexample", "seed": -1, "mode": "clipped", "mean_width": 0.0})
    rows.append(rep)
    lines.append(f"[part 2] counterexample (constant stream, E_t = 0 always): T={T}, mean miss={rep['mean_miss']:.4f}, "
                 f"|deviation|={abs(rep['deviation']):.4f}, old theorem bound={rep['old_theorem_bound']:.5f}, "
                 f"cap steps={rep['frac_hi']:.4f}, identity gap={rep['identity_gap']:.2e}")
    return pd.DataFrame(rows)


def part3(repo, lines):
    p = repo / "data" / "real" / "open_meteo_dallas" / "open_meteo_dallas_hourly_temperature.csv"
    if not p.exists():
        lines.append("[part 3] weather file not found, skipped")
        return pd.DataFrame()
    df = pd.read_csv(p).iloc[:24000]
    y = df["temperature_2m"].to_numpy(dtype=float)
    pred = np.full(len(y), np.nan)
    pred[24:] = y[:-24]
    resid = np.abs(y - pred)
    resid[:24] = 0.0
    feats = all_features(y, 24 * 14)
    rows = []
    for centre_name, off in [("as_published_centre_lag1", 1), ("recentred_lag24", 24)]:
        for mode in ["clipped", "free"]:
            a, e, w = run_stream(y, resid, feats, 1440, mode, 24 * 90, 0.55, 0.45, 24 * 21, off)
            rep = identity_report(a, e, clipped=(mode == "clipped"))
            rep.update({"case": f"weather_{centre_name}", "mode": mode, "mean_width": np.nanmean(w),
                        "coverage": 1 - e.mean()})
            rows.append(rep)
            lines.append(f"[part 3] weather {centre_name} {mode}: coverage={1-e.mean():.4f}, width={np.nanmean(w):.4f}, "
                         f"identity gap={rep['identity_gap']:.2e}, cap={rep['frac_hi']:.3f}, floor={rep['frac_lo']:.3f}, "
                         f"alpha range=[{rep['alpha_min_seen']:.3f},{rep['alpha_max_seen']:.3f}], "
                         f"telescoping={rep['telescoping_term']:+.5f}, projection={rep['projection_term']:+.5f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--repo", default=str(here.parents[1]),
                    help="root of a checkout of the earlier project (read only); part 1 needs its results/real_*_results.csv "
                         "and part 3 reads data/real/open_meteo_dallas under this root; default: the root of this repository")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    repo = Path(args.repo)
    out = here / args.out
    out.mkdir(exist_ok=True)
    lines = [f"Long-run checks. alpha={ALPHA}, gamma={GAMMA}, clip=[{AMIN},{AMAX}]"]
    p1 = part1(repo, lines)
    p2 = part2(lines)
    p3 = part3(repo, lines)
    if len(p1):
        p1.to_csv(out / "longrun_published_paths.csv", index=False)
    p2.to_csv(out / "longrun_synthetic.csv", index=False)
    if len(p3):
        p3.to_csv(out / "longrun_weather_rerun.csv", index=False)
    text = "\n".join(lines) + "\n"
    (out / "longrun_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
