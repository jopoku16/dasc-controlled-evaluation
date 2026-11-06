"""
lead4p_checks.py: numerical checks of the statements of the averaged-recursion account (lead 4).

Nothing in the existing code or results is modified: lead4_model is imported read-only, the
existing lead4_*.csv files are read, and every new output carries the prefix lead4p_.
One process only.

Outputs (all in results/):
  lead4p_ode_bounds.csv      36 cells: orbit endpoints, two-sided bound on the deviation, antisymmetry,
                              Gibbs-Candes square-root bound, F1 error bound (rows follow lead4_cells.csv)
  lead4p_match_closed.csv    weighted quantile: closed forms versus lead4_match.csv (21 rows with w < 1)
  lead4p_xstar_check.csv     x* re-computed by an independent ODE solver; x*(1-w) for w up to 0.98
  lead4p_dx_monotone.csv     d(x) on a log grid for three ratios (monotonicity evidence)
  lead4p_attraction.csv      iterates of the one-period ODE map from far-away starts
  lead4p_gap_scaling.csv     exact lattice chain versus ODE at fixed x for halving gamma
  lead4p_thm5_constants.csv  explicit constants of the chain-to-ODE theorem for one cell
  lead4p_summary.json        scalars of the checks
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(HERE), "results")
import lead4_model as M  # noqa: E402  (read-only import)

A0 = 0.1
RHOS = [1.5, 2.0, 3.0]
_POOLS = {}
SUM = {}


def pool_for(rho):
    if rho not in _POOLS:
        _POOLS[rho] = M.Pool((1.0, rho), (0.5, 0.5), A0).build_tables(40001)
    return _POOLS[rho]


# ---------------------------------------------------------------------------
# independent ODE solver (rescaled time tau in [0,1] per block, rate x)
# ---------------------------------------------------------------------------

def flow(pool, r, a_start, x):
    f = lambda tau, a: [x * (A0 - float(pool.p(r, a[0])))]
    sol = solve_ivp(f, (0.0, 1.0), [a_start], method="LSODA", rtol=1e-11, atol=1e-13)
    return float(sol.y[0, -1])


def psi(pool, a_start, x):
    return flow(pool, 1, flow(pool, 0, a_start, x), x)


def orbit(pool, x):
    """Unique fixed point A* of the one-period map (bracketed by the equilibria), and B*."""
    a1, a2 = pool.fixed_points()[0], pool.fixed_points()[1]
    g = lambda A: psi(pool, A, x) - A
    A = brentq(g, a2, a1, xtol=1e-13, rtol=1e-13)
    B = flow(pool, 0, A, x)
    return A, B


def ode_miss(pool, x):
    A, B = orbit(pool, x)
    d = (B - A) / x
    return A0 - d, A0 + d, A, B


def s_slope(pool, r, a):
    return float(pool.slope(r, a)[0])


def p1_second(pool, a):
    """p_1''(a) = 4 theta q ell / ((1+ell)^3 f_1(q)), theta = 1 - rho^-2, ell = f_2/f_1."""
    rho = float(pool.sig[1])
    theta = 1.0 - rho ** -2
    q = float(pool.q_of_alpha(a)[0])
    f1 = float(pool.dens(0, q))
    ell = float(pool.dens(1, q)) / f1
    return 4.0 * theta * q * ell / ((1.0 + ell) ** 3 * f1)


# ---------------------------------------------------------------------------
# C1: block identity on a fresh trajectory (algebra) and on the stored CSVs
# ---------------------------------------------------------------------------

def check_block_identity():
    out = {}
    rng = np.random.default_rng(20261102)
    for a0, gamma, m, rho in [(0.1, 0.02, 50, 2.0), (0.137, 0.013, 40, 3.0)]:
        pool = M.Pool((1.0, rho), (0.5, 0.5), a0).build_tables(20001)
        n_periods = 300
        alpha = a0
        worst_block = 0.0
        sum_starts = []
        tot_e = np.zeros(2)
        for per in range(n_periods):
            for r in (0, 1):
                a_start = alpha
                s = 0.0
                for t in range(m):
                    p = float(pool.p(r, alpha))
                    e = 1.0 if rng.random() < p else 0.0
                    s += e - a0
                    tot_e[r] += e
                    alpha = alpha + gamma * (a0 - e)
                worst_block = max(worst_block, abs(s - (a_start - alpha) / gamma))
                if r == 0:
                    sum_starts.append(a_start)
        # Cesaro identity: d1 + d2 = (alpha_first - alpha_last_start)/(gamma m J)
        J = n_periods
        d1 = tot_e[0] / (J * m) - a0
        d2 = tot_e[1] / (J * m) - a0
        lhs = d1 + d2
        rhs = (sum_starts[0] - alpha) / (gamma * m * J)
        out[f"a0={a0},gamma={gamma},m={m},rho={rho}"] = {
            "max_block_residual": worst_block, "cesaro_lhs": lhs, "cesaro_rhs": rhs,
            "cesaro_abs_diff": abs(lhs - rhs), "cesaro_bound": (1 + 2 * gamma) / (gamma * m * J)}
    SUM["C1_fresh_trajectories"] = out
    df = pd.read_csv(os.path.join(RES, "lead4_cells_with_checks.csv"))
    SUM["C1_chain_I2_maxdev"] = float((df.F4_miss1 - df.I2_chain).abs().max())
    SUM["C1_mc_I2_maxdev"] = float((df.mc_miss1 - df.I2_mc).abs().max())
    sym = ((df.F4_miss1 + df.F4_miss2) / 2 - A0).abs()
    SUM["C1_chain_symmetry_maxdev"] = float(sym.max())
    SUM["C1_chain_symmetry_row"] = int(sym.values.argmax() + 1)
    symmc = ((df.mc_miss1 + df.mc_miss2) / 2 - A0).abs()
    z = symmc / (0.5 * np.sqrt(df.mc_se1 ** 2 + df.mc_se2 ** 2))
    SUM["C1_mc_symmetry_maxdev"] = float(symmc.max())
    SUM["C1_mc_symmetry_maxz"] = float(z.max())
    SUM["C1_mc_symmetry_row_z"] = int(z.values.argmax() + 1)
    # rolling-window stored run
    rw = pd.read_csv(os.path.join(RES, "lead4_rolling_cells.csv"))
    symr = ((rw.roll_miss1 + rw.roll_miss2) / 2 - A0).abs()
    zr = symr / (0.5 * np.sqrt(rw.roll_se1 ** 2 + rw.roll_se2 ** 2))
    SUM["C1_roll_symmetry_maxdev"] = float(symr.max())
    SUM["C1_roll_symmetry_maxz"] = float(zr.max())
    SUM["C1_roll_symmetry_row_z"] = int(zr.values.argmax() + 1)


# ---------------------------------------------------------------------------
# C2: scaling in x
# ---------------------------------------------------------------------------

def check_scaling():
    rows = []
    for rho, x, pairs in [(2.0, 3.0, [(0.02, 150), (0.06, 50), (0.01, 300)]),
                          (3.0, 0.75, [(0.015, 50), (0.005, 150), (0.0025, 300)]),
                          (1.5, 9.0, [(0.18, 50), (0.06, 150), (0.02, 450)])]:
        pool = pool_for(rho)
        vals = []
        for g, m in pairs:
            f2 = M.f2_ode(pool, g, m)
            vals.append(f2[0])
            rows.append({"rho": rho, "x": x, "gamma": g, "m": m, "F2_miss1": f2[0]})
        SUM[f"C2_spread_rho{rho}_x{x}"] = float(max(vals) - min(vals))


# ---------------------------------------------------------------------------
# C3: ODE theory on the 36 cells
# ---------------------------------------------------------------------------

def check_ode_cells():
    df = pd.read_csv(os.path.join(RES, "lead4_cells.csv"))
    rows = []
    for k, r in df.iterrows():
        rho, x, gamma, m = r.rho, r.gamma_m, r.gamma, int(r.m)
        pool = pool_for(rho)
        a1, a2 = r.alpha1_star, r.alpha2_star
        A, B = orbit(pool, x)
        D = B - A
        dev = D / x
        s1_a2 = s_slope(pool, 0, a2)
        s2_a1 = s_slope(pool, 1, a1)
        up = (a1 - a2) / x
        lo = (a1 - a2) * (1 - np.exp(-x * s1_a2) - np.exp(-x * s2_a1)) / x
        dev_F4 = max(abs(r.F4_miss1 - A0), abs(r.F4_miss2 - A0))
        gc = np.sqrt(2.0 * (1.0 + gamma) * (a1 - a2) / x + gamma)
        # F1 error bound (curvature on the orbit interval)
        q0, p0, s0 = pool.zero_order()
        e1, e2 = np.exp(-x * s0[0]), np.exp(-x * s0[1])
        grid = np.linspace(A, B, 41)
        Mc = max(abs(p1_second(pool, a)) for a in grid)
        Ubar = max(abs(A - A0), abs(B - A0))
        Rbar = 0.5 * Mc * Ubar ** 2
        VA = Rbar * (e2 * (1 - e1) / s0[0] + (1 - e2) / s0[1]) / (1 - e1 * e2)
        dbound = (1 - e1) * (VA + Rbar / s0[0])
        f1_gap = abs(r.F1_miss1 - (A0 - dev))      # F1 versus the ODE solution computed here
        rows.append({
            "row": k + 1, "gamma": gamma, "m": m, "rho": rho, "x": x, "a1_star": a1, "a2_star": a2,
            "A_star": A, "B_star": B, "alpha0_in_orbit": bool(A <= A0 <= B), "orbit_in_J": bool(a2 < A and B < a1),
            "D_over_x_min_a0": D / (x * min(A0, 1 - A0)),
            "dev_ode": dev, "dev_F2_csv": abs(r.F2_miss1 - A0), "dev_F4": dev_F4,
            "bound_up": up, "bound_lo": lo, "ratio_up": dev / up, "ratio_lo": dev / lo if lo > 0 else np.nan,
            "s1_at_a2star": s1_a2, "s2_at_a1star": s2_a1,
            "antisym_F2": abs((r.F2_miss1 + r.F2_miss2) / 2 - A0),
            "gc_sqrt_bound": gc, "gc_valid": bool(dev_F4 <= gc),
            "curv_M": Mc, "Ubar": Ubar, "F1_err_vs_ode": f1_gap, "F1_err_bound": dbound / x,
            "F1_bound_valid": bool(f1_gap <= dbound / x + 1e-12),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "lead4p_ode_bounds.csv"), index=False)
    SUM["C3_orbit_in_J_all"] = bool(out.orbit_in_J.all())
    SUM["C3_alpha0_in_orbit_all"] = bool(out.alpha0_in_orbit.all())
    SUM["C3_Dx_over_min_max"] = float(out.D_over_x_min_a0.max())
    SUM["C3_ratio_up_max"] = float(out.ratio_up.max())
    SUM["C3_ratio_up_max_row"] = int(out.ratio_up.values.argmax() + 1)
    big = out[out.x >= 5]
    SUM["C3_ratio_up_min_x_ge_5"] = float(big.ratio_up.min())
    SUM["C3_ratio_up_max_x_ge_5"] = float(big.ratio_up.max())
    SUM["C3_lower_bound_valid_all"] = bool((out.dev_ode >= out.bound_lo - 1e-12).all())
    SUM["C3_lower_bound_positive_cells"] = int((out.bound_lo > 0).sum())
    pos = out[out.bound_lo > 0]
    SUM["C3_lower_bound_min_ratio"] = float(pos.ratio_lo.min()) if len(pos) else None
    SUM["C3_F2_vs_ode_maxdiff"] = float((out.dev_ode - out.dev_F2_csv).abs().max())
    SUM["C3_ode_antisym_maxdev"] = float(out.antisym_F2.max())
    SUM["C3_gc_valid_all"] = bool(out.gc_valid.all())
    SUM["C3_gc_ratio_min"] = float((out.dev_F4 / out.gc_sqrt_bound).min())
    SUM["C3_gc_ratio_max"] = float((out.dev_F4 / out.gc_sqrt_bound).max())
    SUM["C3_F1_bound_valid_all"] = bool(out.F1_bound_valid.all())
    SUM["C3_F1_bound_valid_count"] = int(out.F1_bound_valid.sum())
    SUM["C3_F1_bound_over_err_median"] = float((out.F1_err_bound / out.F1_err_vs_ode).median())
    SUM["C3_F1_err_vs_ode_max"] = float(out.F1_err_vs_ode.max())
    return out


def check_attraction():
    rows = []
    for rho, x in [(2.0, 2.5), (3.0, 7.5), (1.5, 0.5)]:
        pool = pool_for(rho)
        A, B = orbit(pool, x)
        for start in [-0.5, 0.0, 0.001, 0.5, 0.999, 1.5]:
            a = start
            traj = []
            for n in range(1, 13):
                a = psi(pool, a, x)
                traj.append(abs(a - A))
            rows.append({"rho": rho, "x": x, "start": start, "A_star": A,
                         "err_after_1": traj[0], "err_after_3": traj[2], "err_after_6": traj[5], "err_after_12": traj[11],
                         "monotone": bool(all(traj[i + 1] <= traj[i] + 1e-9 for i in range(len(traj) - 1)))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lead4p_attraction.csv"), index=False)
    SUM["C3_attraction_all_monotone"] = bool(df.monotone.all())
    SUM["C3_attraction_max_err_after_12"] = float(df.err_after_12.max())


def check_dx_monotone():
    xs = np.exp(np.linspace(np.log(0.02), np.log(80.0), 45))
    rows = []
    for rho in RHOS:
        pool = pool_for(rho)
        a1, a2 = pool.fixed_points()
        g0 = A0 - float(pool.p(0, A0))
        prev = None
        mono = True
        for x in xs:
            m1, m2, A, B = ode_miss(pool, x)
            d = A0 - m1
            if prev is not None and d > prev + 1e-12:
                mono = False
            prev = d
            rows.append({"rho": rho, "x": x, "d": d, "x_times_d": x * d, "a1_minus_a2": a1 - a2, "g_frozen": g0})
        SUM[f"C4_d_monotone_rho{rho}"] = mono
        SUM[f"C4_d_at_smallest_x_over_g_rho{rho}"] = float(rows[-len(xs)]["d"] / g0)
        SUM[f"C4_xd_at_largest_x_over_gap_rho{rho}"] = float(rows[-1]["x_times_d"] / (a1 - a2))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "lead4p_dx_monotone.csv"), index=False)


# ---------------------------------------------------------------------------
# C4: weighted quantile closed forms
# ---------------------------------------------------------------------------

def delta_fn(pool):
    return lambda a: a - float(pool.p_exact(0, a)[0])


def check_match_closed():
    df = pd.read_csv(os.path.join(RES, "lead4_match.csv"))
    ck = pd.read_csv(os.path.join(RES, "lead4_match_chain_check.csv"))
    rows = []
    for k, r in df.iterrows():
        if r.w >= 1.0:
            continue
        pool = pool_for(r.rho)
        D = delta_fn(pool)
        kap = 2 * r.w - 1
        # a_1 solves a = A0 + kap D(a); a_2 solves a = A0 - kap D(a)
        a1 = brentq(lambda a: a - kap * D(a) - A0, 1e-12, 1 - 1e-12, xtol=1e-15)
        a2 = brentq(lambda a: a + kap * D(a) - A0, 1e-12, 1 - 1e-12, xtol=1e-15)
        m1 = A0 - 2 * (1 - r.w) * D(a1)
        m2 = A0 + 2 * (1 - r.w) * D(a2)
        c = (1 - r.w) * (D(a2) - D(a1))
        dW = (1 - r.w) * (D(a1) + D(a2))
        sub = ck[(ck.rho == r.rho) & (ck.w == r.w)]
        excess = (sub.max_abs_err - sub.predicted_err_abs_cW) if len(sub) else pd.Series([np.nan])
        rows.append({"match_row": k + 1, "rho": r.rho, "w": r.w, "a1": a1, "a2": a2,
                     "res_miss1": abs(m1 - r.W_miss1), "res_miss2": abs(m2 - r.W_miss2),
                     "res_c": abs(c - r.W_common_mode), "res_dW": abs(dW - r.W_gap / 2),
                     "c": c, "d_W": dW, "x_star_csv": r.x_star_ode, "x_star_times_1mw": r.x_star_ode * (1 - r.w),
                     "chain_excess_min": float(excess.min()), "chain_excess_max": float(excess.max())})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "lead4p_match_closed.csv"), index=False)
    SUM["C5_closed_form_max_residual"] = float(out[["res_miss1", "res_miss2", "res_c", "res_dW"]].max().max())
    SUM["C5_c_all_negative"] = bool((out.c < 0).all())
    SUM["C5_chain_excess_min"] = float(out.chain_excess_min.min())
    SUM["C5_chain_excess_max"] = float(out.chain_excess_max.max())
    # local slope at w = 1/2 and the crossing condition
    for rho in RHOS:
        pool = pool_for(rho)
        q0, p0, s0 = pool.zero_order()
        g = A0 - p0[0]
        D = delta_fn(pool)
        eps = 1e-4
        w = 0.5 + eps
        kap = 2 * w - 1
        a1 = brentq(lambda a: a - kap * D(a) - A0, 1e-12, 1 - 1e-12, xtol=1e-15)
        a2 = brentq(lambda a: a + kap * D(a) - A0, 1e-12, 1 - 1e-12, xtol=1e-15)
        c_num = (1 - w) * (D(a2) - D(a1)) / eps
        SUM[f"C5_cprime_half_formula_rho{rho}"] = float(-2 * g * (1 - s0[0]))
        SUM[f"C5_cprime_half_numeric_rho{rho}"] = float(c_num)
        SUM[f"C5_s1_alpha0_rho{rho}"] = float(s0[0])
        SUM[f"C5_g_rho{rho}"] = float(g)
        qs = norm.ppf(1 - A0 / 2)
        qc2 = 2 * np.log(rho) / (1 - rho ** -2)
        SUM[f"C5_qstar_sq_minus_qc_sq_rho{rho}"] = float(qs ** 2 - qc2)
    qs2 = norm.ppf(1 - A0 / 2) ** 2
    SUM["C5_rho_crit"] = float(brentq(lambda r: 2 * np.log(r) / (1 - r ** -2) - qs2, 1.5, 20.0))
    SUM["C5_qstar"] = float(norm.ppf(1 - A0 / 2))


def check_xstar():
    df = pd.read_csv(os.path.join(RES, "lead4_match.csv"))
    rows = []
    for rho in RHOS:
        pool = pool_for(rho)
        D = delta_fn(pool)
        for w in [0.6, 0.75, 0.9, 0.95, 0.98]:
            kap = 2 * w - 1
            a1 = brentq(lambda a: a - kap * D(a) - A0, 1e-12, 1 - 1e-12, xtol=1e-15)
            a2 = brentq(lambda a: a + kap * D(a) - A0, 1e-12, 1 - 1e-12, xtol=1e-15)
            dW = (1 - w) * (D(a1) + D(a2))
            f = lambda lx: (A0 - ode_miss(pool, np.exp(lx))[0]) - dW
            lo, hi = np.log(0.05), np.log(400.0)
            xs = float(np.exp(brentq(f, lo, hi, xtol=1e-10)))
            sub = df[(df.rho == rho) & (np.isclose(df.w, w))]
            csvx = float(sub.x_star_ode.iloc[0]) if len(sub) else np.nan
            rows.append({"rho": rho, "w": w, "x_star_mine": xs, "x_star_csv": csvx,
                         "rel_diff": abs(xs - csvx) / csvx if np.isfinite(csvx) else np.nan,
                         "x_star_times_1mw": xs * (1 - w)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "lead4p_xstar_check.csv"), index=False)
    SUM["C5_xstar_rel_diff_max"] = float(out.rel_diff.max())


# ---------------------------------------------------------------------------
# C5: chain versus ODE at fixed x for halving gamma
# ---------------------------------------------------------------------------

def check_gap_scaling():
    rows = []
    for rho in RHOS:
        pool = pool_for(rho)
        for x in (1.0, 2.5, 5.0):
            m1o, m2o, A, B = ode_miss(pool, x)
            for gamma in (0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125):
                m = x / gamma
                if abs(m - round(m)) > 1e-9:
                    continue
                m = int(round(m))
                t0 = time.time()
                f4 = M.f4_exact_chain(pool, gamma, m)
                gap1 = f4[0] - m1o
                gap2 = f4[1] - m2o
                gap = max(abs(gap1), abs(gap2))
                rows.append({"rho": rho, "x": x, "gamma": gamma, "m": m, "ode_miss1": m1o, "chain_miss1": f4[0],
                             "signed_gap_regime1": gap1, "abs_gap": gap,
                             "gap_over_gamma": gap / gamma, "gap_over_sqrt_gamma": gap / np.sqrt(gamma),
                             "edge_mass": f4[4], "cycles": f4[5], "seconds": time.time() - t0})
                print(f"gap rho={rho} x={x} gamma={gamma} m={m}: gap={gap:.3e} cycles={f4[5]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lead4p_gap_scaling.csv"), index=False)
    # log-log slope of gap against gamma, per (rho, x), using the smallest four gammas
    for (rho, x), g in df.groupby(["rho", "x"]):
        g = g.sort_values("gamma")
        use = g[g.gamma <= 0.025]
        sl = np.polyfit(np.log(use.gamma), np.log(use.abs_gap), 1)[0]
        SUM[f"C6_slope_rho{rho}_x{x}"] = float(sl)
    SUM["C6_gamma_over_gap_ratio_max"] = float(df.gap_over_gamma.max())


# ---------------------------------------------------------------------------
# C6: explicit constants of the chain-to-ODE theorem (one cell, several gamma)
# ---------------------------------------------------------------------------

def check_thm_constants():
    """Explicit constants of Theorem 6 (chain versus ODE), for one cell and several gamma.
    eta(T, gamma) = sqrt(gamma T) + L b0 gamma T / 2 is the L2 one-horizon tracking error (Lemma 6);
    delta_B = eta(x), delta_P = eta(2x); p_out <= 2 n0 delta_P / r (Markov);
    u <= delta_P / (1 - lambda) + (1 + gamma) p_out; miss-rate gap <= (delta_B + 2u) / x."""
    rho, x = 2.0, 2.5
    pool = pool_for(rho)
    L, b0 = 2.0, max(A0, 1 - A0)
    a1, a2 = pool.fixed_points()[0], pool.fixed_points()[1]
    A, B = orbit(pool, x)
    r = 0.5 * min(a2, 1 - a1)
    s1 = s_slope(pool, 0, a2 - r)
    s2 = s_slope(pool, 1, a1 + r)
    lam = float(np.exp(-x * (s1 + s2)))
    lo, hi = -1.0, 2.0
    n0 = 0
    while max(abs(lo - A), abs(hi - A)) > r / 2 and n0 < 500:
        lo, hi = psi(pool, lo, x), psi(pool, hi, x)
        n0 += 1
    eta = lambda T, g: np.sqrt(g * T) + L * b0 * g * T / 2
    rows = []
    for gamma in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        dB, dP = eta(x, gamma), eta(2 * x, gamma)
        p_out = min(1.0, 2 * n0 * dP / r)
        u = dP / (1 - lam) + (1 + gamma) * p_out
        bound = (dB + 2 * u) / x
        rows.append({"rho": rho, "x": x, "gamma": gamma, "r": r, "lambda": lam, "n0": n0,
                     "delta_B": dB, "delta_P": dP, "p_out": p_out, "u_bound": u, "miss_gap_bound": bound,
                     "bound_over_sqrt_gamma": bound / np.sqrt(gamma)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lead4p_thm5_constants.csv"), index=False)
    ok = df[df.miss_gap_bound < 0.01]
    SUM["C7_thm_n0"] = int(n0)
    SUM["C7_thm_lambda"] = float(lam)
    SUM["C7_thm_r"] = float(r)
    SUM["C7_thm_first_gamma_below_0.01"] = float(ok.gamma.max()) if len(ok) else None
    SUM["C7_thm_const_over_sqrtgamma_small_gamma"] = float(df.bound_over_sqrt_gamma.iloc[-1])


# ---------------------------------------------------------------------------
# C7: tanh special case and rolling window versus ideal pool
# ---------------------------------------------------------------------------

def check_f1_special():
    rows = []
    for rho in (1.02, 1.05):
        pool = M.Pool((1.0, rho), (0.5, 0.5), A0)
        q0, p0, s0 = pool.zero_order()
        g = A0 - p0[0]
        for x in (0.5, 2.0, 6.0):
            m1, m2, delta = M.f1_linear(pool, x / 100.0, 100)
            dev = A0 - m1
            tanh_form = g * np.tanh(x / 2) / (x / 2)
            rows.append({"rho": rho, "x": x, "F1_dev": dev, "tanh_form": tanh_form, "g": g,
                         "s1": s0[0], "s2": s0[1], "rel_diff": abs(dev - tanh_form) / tanh_form})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lead4p_f1_tanh.csv"), index=False)
    SUM["C8_tanh_rel_diff_max"] = float(df.rel_diff.max())
    # exactness of the closed form for the linearised ODE: compare F1 with a direct linear solve
    worst = 0.0
    for rho in RHOS:
        pool = pool_for(rho)
        q0, p0, s0 = pool.zero_order()
        for x in (0.3, 1.0, 4.0):
            m1, m2, _ = M.f1_linear(pool, x / 100.0, 100)
            # linearised ODE solved by RK on the tangent line
            def lin_flow(r, a0_, xx):
                f = lambda t, a: [xx * (A0 - (p0[r] + s0[r] * (a[0] - A0)))]
                return float(solve_ivp(f, (0, 1), [a0_], rtol=1e-12, atol=1e-14).y[0, -1])
            g = lambda A: lin_flow(1, lin_flow(0, A, x), x) - A
            Alin = brentq(g, -0.5, 1.5, xtol=1e-14)
            Blin = lin_flow(0, Alin, x)
            worst = max(worst, abs((A0 - (Blin - Alin) / x) - m1))
    SUM["C8_f1_vs_direct_linear_solve_maxdiff"] = float(worst)


def check_rolling():
    rw = pd.read_csv(os.path.join(RES, "lead4_rolling_cells.csv"))
    bad = rw[rw.err_IDEAL_F4 > 0.01]
    SUM["C9_ideal_fail_count"] = int(len(bad))
    SUM["C9_ideal_fail_m450"] = int((bad.m == 450).sum())
    SUM["C9_ideal_fail_max_err"] = float(rw.err_IDEAL_F4.max())
    SUM["C9_ideal_fail_rows"] = [int(i + 1) for i in bad.index]
    SUM["C9_win_mf_max_err"] = float(rw.err_WIN_MF.max())
    SUM["C9_win_cl_max_err"] = float(rw.err_WIN_CL.max())
    SUM["C9_win_mf_within"] = int((rw.err_WIN_MF <= 0.01).sum())
    SUM["C9_win_cl_within"] = int((rw.err_WIN_CL <= 0.01).sum())
    SUM["C9_roll_reps"] = int(rw.roll_reps.iloc[0])
    SUM["C9_ideal_within"] = int((rw.err_IDEAL_F4 <= 0.01).sum())
    SUM["C9_n_cells"] = int(len(rw))
    SUM["C9_max_roll_se"] = float(max(rw.roll_se1.max(), rw.roll_se2.max()))


def check_local_error():
    """Lemma 10 (L2 tracking error over one period) against simulation of the chain."""
    rho, x = 2.0, 2.5
    pool = pool_for(rho)
    L, b0 = 2.0, max(A0, 1 - A0)
    Astar, Bstar = orbit(pool, x)
    rng = np.random.default_rng(20261105)
    reps = 4000
    rows = []
    for gamma in (0.05, 0.0125):
        m = int(round(x / gamma))
        eta = np.sqrt(gamma * 2 * x) + L * b0 * gamma * 2 * x / 2
        for start in (0.0, 0.3, 0.9, Astar):
            a = np.full(reps, start)
            for r in (0, 1):
                for _ in range(m):
                    p = pool.p(r, a)
                    miss = rng.random(reps) < p
                    a = a + gamma * (A0 - miss)
            target = psi(pool, start, x)
            rms = float(np.sqrt(np.mean((a - target) ** 2)))
            rows.append({"gamma": gamma, "m": m, "start": start, "rms_error": rms, "eta_bound": eta,
                         "ratio": rms / eta})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lead4p_local_error.csv"), index=False)
    SUM["C10_local_ratio_max"] = float(df.ratio.max())
    SUM["C10_local_all_below_bound"] = bool((df.ratio < 1).all())
    SUM["C10_local_reps"] = reps


def check_contraction():
    """Theorem 2(ii) contraction ratio on J and Theorem 6 local contraction on the ball."""
    rows = []
    for rho in RHOS:
        pool = pool_for(rho)
        x = 2.5
        a1, a2 = pool.fixed_points()[0], pool.fixed_points()[1]
        A, B = orbit(pool, x)
        lamJ = float(np.exp(-x * (s_slope(pool, 0, a2) + s_slope(pool, 1, a1))))
        pts = np.linspace(a2, a1, 25)
        vals = [psi(pool, a, x) for a in pts]
        rJ = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                rJ = max(rJ, abs(vals[j] - vals[i]) / (pts[j] - pts[i]))
        beta = 0.5 * min(a2, 1 - a1)
        lamB = float(np.exp(-x * (s_slope(pool, 0, a2 - beta) + s_slope(pool, 1, a1 + beta))))
        rB = 0.0
        for a in np.linspace(A - beta, A + beta, 31):
            if abs(a - A) > 1e-9:
                rB = max(rB, abs(psi(pool, a, x) - A) / abs(a - A))
        rows.append({"rho": rho, "x": x, "lambda_J": lamJ, "max_secant_ratio_on_J": rJ,
                     "lambda_beta": lamB, "max_ratio_on_ball": rB, "beta": beta})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lead4p_contraction.csv"), index=False)
    SUM["C11_J_ratio_ok"] = bool((df.max_secant_ratio_on_J <= df.lambda_J + 1e-9).all())
    SUM["C11_ball_ratio_ok"] = bool((df.max_ratio_on_ball <= df.lambda_beta + 1e-9).all())
    SUM["C11_J_slack_min"] = float((df.lambda_J - df.max_secant_ratio_on_J).min())
    SUM["C11_ball_slack_min"] = float((df.lambda_beta - df.max_ratio_on_ball).min())


def check_curvature_and_c_asymptotics():
    rows = []
    for rho in (1.2, 1.5, 2.0):
        pool = M.Pool((1.0, rho), (0.5, 0.5), A0)
        for a in (1e-3, 1e-6, 1e-9, 1e-12):
            rows.append({"rho": rho, "a": a, "p1_second": p1_second(pool, a)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lead4p_curvature.csv"), index=False)
    for rho in (1.2, 1.5, 2.0):
        v = df[df.rho == rho].p1_second.values
        SUM[f"C12_p1pp_rho{rho}_first"] = float(v[0])
        SUM[f"C12_p1pp_rho{rho}_last"] = float(v[-1])
        SUM[f"C12_p1pp_rho{rho}_grows"] = bool(v[-1] > v[0])
    # w -> 1 asymptotics of c and d_W
    rows = []
    for rho in RHOS:
        pool = pool_for(rho)
        D = delta_fn(pool)
        a1s, a2s = pool.fixed_points()[0], pool.fixed_points()[1]
        for w in (0.99, 0.999):
            kap = 2 * w - 1
            a1 = brentq(lambda a: a - kap * D(a) - A0, 1e-12, 1 - 1e-12, xtol=1e-15)
            a2 = brentq(lambda a: a + kap * D(a) - A0, 1e-12, 1 - 1e-12, xtol=1e-15)
            c = (1 - w) * (D(a2) - D(a1))
            dW = (1 - w) * (D(a1) + D(a2))
            rows.append({"rho": rho, "w": w, "c_over_1mw": c / (1 - w), "pred_c": -(a1s + a2s - 2 * A0),
                         "dW_over_1mw": dW / (1 - w), "pred_dW": a1s - a2s})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lead4p_w_asymptotics.csv"), index=False)
    last = df[np.isclose(df.w, 0.999)]
    SUM["C12_c_asym_maxrel"] = float((abs(last.c_over_1mw - last.pred_c) / abs(last.pred_c)).max())
    SUM["C12_dW_asym_maxrel"] = float((abs(last.dW_over_1mw - last.pred_dW) / abs(last.pred_dW)).max())


def main():
    t0 = time.time()
    check_block_identity(); print("C1 done", time.time() - t0, flush=True)
    check_scaling(); print("C2 done", time.time() - t0, flush=True)
    check_ode_cells(); print("C3 cells done", time.time() - t0, flush=True)
    check_attraction(); print("C3 attraction done", time.time() - t0, flush=True)
    check_dx_monotone(); print("C4 dx done", time.time() - t0, flush=True)
    check_match_closed(); print("C5 closed done", time.time() - t0, flush=True)
    check_xstar(); print("C5 xstar done", time.time() - t0, flush=True)
    check_thm_constants(); print("C7 constants done", time.time() - t0, flush=True)
    check_f1_special(); print("C8 done", time.time() - t0, flush=True)
    check_rolling(); print("C9 done", time.time() - t0, flush=True)
    check_local_error(); print("C10 done", time.time() - t0, flush=True)
    check_contraction(); print("C11 done", time.time() - t0, flush=True)
    check_curvature_and_c_asymptotics(); print("C12 done", time.time() - t0, flush=True)
    check_gap_scaling(); print("C6 gap done", time.time() - t0, flush=True)
    SUM["seconds"] = time.time() - t0
    with open(os.path.join(RES, "lead4p_summary.json"), "w") as fh:
        json.dump(SUM, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(SUM, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
