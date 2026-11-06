"""
lead4_model.py: regime-wise long-run coverage of ACI under alternating regimes.

Model (stated once, used everywhere in the lead4_* files)
---------------------------------------------------------
* Time is cut into blocks of length m.  Block 1 is regime 1 (residual scale
  sigma_1 = 1), block 2 is regime 2 (scale sigma_2 = rho), block 3 is regime 1,
  and so on.  In regime r the residual is |R|, R ~ N(0, sigma_r^2), independent
  over time.
* Pool model: the calibration pool is the mixture of the two half-normal laws
  with weights pi_1, pi_2 (default 1/2, 1/2: equal blocks, a pool that
  contains whole cycles).  The pool is treated as the population mixture (an
  infinite pool).  The quantile used at level alpha is the population mixture
  quantile q(alpha), the solution of  sum_r pi_r S_r(q) = alpha,
  S_r(q) = P(|R_r| > q) = erfc(q / (sigma_r sqrt 2)).
* Free ACI update (no projection):  alpha_{t+1} = alpha_t + gamma (alpha_0 -
  miss_t); miss_t = 1{|R_t| > q(alpha_t)} with the conventions of dasc_lib:
  alpha_t <= 0 gives the whole line (miss = 0), alpha_t >= 1 gives the empty
  set (miss = 1).  So the miss probability in regime r at level alpha is
  p_r(alpha) = S_r(q(alpha)), extended by 0 for alpha <= 0 and 1 for alpha >= 1.
* alpha_0 = 0.1 is the target miscoverage; alpha_1 = alpha_0.

Identities used (proved in the report)
--------------------------------------
(I1) pooled-mixture identity: sum_r pi_r p_r(alpha) = alpha for alpha in [0,1].
(I2) block telescoping, exact for the actual recursion: summing the update
     over one block of length m gives
         (1/m) sum_{t in block} E[miss_t] = alpha_0 - E[alpha_end - alpha_start] / (gamma m).
     In the periodic steady state with equal blocks the swing in regime 2 is minus
     the swing in regime 1, so with Delta = E[alpha_end - alpha_start] in regime 1
         miss_1 = alpha_0 - Delta/(gamma m),  miss_2 = alpha_0 + Delta/(gamma m).
(I3) slope: p_r'(alpha) = f_r(q)/f_mix(q) =: s_r(alpha), sum_r pi_r s_r = 1.

Formulas implemented
--------------------
F1  closed form from linearising p_r at alpha_0 (explicit in exponentials).
F2  exact solution of the averaged ODE alpha' = gamma (alpha_0 - p_r(alpha)),
    periodic steady state by shooting (quadrature level).
F3  Gaussian moment closure of the stochastic recursion (mean and variance
    carried through the cycle; the variance update is exact, the closure
    E[p(alpha)] under a normal law for alpha is the approximation).
F4  exact periodic stationary law of the chain on the lattice
    alpha = alpha_0 + (gamma alpha_0) k (valid for any alpha_0 = 1/n); this is the
    exact numerical benchmark, not a closed form.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.special import erfc

SQ2 = np.sqrt(2.0)
SQ2PI = np.sqrt(2.0 * np.pi)


class Pool:
    """Two-scale half-normal pool; scales sig[0], sig[1]; mixture weights pi."""

    def __init__(self, sig=(1.0, 2.0), pi=(0.5, 0.5), alpha0=0.1):
        self.sig = np.asarray(sig, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        self.alpha0 = float(alpha0)
        self._tab = None

    # --- laws -----------------------------------------------------------
    def surv(self, r, q):
        return erfc(np.asarray(q, dtype=float) / (self.sig[r] * SQ2))

    def dens(self, r, q):
        z = np.asarray(q, dtype=float) / self.sig[r]
        return 2.0 * np.exp(-0.5 * z * z) / (SQ2PI * self.sig[r])

    def smix(self, q):
        return self.pi[0] * self.surv(0, q) + self.pi[1] * self.surv(1, q)

    def fmix(self, q):
        return self.pi[0] * self.dens(0, q) + self.pi[1] * self.dens(1, q)

    def q_of_alpha(self, alpha):
        """Mixture quantile at level 1 - alpha, vectorised bisection.
        alpha <= 0 -> inf (whole line); alpha >= 1 -> 0 (empty set)."""
        a = np.atleast_1d(np.asarray(alpha, dtype=float))
        lo = np.zeros_like(a)
        hi = np.full_like(a, 14.0 * self.sig.max())
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            big = self.smix(mid) > a        # survival still above alpha -> need larger q
            lo = np.where(big, mid, lo)
            hi = np.where(big, hi, mid)
        q = 0.5 * (lo + hi)
        q = np.where(a <= 0.0, np.inf, q)
        q = np.where(a >= 1.0, 0.0, q)
        return q

    def p_exact(self, r, alpha):
        """p_r(alpha) = S_r(q(alpha)) with the free-update conventions."""
        a = np.atleast_1d(np.asarray(alpha, dtype=float))
        q = self.q_of_alpha(a)
        with np.errstate(invalid="ignore"):
            p = np.where(np.isinf(q), 0.0, self.surv(r, np.where(np.isinf(q), 0.0, q)))
        p = np.where(a <= 0.0, 0.0, p)
        p = np.where(a >= 1.0, 1.0, p)
        return p

    def slope(self, r, alpha):
        """s_r(alpha) = f_r(q)/f_mix(q) = p_r'(alpha), for 0 < alpha < 1."""
        q = self.q_of_alpha(alpha)
        return self.dens(r, q) / self.fmix(q)

    # --- interpolation tables for p_r (used by the ODE and moment closure) --
    def build_tables(self, n=40001):
        # grid denser near the ends where p_r is least linear
        u = np.linspace(0.0, 1.0, n)
        g = 0.5 * (1 - np.cos(np.pi * u))     # Chebyshev-like spacing
        self._g = g
        self._tab = [self.p_exact(0, g), self.p_exact(1, g)]
        return self

    def p(self, r, alpha):
        if self._tab is None:
            self.build_tables()
        return np.interp(alpha, self._g, self._tab[r])

    # --- the two fixed points and the zero-order quantities -----------------
    def zero_order(self):
        a0 = self.alpha0
        q0 = float(self.q_of_alpha(a0)[0])
        p0 = np.array([float(self.surv(0, q0)), float(self.surv(1, q0))])
        s0 = np.array([float(self.slope(0, a0)[0]), float(self.slope(1, a0)[0])])
        return q0, p0, s0

    def fixed_points(self):
        """alpha_r* with p_r(alpha_r*) = alpha_0 (the level that gives regime r its own coverage)."""
        a0 = self.alpha0
        out = []
        for r in (0, 1):
            # p_r(alpha) - alpha_0 is increasing in alpha on (0, 1)
            f = lambda a: float(self.p_exact(r, a)[0]) - a0
            out.append(brentq(f, 1e-12, 1 - 1e-12, xtol=1e-14))
        return np.array(out)


# ----------------------------------------------------------------------------
# F1: linearised closed form
# ----------------------------------------------------------------------------

def f1_linear(pool: Pool, gamma, m):
    """Closed form from linearising p_r about alpha_0.

    x = alpha - alpha_0 obeys x' = -gamma s_r (x - c_r) in regime r with
    c_r = (alpha_0 - p_r(alpha_0)) / s_r,  a_r = exp(-gamma s_r m).
    Periodic solution (equal blocks):
        Delta = x_end - x_start (regime 1) = (c_1 - c_2)(1-a_1)(1-a_2)/(1-a_1 a_2)
        miss_1 = alpha_0 - Delta/(gamma m),  miss_2 = alpha_0 + Delta/(gamma m).
    Valid for equal blocks and any pool weights; the weights enter only through
    p_r(alpha_0) and s_r(alpha_0).
    """
    a0 = pool.alpha0
    _, p0, s0 = pool.zero_order()
    c = (a0 - p0) / s0
    a = np.exp(-gamma * s0 * m)
    delta = (c[0] - c[1]) * (1 - a[0]) * (1 - a[1]) / (1 - a[0] * a[1])
    d = delta / (gamma * m)
    return a0 - d, a0 + d, delta


# ----------------------------------------------------------------------------
# F2: averaged ODE, periodic steady state by shooting
# ----------------------------------------------------------------------------

def _flow(pool: Pool, r, x0, gamma, m, hscale=0.02):
    """Exact flow of alpha' = gamma (alpha_0 - p_r(alpha)) over time m (RK4, fine step)."""
    a0 = pool.alpha0
    # step so that gamma * s_max * h <= 0.02 and h <= 1
    smax = 4.0
    h = min(1.0, hscale / (gamma * smax))
    n = max(int(np.ceil(m / h)), 50)
    h = m / n
    x = float(x0)
    f = lambda y: gamma * (a0 - float(pool.p(r, y)))
    for _ in range(n):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x


def f2_ode(pool: Pool, gamma, m, hscale=0.02):
    """Periodic steady state of the averaged ODE.  Returns miss_1, miss_2, A (alpha at
    the start of regime 1), B (alpha at the end of regime 1)."""
    a0 = pool.alpha0
    fp = pool.fixed_points()

    def cyc(A):
        B = _flow(pool, 0, A, gamma, m, hscale)
        return _flow(pool, 1, B, gamma, m, hscale)

    lo = min(fp) - 0.05
    hi = max(fp) + 0.05
    g = lambda A: cyc(A) - A
    # cyc is a contraction; g(lo) > 0 and g(hi) < 0
    A = brentq(g, lo, hi, xtol=1e-12)
    B = _flow(pool, 0, A, gamma, m, hscale)
    d = (B - A) / (gamma * m)
    return a0 - d, a0 + d, A, B


# ----------------------------------------------------------------------------
# F3: Gaussian moment closure (discrete time, carried through the cycle)
# ----------------------------------------------------------------------------

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(31)
_GH_W = _GH_W / _GH_W.sum()


def _closure_step(pool: Pool, r, mean, var, gamma):
    """One step of the moment recursion.  Exact: mean' = mean + gamma(alpha_0 - E p),
    var' = var + gamma^2 E p (1 - E p) - 2 gamma Cov(alpha, p(alpha)).
    Approximation: alpha ~ N(mean, var) when evaluating E p and Cov."""
    sd = np.sqrt(max(var, 0.0))
    al = mean + sd * _GH_X
    pv = pool.p(r, al)
    mp = float(np.dot(_GH_W, pv))
    cov = float(np.dot(_GH_W, (al - mean) * (pv - mp)))
    mean2 = mean + gamma * (pool.alpha0 - mp)
    var2 = var + gamma * gamma * mp * (1 - mp) - 2 * gamma * cov
    return mean2, max(var2, 0.0), mp


def f3_closure(pool: Pool, gamma, m, init=None, tol=1e-11, max_cycles=2000):
    """Periodic fixed point of the moment closure.  Returns miss_1, miss_2 (block
    averages of E p), mean at start of regime 1, variance at start of regime 1."""
    a0 = pool.alpha0
    if init is None:
        fp = pool.fixed_points()
        init = (a0, gamma * a0 * (1 - a0) / 2.0)
    mean, var = init
    for it in range(max_cycles):
        s1 = 0.0
        for _ in range(m):
            mean, var, mp = _closure_step(pool, 0, mean, var, gamma)
            s1 += mp
        s2 = 0.0
        for _ in range(m):
            mean, var, mp = _closure_step(pool, 1, mean, var, gamma)
            s2 += mp
        if it > 0 and abs(mean - prev[0]) < tol and abs(var - prev[1]) < tol:
            break
        prev = (mean, var)
    # one more recorded cycle from the fixed point
    mean0, var0 = mean, var
    s1 = s2 = 0.0
    for _ in range(m):
        mean, var, mp = _closure_step(pool, 0, mean, var, gamma)
        s1 += mp
    for _ in range(m):
        mean, var, mp = _closure_step(pool, 1, mean, var, gamma)
        s2 += mp
    return s1 / m, s2 / m, mean0, var0, it + 1


# ----------------------------------------------------------------------------
# F4: exact periodic stationary law of the lattice chain
# ----------------------------------------------------------------------------

def f4_exact_chain(pool: Pool, gamma, m, tol=1e-12, max_cycles=20000):
    """Exact periodic stationary law of alpha_t for the free update, alpha_0 = 1/n.

    alpha_t = alpha_0 + (gamma alpha_0) k_t with k_t integer: a miss-free step adds
    +1 to k, a miss adds -(n-1) = -(1-alpha_0)/alpha_0 (9 when alpha_0 = 0.1).
    Returns block-average miss_1, miss_2, E[alpha] at start and end of regime 1,
    the probability mass within 1e-9 of the lattice ends (truncation check) and the
    number of cycles used."""
    a0 = pool.alpha0
    n = int(round(1.0 / a0))
    assert abs(n * a0 - 1.0) < 1e-12, "alpha_0 must be 1/n"
    step_up = 1
    step_dn = n - 1
    unit = gamma * a0
    kmin = int(np.floor((-gamma - a0) / unit)) - 2
    kmax = int(np.ceil((1 + gamma - a0) / unit)) + 2
    ks = np.arange(kmin, kmax + 1)
    K = len(ks)
    alpha_k = a0 + unit * ks
    P = [pool.p_exact(0, alpha_k), pool.p_exact(1, alpha_k)]
    # ensure absorbing logic: p=1 above alpha>=1 (move down), p=0 at alpha<=0 (move up)
    dist = np.zeros(K)
    dist[-kmin] = 1.0

    def step(d, r):
        up = d * (1.0 - P[r])
        dn = d * P[r]
        new = np.zeros(K)
        new[step_up:] += up[:-step_up]
        new[:-step_dn] += dn[step_dn:]
        # mass that would leave the lattice window: only possible if truncation is wrong
        lost = up[-step_up:].sum() + dn[:step_dn].sum()
        return new, lost

    lost_total = 0.0
    prev = None
    for it in range(max_cycles):
        d = dist
        for r in (0, 1):
            for _ in range(m):
                d, lost = step(d, r)
                lost_total += lost
        dist = d
        if prev is not None and np.abs(dist - prev).sum() < tol:
            break
        prev = dist.copy()
    # recorded cycle
    d = dist
    start_mean = float(np.dot(d, alpha_k))
    s = [0.0, 0.0]
    mean_end1 = None
    for r in (0, 1):
        for _ in range(m):
            s[r] += float(np.dot(d, P[r]))
            d, lost = step(d, r)
        if r == 0:
            mean_end1 = float(np.dot(d, alpha_k))
    edge_mass = float(dist[:3].sum() + dist[-3:].sum())
    return s[0] / m, s[1] / m, start_mean, mean_end1, edge_mass, it + 1, lost_total


# ----------------------------------------------------------------------------
# (ii) weighted quantile at a fixed level, weight mass w on the same regime
# ----------------------------------------------------------------------------

def weighted_miss(pool: Pool, w):
    """Regime-wise miss rates of the weighted quantile at fixed level alpha_0.
    In regime r the pool puts mass w on regime-r residuals and 1-w on the other.
    q_r solves  w S_r(q) + (1-w) S_other(q) = alpha_0;  miss_r = S_r(q_r).
    Returns (miss_1, miss_2)."""
    a0 = pool.alpha0
    out = []
    for r in (0, 1):
        o = 1 - r
        f = lambda q: w * float(pool.surv(r, q)) + (1 - w) * float(pool.surv(o, q)) - a0
        q = brentq(f, 1e-12, 14.0 * pool.sig.max(), xtol=1e-14)
        out.append(float(pool.surv(r, q)))
    return out[0], out[1]


def weighted_miss_linear(pool: Pool, w):
    """First order in (w - 1/2) about the pooled quantile (pi = 1/2):
    miss_1 ~ p_1 + (w-1/2) s_1 g0,  miss_2 ~ p_2 - (w-1/2) s_2 g0,  g0 = p_2 - p_1."""
    _, p0, s0 = pool.zero_order()
    g0 = p0[1] - p0[0]
    return p0[0] + (w - 0.5) * s0[0] * g0, p0[1] - (w - 0.5) * s0[1] * g0
