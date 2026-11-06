"""
lead1_sim.py: kill test of "lead 1" (block-calibrated test / lower confidence bound for the
Lipschitz constant of the conditional residual law, under serial dependence).

Nothing in the existing code or results is modified; this file only reads dasc_lib.grid_term.

DATA-GENERATING MODEL (exogenous features, so conditioning on the feature path is legitimate)
  s_t in R^d (d = 1 or 2): each coordinate a stationary AR(1), coefficient PHI_S = 0.95, N(0,1) marginal.
  g(s) = (sum_j s_j)/sqrt(d)  (a unit direction, so g is 1-Lipschitz in the Euclidean norm)
  sigma_t = exp(kappa * g(s_t))
  Z_t stationary Gaussian AR(1) with correlation rho in {0, 0.5, 0.9}, independent of s
  R_t = sigma_t * |Z_t|, so F(r | s) = 2 Phi(r / sigma(s)) - 1 (half-normal), as in check_perstep_bound.py.
  Assumption A(ii) holds with delta = 0.  The conditional laws given the feature path are NOT independent
  across t when rho > 0, so the independence step of the pair bound of the article fails.

TRUE CONSTANTS (computed from the model on the realised feature path)
  L_path : max over realised point pairs (i, j) of KS(half-normal(sigma_i), half-normal(sigma_j)) / ||s_i - s_j||
           (closed form for the KS distance; the maximum is found among the 40 nearest neighbours of each
           point, and checked against all pairs in lead1_selftest.py).  This is "the true L" of the task.
  L_cl   : max over retained cluster pairs (A,B) of sup_r |F_A(r) - F_B(r)| / D_AB with F_A the average of the
           exact conditional CDFs over cluster A.  L_cl <= L_path always.  It is the sharp population target of
           the diagnostic (the quantity the cluster-pair statistic estimates), reported as a second target.

ESTIMATORS (all are lower confidence bounds Lhat; the test of H0: L <= L0 rejects iff Lhat > L0)
  C0   existing diagnostic, independence DKW-grid term eps(|A|, eta), eta = 0.05 per set (as in
       run_assumption_diagnostic.py).  C0b: the same with eta/J (J = number of retained clusters), which is the
       form the proposition actually guarantees (probability >= 1 - J*eta).
  C1   blocked analytic term: |A| replaced by |A| / tau_hat, tau_hat = 1 + 2 sum_k Bartlett_k rho_hat(k) estimated
       from the within-cluster probability-integral-transform indicators.  Heuristic, unproved.
  B1   simultaneous circular block bootstrap, FIXED block length b = round(n^(1/3)), common critical value.
  B2   same with a DATA-DRIVEN block length (Politis-White rule for the circular bootstrap) [redesign 1].
  B3   same as B2 but each pair's statistic is divided by its own bootstrap scale (studentised) [redesign 2].

  Bootstrap statistic.  For pair p = (A,B), KShat_p = max_g |Fhat_A(r_g) - Fhat_B(r_g)| on a grid of G pooled-sample
  quantiles.  Because sup|F_A - F_B| >= KShat_p - sup|Delta_p| with Delta_p = (Fhat_A - F_A) - (Fhat_B - F_B)
  (reverse triangle inequality in the sup norm), and sup|F_A - F_B| <= L D_p under A(ii) with delta = 0,
      Lhat = max_p (KShat_p - c_p)^+ / D_p,
  with c_p a simultaneous (1 - alpha) quantile of sup|Delta_p| (B1, B2: one common c; B3: c_p = z * s_p, s_p the
  bootstrap mean of sup|Delta*_p|, z the (1-alpha) quantile of max_p sup|Delta*_p| / s_p).
  Delta* is built from the time-ordered centred indicators e_t(r) = 1{R_t <= r} - Fhat_{c(t)}(r) (c(t) the cluster
  of t; points in dropped small clusters are set to zero): resample ceil-free m = round(n/b) circular blocks of
  length b with random starts, sum e* within each cluster, divide by the fixed cluster size, rescale by
  sqrt(n/(m b)).  The same resampled index set is used for all clusters, so cross-cluster serial dependence is
  kept.  Centring at the cluster mean makes the bootstrap law of Delta* conservative for within-cluster
  heterogeneity of F(.|s_i) (variance of an indicator mixture is at least the mean of the individual variances).
  Test/dual: reject H0: L <= L0 iff Lhat > L0, so size at L0 = L is 1 - coverage of L by Lhat.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from sklearn.cluster import KMeans

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from dasc_lib import grid_term  # existing code, read only

PHI_S = 0.95
ALPHA = 0.05          # eta for all bounds
G_GRID = 64           # grid size for the bootstrap statistic
G_FINE = 400          # grid size for the direct check of the independence terms
B_BOOT = 400          # bootstrap draws
MIN_PER = 30          # as in run_assumption_diagnostic.py
KNN = 40


# ----------------------------------------------------------------------------- data

def ar1_gauss(rng, n, rho, d=1):
    """Stationary AR(1) with N(0,1) marginals, shape (n, d)."""
    x = np.empty((n, d))
    x[0] = rng.standard_normal(d)
    c = math.sqrt(max(0.0, 1.0 - rho * rho))
    eta = rng.standard_normal((n, d))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + c * eta[t]
    return x


def simulate(seed_seq, n, rho, d, kappa):
    rng = np.random.default_rng(seed_seq)
    s = ar1_gauss(rng, n, PHI_S, d)
    z = ar1_gauss(rng, n, rho, 1)[:, 0]
    g = s.sum(axis=1) / math.sqrt(d)
    sigma = np.exp(kappa * g)
    r = sigma * np.abs(z)
    return s, sigma, r


def ks_half_normal(s1, s2):
    """Exact sup-distance between |N(0,s1^2)| and |N(0,s2^2)| (same closed form as check_perstep_bound.py)."""
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    c = np.maximum(s1, s2) / np.minimum(s1, s2)
    out = np.zeros_like(c)
    m = c > 1 + 1e-12
    cm = c[m]
    xs = np.sqrt(2.0 * np.log(cm) / (cm ** 2 - 1.0))
    out[m] = 2.0 * (ndtr(cm * xs) - ndtr(xs))
    return out


def l_path(s, sigma, knn=KNN, full=False):
    """True Lipschitz constant on the realised path: max over point pairs of KS/||s_i - s_j||."""
    n = len(sigma)
    if full:
        best = 0.0
        for j in range(n):
            d = np.linalg.norm(s - s[j], axis=1)
            ks = ks_half_normal(sigma, np.full(n, sigma[j]))
            ok = d > 1e-12
            best = max(best, float(np.max(ks[ok] / d[ok])))
        return best
    k = min(knn + 1, n)
    tree = cKDTree(s)
    dist, idx = tree.query(s, k=k)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    ks = ks_half_normal(np.repeat(sigma[:, None], k - 1, axis=1), sigma[idx])
    ok = dist > 1e-12
    return float(np.max(ks[ok] / dist[ok]))


# ----------------------------------------------------------------------------- clusters

def clusters(s, k, seed):
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(s)
    labels = km.labels_
    keep = [c for c in range(k) if np.sum(labels == c) >= MIN_PER]
    remap = -np.ones(k, dtype=int)
    for new, c in enumerate(keep):
        remap[c] = new
    return remap[labels], len(keep)  # label -1 for dropped clusters


def pair_index(K):
    ia, ib = np.triu_indices(K, 1)
    return ia, ib


def mean_dist(s, lab, K, ia, ib):
    groups = [np.where(lab == c)[0] for c in range(K)]
    D = np.empty(len(ia))
    for p, (a, b) in enumerate(zip(ia, ib)):
        D[p] = cdist(s[groups[a]], s[groups[b]]).mean()
    return D, groups


def exact_ks(r, lab, K, ia, ib):
    """Exact two-sample KS for every pair via cumulative counts along the pooled sort order."""
    order = np.argsort(r)
    onehot = np.zeros((len(r), K))
    ok = lab[order] >= 0
    onehot[np.where(ok)[0], lab[order][ok]] = 1.0
    sizes = onehot.sum(axis=0)
    F = np.cumsum(onehot, axis=0) / sizes
    return np.abs(F[:, ia] - F[:, ib]).max(axis=0), sizes


def true_cluster_ratio(sigma, groups, ia, ib, D):
    """L_cl: max over cluster pairs of sup_r |F_A - F_B| / D_AB using the exact conditional CDFs."""
    rmax = 5.0 * sigma.max()
    rmin = sigma.min() / 20.0
    rg = np.geomspace(rmin, rmax, 300)
    FA = []
    for g in groups:
        FA.append((2.0 * ndtr(rg[None, :] / sigma[g][:, None]) - 1.0).mean(axis=0))
    FA = np.asarray(FA)
    ks = np.abs(FA[ia] - FA[ib]).max(axis=1)
    return float(np.max(ks / D)), ks


# ----------------------------------------------------------------------------- dependence summaries

def pit_series(r, lab, K):
    """Within-cluster probability integral transform, in time order, for kept points."""
    u = np.full(len(r), np.nan)
    for c in range(K):
        idx = np.where(lab == c)[0]
        rk = np.argsort(np.argsort(r[idx]))
        u[idx] = (rk + 0.5) / len(idx)
    return u[lab >= 0]


def acf(x, maxlag):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: maxlag + 1]
    return ac / ac[0] if ac[0] > 0 else np.zeros(maxlag + 1)


def tau_hat(vs):
    """Integrated autocorrelation time with Bartlett window, lag cut at twice the first insignificant lag."""
    out = []
    for v in vs:
        n = len(v)
        ml = min(n // 4, int(math.ceil(math.sqrt(n))) + 20 * 5)
        ro = acf(v, ml)
        thr = 2.0 / math.sqrt(n)
        below = np.where(np.abs(ro[1:]) < thr)[0]
        m1 = (below[0] + 1) if len(below) else ml
        M = min(2 * m1, ml)
        w = 1.0 - np.arange(1, M + 1) / (M + 1.0)
        out.append(1.0 + 2.0 * np.sum(w * ro[1 : M + 1]))
    return max(1.0, float(np.mean(out)))


def pw_block_length(vs, n_total):
    """Politis-White (2004) rule for the circular bootstrap (D_CB = 4/3 g(0)^2), maximum over the series."""
    best = 2.0
    for v in vs:
        n = len(v)
        KN = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
        mmax = int(math.ceil(math.sqrt(n))) + KN
        ro = acf(v, min(n - 1, mmax + KN))
        cvar = np.var(v)
        thr = 2.0 * math.sqrt(math.log10(n) / n)
        mhat = None
        for m in range(1, mmax + 1):
            if np.all(np.abs(ro[m + 1 : m + KN + 1]) < thr):
                mhat = m
                break
        if mhat is None:
            mhat = mmax
        M = min(2 * mhat, mmax)
        k = np.arange(-M, M + 1)
        t = np.abs(k) / M
        lam = np.where(t <= 0.5, 1.0, 2.0 * (1.0 - t))
        rk = np.concatenate([ro[1 : M + 1][::-1], ro[: M + 1]]) * cvar
        G = np.sum(lam * np.abs(k) * rk)
        g0 = np.sum(lam * rk)
        if g0 <= 0:
            continue
        b = (2.0 * G * G / ((4.0 / 3.0) * g0 * g0)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
        best = max(best, b)
    return int(min(max(2, round(best)), max(2, n_total // 8)))


# ----------------------------------------------------------------------------- bootstrap

def bootstrap_deltas(r, lab, K, ia, ib, b, rng, B=B_BOOT, G=G_GRID):
    """Return (KShat_p on the grid, array (B, P) of sup_g |Delta*_p|, grid, Fg).

    Same circular block bootstrap as bootstrap_deltas_reference (kept for the self-test), computed with histograms:
    the centred cluster sum is  sum_{t* in c} 1{R_t* <= r_g} - N*_c Fhat_c(r_g),  and the first term is a cumulative
    count over the grid bin of each resampled point.
    """
    n = len(r)
    keep = lab >= 0
    grid = np.quantile(r[keep], np.arange(1, G + 1) / (G + 1.0))
    sizes = np.array([np.sum(lab == c) for c in range(K)], dtype=float)
    g0 = np.searchsorted(grid, r, side="left")                  # first grid index with grid_g >= R_t  (G if none)
    Fg = np.empty((K, G))
    for c in range(K):
        Fg[c] = (r[lab == c][:, None] <= grid[None, :]).mean(axis=0)
    ks_hat = np.abs(Fg[ia] - Fg[ib]).max(axis=1)
    W = G + 1
    key = np.where(keep, lab * W + g0, K * W)                   # dropped points go to a dummy slot
    m = max(1, int(round(n / b)))
    scale = math.sqrt(n / (m * b))
    starts = rng.integers(0, n, size=(B, m))
    idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % n).reshape(B, m * b)
    kk = key[idx] + (np.arange(B)[:, None] * (K * W + 1))
    cnt = np.bincount(kk.ravel(), minlength=B * (K * W + 1)).reshape(B, K * W + 1)[:, : K * W].reshape(B, K, W)
    Nc = cnt.sum(axis=2)                                        # resampled count per cluster
    cum = np.cumsum(cnt[:, :, :G], axis=2)                      # sum of 1{R <= grid_g}
    T = (cum - Nc[:, :, None] * Fg[None, :, :]) * scale / sizes[None, :, None]
    D = T[:, ia, :] - T[:, ib, :]
    out = np.abs(D).max(axis=2)
    return ks_hat, out, grid, Fg


def bootstrap_deltas_reference(r, lab, K, ia, ib, b, rng, B=B_BOOT, G=G_GRID):
    """Return (KShat_p on the grid, array (B, P) of sup_g |Delta*_p|, grid, Fg)."""
    n = len(r)
    grid = np.quantile(r[lab >= 0], np.arange(1, G + 1) / (G + 1.0))
    U = (r[:, None] <= grid[None, :]).astype(float)           # n x G
    keep = lab >= 0
    sizes = np.array([np.sum(lab == c) for c in range(K)], dtype=float)
    M = np.zeros((K, n))
    M[lab[keep], np.where(keep)[0]] = 1.0
    Fg = (M @ U) / sizes[:, None]                              # K x G
    ks_hat = np.abs(Fg[ia] - Fg[ib]).max(axis=1)
    # centred indicators, zero for dropped points
    E = np.zeros((n, K, G))
    t_keep = np.where(keep)[0]
    E[t_keep, lab[t_keep], :] = U[t_keep] - Fg[lab[t_keep]]
    # circular cumulative sums over the doubled series
    E2 = np.concatenate([E, E[: b]], axis=0)
    cs = np.concatenate([np.zeros((1, K, G)), np.cumsum(E2, axis=0)], axis=0)
    S = (cs[b : b + n] - cs[0:n]).astype(np.float32)           # S[j] = sum_{t=j}^{j+b-1} (mod n)
    m = max(1, int(round(n / b)))
    scale = math.sqrt(n / (m * b))
    out = np.empty((B, len(ia)))
    chunk = 50
    for b0 in range(0, B, chunk):
        bc = min(chunk, B - b0)
        starts = rng.integers(0, n, size=(bc, m))
        T = S[starts].sum(axis=1, dtype=np.float64) * scale    # bc x K x G
        T = T / sizes[None, :, None]
        D = T[:, ia, :] - T[:, ib, :]
        out[b0 : b0 + bc] = np.abs(D).max(axis=2)
    return ks_hat, out, grid, Fg


def lower_bound_bootstrap(ks_hat, D, sup_star, studentise):
    if studentise:
        s = sup_star.mean(axis=0)
        s = np.maximum(s, 1e-12)
        z = np.quantile((sup_star / s[None, :]).max(axis=1), 1.0 - ALPHA)
        c = z * s
    else:
        cc = np.quantile(sup_star.max(axis=1), 1.0 - ALPHA)
        c = np.full(len(D), cc)
    return float(np.max(np.maximum(ks_hat - c, 0.0) / D)) if len(D) else 0.0, c


def lower_bound_dkw(ks, D, sizes_eff, ia, ib, eta):
    eps = np.array([grid_term(x, eta) for x in sizes_eff])
    num = ks - eps[ia] - eps[ib]
    return float(np.max(np.maximum(num, 0.0) / D)) if len(D) else 0.0


def true_cdf_on_grid(sigma, lab, K, grid):
    """F_A(r_g) = average over cluster members of the exact conditional CDF 2 Phi(r/sigma_i) - 1."""
    out = np.empty((K, len(grid)))
    for c in range(K):
        idx = np.where(lab == c)[0]
        out[c] = (2.0 * ndtr(grid[None, :] / sigma[idx][:, None]) - 1.0).mean(axis=0)
    return out


def emp_cdf_on_grid(r, lab, K, grid):
    keep = lab >= 0
    sizes = np.array([np.sum(lab == c) for c in range(K)], dtype=float)
    out = np.empty((K, len(grid)))
    for c in range(K):
        out[c] = (r[lab == c][:, None] <= grid[None, :]).mean(axis=0)
    return out


# ----------------------------------------------------------------------------- one replication

def one_rep(seed_seq, n, rho, d, kappa, k_clusters, with_fixed_b=True):
    ss = np.random.SeedSequence(seed_seq)
    child = ss.spawn(3)
    s, sigma, r = simulate(child[0], n, rho, d, kappa)
    lp = l_path(s, sigma)
    lab, K = clusters(s, k_clusters, int(child[1].generate_state(1)[0] % (2 ** 31)))
    row = dict(n=n, rho=rho, d=d, kappa=kappa, k_target=k_clusters, J=K, L_path=lp)
    r_lag1 = float(np.corrcoef(r[1:], r[:-1])[0, 1])
    row["lag1_R"] = r_lag1
    if K < 2:
        row.update(L_cl=np.nan)
        return row
    ia, ib = pair_index(K)
    D, groups = mean_dist(s, lab, K, ia, ib)
    lcl, ks_true = true_cluster_ratio(sigma, groups, ia, ib, D)
    ks_ex, sizes = exact_ks(r, lab, K, ia, ib)
    row["L_cl"] = lcl
    row["frac_pairs_true_ks_pos"] = float(np.mean(ks_true > 1e-9))
    # C0, C0b, C1
    row["C0"] = lower_bound_dkw(ks_ex, D, sizes, ia, ib, 0.05)
    row["C0b"] = lower_bound_dkw(ks_ex, D, sizes, ia, ib, 0.05 / K)
    pit = pit_series(r, lab, K)
    vs = [(pit <= q).astype(float) - q for q in (0.25, 0.5, 0.75)]
    tau = tau_hat(vs)
    row["tau_hat"] = tau
    row["lag1_pit"] = float(np.corrcoef(pit[1:], pit[:-1])[0, 1])
    row["C1"] = lower_bound_dkw(ks_ex, D, sizes / tau, ia, ib, 0.05)
    row["C1b"] = lower_bound_dkw(ks_ex, D, sizes / tau, ia, ib, 0.05 / K)
    # bootstrap versions
    rng = np.random.default_rng(child[2])
    b_fixed = max(2, int(round(n ** (1.0 / 3.0))))
    b_pw = pw_block_length(vs, n)
    row["b_fixed"] = b_fixed
    row["b_pw"] = b_pw
    ks_hat, sup1, grid, Fg = bootstrap_deltas(r, lab, K, ia, ib, b_fixed, rng)
    row["B1"], c1 = lower_bound_bootstrap(ks_hat, D, sup1, False)
    ks_hat, sup2, grid, Fg = bootstrap_deltas(r, lab, K, ia, ib, b_pw, rng)
    row["B2"], c2 = lower_bound_bootstrap(ks_hat, D, sup2, False)
    row["B3"], c3 = lower_bound_bootstrap(ks_hat, D, sup2, True)
    # DIRECT CALIBRATION CHECK: the true centred sup-norm differences of the empirical CDFs against the critical values.
    # ev_* = 1 if sup_g |Delta_p| <= c_p for every retained pair p (nominal probability 0.95).
    Ft = true_cdf_on_grid(sigma, lab, K, grid)
    dlt = (Fg - Ft)
    supD = np.abs(dlt[ia] - dlt[ib]).max(axis=1)
    row["ev_B1"] = float(np.all(supD <= c1))
    row["ev_B2"] = float(np.all(supD <= c2))
    row["ev_B3"] = float(np.all(supD <= c3))
    row["supD_max"] = float(supD.max())
    row["c_B2"] = float(c2[0])
    # independence-based terms, on a fine grid of pooled-sample quantiles
    gf = np.quantile(r[lab >= 0], np.arange(1, G_FINE + 1) / (G_FINE + 1.0))
    dl2 = emp_cdf_on_grid(r, lab, K, gf) - true_cdf_on_grid(sigma, lab, K, gf)
    supF = np.abs(dl2[ia] - dl2[ib]).max(axis=1)
    for name, neff, eta in (("C0", sizes, 0.05), ("C0b", sizes, 0.05 / K), ("C1", sizes / tau, 0.05), ("C1b", sizes / tau, 0.05 / K)):
        eps = np.array([grid_term(x, eta) for x in neff])
        row["ev_" + name] = float(np.all(supF <= eps[ia] + eps[ib]))
    row["KShat_max_ratio"] = float(np.max(ks_hat / D))   # uncorrected plug-in, for reference
    return row
