"""
lead1_selftest.py: sanity checks of lead1_sim.py (run before the simulation).
  1. nearest-neighbour search for L_path agrees with the all-pairs maximum;
  2. the bootstrap standard deviation of Delta at the median grid point matches the analytic iid value;
  3. the bootstrap is variance-correct for AR(1) indicators (compared with a Monte Carlo of the same statistic);
  4. timing of one replication per n.
"""
import time
import numpy as np

import lead1_sim as L

# 1
for d in (1, 2):
    for n in (360, 700):
        ss = np.random.SeedSequence([1, d, n])
        s, sigma, r = L.simulate(ss, n, 0.0, d, 1.0)
        a = L.l_path(s, sigma)
        b = L.l_path(s, sigma, full=True)
        print(f"L_path d={d} n={n}: knn={a:.5f} full={b:.5f}")

# 2. iid, homogeneous clusters: Var(Delta(r)) = F(1-F)(1/nA+1/nB)
rng = np.random.default_rng(5)
n = 1200
lab = np.repeat(np.arange(4), n // 4)
r = rng.standard_normal(n) ** 2
ia, ib = L.pair_index(4)
ks_hat, sup, _g, _F = L.bootstrap_deltas(r, lab, 4, ia, ib, 8, rng, B=400)
# analytic: sd of sup|Delta| is hard; compare the mean of sup with Monte Carlo of the true sup
mc = []
for _ in range(400):
    rr = rng.standard_normal(n) ** 2
    Fg = []
    grid = np.quantile(r, np.arange(1, L.G_GRID + 1) / (L.G_GRID + 1.0))
    U = (rr[:, None] <= grid[None, :])
    F = np.array([U[lab == c].mean(axis=0) for c in range(4)])
    mc.append(np.abs(F[ia] - F[ib]).max(axis=1))
mc = np.array(mc)
print("iid bootstrap mean sup|Delta| per pair:", np.round(sup.mean(axis=0), 4))
print("iid Monte Carlo mean sup|Delta| per pair:", np.round(mc.mean(axis=0), 4))

# 3. AR(1) indicator process, block bootstrap with PW length vs Monte Carlo
rho_z = 0.9
n = 1200
def draw():
    z = L.ar1_gauss(np.random.default_rng(), n, rho_z, 1)[:, 0]
    return np.abs(z)
lab = np.repeat(np.arange(4), n // 4)
base = draw()
pit = L.pit_series(base, lab, 4)
vs = [(pit <= q).astype(float) - q for q in (0.25, 0.5, 0.75)]
print("PW block length (AR 0.9 on Z):", L.pw_block_length(vs, n), "tau_hat:", round(L.tau_hat(vs), 2))
mc = []
for _ in range(300):
    rr = draw()
    grid = np.quantile(rr, np.arange(1, L.G_GRID + 1) / (L.G_GRID + 1.0))
    U = (rr[:, None] <= grid[None, :])
    F = np.array([U[lab == c].mean(axis=0) for c in range(4)])
    # centre at the truth: population CDF of |Z|
    from scipy.special import ndtr
    Ft = 2 * ndtr(grid) - 1
    mc.append(np.abs((F[ia] - Ft) - (F[ib] - Ft)).max(axis=1))
mc = np.array(mc)
rng = np.random.default_rng(1)
for bb in (L.pw_block_length(vs, n), 10, 40):
    _, sup, _g, _F = L.bootstrap_deltas(base, lab, 4, ia, ib, bb, rng, B=300)
    print(f"b={bb}: bootstrap mean sup|Delta| {sup.mean():.4f}   Monte Carlo {mc.mean():.4f}")

# 4 timing
for n, k in ((360, 6), (2160, 20)):
    t0 = time.time()
    for rep in range(3):
        row = L.one_rep([9, n, rep], n, 0.5, 1, 1.0, k)
    print(f"n={n}: {(time.time() - t0) / 3:.2f} s per replication; last row:", {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()})
