"""lead4_check_rolling.py: the vectorised rolling-window ACI of lead4_link.py reproduces
dasc_lib.ACI (free update, window 360, first_valid 1) step for step on a block-structured
residual stream.  Read-only import of dasc_lib."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lead4_link as L
from dasc_lib import ACI, run_stream

rng = np.random.default_rng(5)
m, W, gamma, rho = 150, 360, 0.05, 3.0
T = 6000
sig = np.where(((np.arange(T) // m) % 2) == 0, 1.0, rho)
R = np.abs(rng.standard_normal((1, T))) * sig[None, :]
t0 = 600
mine = L.rolling_aci_mc(rho, gamma, m, W=W, R_override=R, t0_override=t0, burn_override=0, record=True)[0, t0:]
y = R[0] * np.where(rng.random(T) < 0.5, -1.0, 1.0)
pred = np.zeros(T)
out = run_stream(y, pred, ACI(alpha=0.1, window=W, first_valid=1, gamma=gamma, clip=False), start=t0)
theirs = out["miss"].astype(float)
print("steps compared:", len(theirs), " identical miss sequences:", bool(np.array_equal(mine, theirs)),
      " mismatches:", int((mine != theirs).sum()))
