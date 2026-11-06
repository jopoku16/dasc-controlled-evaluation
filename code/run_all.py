"""
run_all.py: reproduce the main results with one command.

    set DASC_DATA_ROOT=<folder containing uci_household_power, open_meteo_dallas, fred_sp500>
    python -X utf8 run_all.py

All seeds are fixed inside the scripts (simulation evaluation seeds 0-99,
development seeds 1000-1009, bootstrap seed 0, gradient boosting seed 0).
Runtime of every stage is printed; the full run takes roughly two hours on
8 cores.  The stages can also be run one at a time with the same commands;
each rewrites only its own rows in the shared result files.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(step, args=()):
    t0 = time.time()
    print(f"\n=== {step} {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(HERE / step), *args], cwd=HERE, check=True)
    dt = time.time() - t0
    print(f"=== {step} finished in {dt / 60:.1f} min ===", flush=True)
    return dt


def main(argv):
    t0 = time.time()
    root = os.environ.get("DASC_DATA_ROOT", str(HERE.parent / "data" / "real"))
    print(f"data root: {root}")
    times = {}
    times["test_causality.py"] = run("test_causality.py")
    times["check_reproduce_original.py"] = run("check_reproduce_original.py")
    times["run_real_data.py"] = run("run_real_data.py")
    times["run_simulations.py"] = run("run_simulations.py")
    times["run_sim_gamma_sensitivity.py"] = run("run_sim_gamma_sensitivity.py")
    times["run_assumption_diagnostic.py"] = run("run_assumption_diagnostic.py")
    times["run_assumption_diagnostic_bonf.py"] = run("run_assumption_diagnostic_bonf.py", ("all",))
    times["run_timing.py"] = run("run_timing.py")
    print("\nStage runtimes (min):")
    for k, v in times.items():
        print(f"  {k:30s} {v / 60:6.1f}")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
