"""
lead1_run.py: driver for the lead-1 kill test.  At most 2 worker processes, one thread each.
Writes ../results/lead1_reps_kappa<kappa>.csv (one row per replication, all estimators paired on the same data).
Resumable: replications already in the file are skipped.

Run:  python -X utf8 lead1_run.py --reps 500 --kappa 1.0 --dims 1 2
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
NS = {360: 6, 2160: 20}          # n -> number of k-means clusters (mean size 60 and 108)
RHOS = (0.0, 0.5, 0.9)


def task(args):
    import lead1_sim as L
    n, rho, d, kappa, rep = args
    seed = [n, int(round(rho * 10)), d, int(round(kappa * 100)), rep, 20261102]
    try:
        row = L.one_rep(seed, n, rho, d, kappa, NS[n])
    except Exception as e:  # keep the run alive, record the failure
        row = dict(n=n, rho=rho, d=d, kappa=kappa, error=repr(e))
    row["rep"] = rep
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--ns", type=int, nargs="+", default=[360, 2160])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rhos", type=float, nargs="+", default=list(RHOS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = RESULTS / f"lead1_reps_kappa{a.kappa}{a.tag}.csv"
    done = set()
    fields = None
    if out.exists():
        import pandas as pd
        old = pd.read_csv(out)
        fields = list(old.columns)
        done = set(zip(old.n, old.rho, old.d, old.rep))
    tasks = [(n, rho, d, a.kappa, rep) for rep in range(a.reps) for n in a.ns for rho in a.rhos for d in a.dims
             if (n, rho, d, rep) not in done]
    # cheap cells first within a replication index so partial results cover all cells evenly
    print(f"{len(tasks)} tasks, workers={a.workers}", flush=True)
    t0 = time.time()
    with Pool(min(2, a.workers)) as pool:
        buf = []
        for i, row in enumerate(pool.imap_unordered(task, tasks, chunksize=2)):
            if "error" in row:
                print("ERROR", row, flush=True)
                continue
            buf.append(row)
            if len(buf) >= 20 or i == len(tasks) - 1:
                if fields is None:
                    fields = list(buf[0].keys())
                new = out.exists()
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    if not new:
                        w.writeheader()
                    w.writerows(buf)
                buf = []
                print(f"{i + 1}/{len(tasks)} done, {time.time() - t0:.0f} s", flush=True)
    print("finished", time.time() - t0)


if __name__ == "__main__":
    main()
