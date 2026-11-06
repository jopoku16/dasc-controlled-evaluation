"""
blockseed_run.py: block-length and bootstrap-seed sensitivity of the paired real-data intervals.

Recomputes, from the stored per-step diagnostics (results/diagnostics/*.parquet), the paired moving-block
bootstrap of DASC (projected, tuned) minus ACI (projected), spectral ACI and decay-weighted conformal prediction,
using metrics.paired_block_bootstrap unchanged (2,000 resamples, percentile 95 per cent intervals).

Designs
  A  main equal budget:      dasc_clip_tg  (<ds>_<fc>_dasc_clip_dascgamma)  against <ds>_<fc>_aci_clip, _spectral_aci, _decay
  B  extended grid, primary: dasc_clip_ext (ext_<ds>_<fc>_dasc_clip_ext)    against ext_..._aci_clip_ext, ext_..._spectral_aci_ext,
                             and the main-run decay (decay has no step size; the extended study re-ran its main-run choice)
  C  extended grid, stage-0 sensitivity design: dasc_clip_wide, same comparators as B (extra, reported separately)

Block lengths (steps): hourly series (electricity, weather) 168, 720, 2160; finance (daily) 21, 63, 126.
Seeds 0 to 8.  Seed 0 at 168 / 21 is the reported configuration and is checked against the stored CSVs.

Modes
  python blockseed_run.py repro   reproduction check only (reported block, seed 0, all designs), tolerance 1e-9
  python blockseed_run.py run     full grid (at most 2 worker processes)

Writes only new files: results/blockseed_repro.csv, results/blockseed_cells.csv (mode run).
Does not modify any existing file.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import paired_block_bootstrap  # noqa: E402

RESULTS = HERE.parent / "results"
DIAG = RESULTS / "diagnostics"

CELLS = [(d, f) for d, fs in [("electricity", ["naive24", "ridge", "xgb"]),
                              ("weather", ["naive24", "ridge", "xgb"]),
                              ("finance", ["naive1", "ridge", "xgb"])] for f in fs]
BLOCKS = {"electricity": [168, 720, 2160], "weather": [168, 720, 2160], "finance": [21, 63, 126]}
BASE_BLOCK = {"electricity": 168, "weather": 168, "finance": 21}
SEEDS = list(range(9))
N_BOOT = 2000
COMPARATORS = ["aci_clip", "spectral_aci", "decay"]


def diag_names(design, ds, fc, comp):
    """File stems (without .parquet) of DASC and comparator for a design."""
    if design == "A":
        a = f"{ds}_{fc}_dasc_clip_dascgamma"
        b = {"aci_clip": f"{ds}_{fc}_aci_clip", "spectral_aci": f"{ds}_{fc}_spectral_aci",
             "decay": f"{ds}_{fc}_decay"}[comp]
    else:
        a = f"ext_{ds}_{fc}_dasc_clip_" + ("ext" if design == "B" else "wide")
        b = {"aci_clip": f"ext_{ds}_{fc}_aci_clip_ext", "spectral_aci": f"ext_{ds}_{fc}_spectral_aci_ext",
             "decay": f"{ds}_{fc}_decay"}[comp]
    return a, b


def load(stem):
    d = pd.read_parquet(DIAG / f"{stem}.parquet")
    out = {k: d[k].to_numpy() for k in ["t", "width", "score", "miss"]}
    return out, d["in_eval"].to_numpy().astype(bool)


def stored_reference():
    """Stored paired-bootstrap rows for the reported configuration, keyed by (design, ds, fc, comp)."""
    ref = {}
    A = pd.read_csv(RESULTS / "real_paired_bootstrap_dascgamma.csv")
    A = A[A.method_a == "dasc_clip_tg"]
    for _, r in A.iterrows():
        if r.method_b in COMPARATORS:
            ref[("A", r.dataset, r.forecaster, r.method_b)] = r
    E = pd.read_csv(RESULTS / "ext_real_paired_bootstrap.csv")
    mapb = {"aci_clip_ext": "aci_clip", "spectral_aci_ext": "spectral_aci", "decay": "decay"}
    for design, lab in [("B", "dasc_clip_ext"), ("C", "dasc_clip_wide")]:
        sub = E[E.method_a == lab]
        for _, r in sub.iterrows():
            if r.method_b in mapb:
                ref[(design, r.dataset, r.forecaster, mapb[r.method_b])] = r
    return ref


FIELDS = ["n", "n_masked", "block_len", "n_boot", "width_diff", "width_diff_lo", "width_diff_hi",
          "coverage_diff", "coverage_diff_lo", "coverage_diff_hi", "score_diff", "score_diff_lo", "score_diff_hi",
          "rel_width_diff", "rel_width_diff_lo", "rel_width_diff_hi"]


def one_cell_design(args):
    """All comparators, block lengths and seeds for one (design, cell)."""
    design, ds, fc, mode = args
    rows, repro = [], []
    ref = stored_reference() if mode == "repro" or True else {}
    a_stem, _ = diag_names(design, ds, fc, "aci_clip")
    A, ev_a = load(a_stem)
    for comp in COMPARATORS:
        _, b_stem = diag_names(design, ds, fc, comp)
        B, ev_b = load(b_stem)
        assert np.array_equal(ev_a, ev_b) and np.array_equal(A["t"], B["t"]), (design, ds, fc, comp, "misaligned")
        wa = A["width"][ev_a]
        wb_ = B["width"][ev_a]
        cov_a = 1 - A["miss"][ev_a].mean()
        cov_b = 1 - B["miss"][ev_a].mean()
        fin_w_a = float(np.mean(wa[np.isfinite(wa)]))
        fin_w_b = float(np.mean(wb_[np.isfinite(wb_)]))
        sa, sb = A["score"][ev_a], B["score"][ev_a]
        fin_s_a = float(np.mean(sa[np.isfinite(sa)]))      # mean interval score over finite evaluation steps
        fin_s_b = float(np.mean(sb[np.isfinite(sb)]))
        blocks = [BASE_BLOCK[ds]] if mode == "repro" else BLOCKS[ds]
        seeds = [0] if mode == "repro" else SEEDS
        for bl in blocks:
            for sd in seeds:
                r = paired_block_bootstrap(A, B, bl, n_boot=N_BOOT, seed=sd, mask=ev_a)
                row = {"design": design, "dataset": ds, "forecaster": fc, "comparator": comp,
                       "dasc_file": a_stem, "comp_file": b_stem, "block_len": bl,
                       "block_rank": BLOCKS[ds].index(bl) + 1, "seed": sd,
                       "reported": bool(bl == BASE_BLOCK[ds] and sd == 0),
                       "coverage_dasc": cov_a, "coverage_comp": cov_b, "width_dasc": fin_w_a, "width_comp": fin_w_b,
                       "score_dasc": fin_s_a, "score_comp": fin_s_b}
                row.update({k: r[k] for k in FIELDS})
                rows.append(row)
                if bl == BASE_BLOCK[ds] and sd == 0:
                    sref = ref[(design, ds, fc, comp)]
                    diffs = {k: abs(float(r[k]) - float(sref[k])) for k in FIELDS if k not in ("n", "n_masked", "block_len", "n_boot")}
                    diffs["n_equal"] = float(int(r["n"]) != int(sref["n"]))
                    diffs["cov_a_diff"] = abs(cov_a - float(sref["coverage_a"]))
                    diffs["cov_b_diff"] = abs(cov_b - float(sref["coverage_b"]))
                    diffs["width_a_diff"] = abs(fin_w_a - float(sref["width_a"]))
                    diffs["width_b_diff"] = abs(fin_w_b - float(sref["width_b"]))
                    repro.append({"design": design, "dataset": ds, "forecaster": fc, "comparator": comp,
                                  "block_len": bl, "seed": 0, "max_abs_diff": max(diffs.values()),
                                  "worst_field": max(diffs, key=diffs.get), "pass_1e-9": bool(max(diffs.values()) < 1e-9)})
    return rows, repro


def verdict(lo, hi, pos="against_DASC", neg="favours_DASC"):
    """Interval excludes zero below (neg) or above (pos); otherwise includes zero."""
    return np.where(hi < 0, neg, np.where(lo > 0, pos, "includes_zero"))


def add_verdicts(df):
    """Width and score: negative difference favours DASC (narrower, lower score). Coverage: positive favours DASC."""
    df = df.copy()
    df["rel_width_pct"] = 100 * df["rel_width_diff"]
    df["rel_width_lo_pct"] = 100 * df["rel_width_diff_lo"]
    df["rel_width_hi_pct"] = 100 * df["rel_width_diff_hi"]
    df["rel_score_pct"] = 100 * df["score_diff"] / df["score_comp"]      # score difference in per cent of the comparator's score
    df["rel_score_lo_pct"] = 100 * df["score_diff_lo"] / df["score_comp"]
    df["rel_score_hi_pct"] = 100 * df["score_diff_hi"] / df["score_comp"]
    df["coverage_diff_pts"] = 100 * df["coverage_diff"]
    df["coverage_lo_pts"] = 100 * df["coverage_diff_lo"]
    df["coverage_hi_pts"] = 100 * df["coverage_diff_hi"]
    df["width_verdict"] = verdict(df["rel_width_diff_lo"].to_numpy(), df["rel_width_diff_hi"].to_numpy())
    df["score_verdict"] = verdict(df["score_diff_lo"].to_numpy(), df["score_diff_hi"].to_numpy())
    df["coverage_verdict"] = verdict(df["coverage_diff_lo"].to_numpy(), df["coverage_diff_hi"].to_numpy(),
                                     pos="DASC_covers_more", neg="DASC_covers_less")
    key = ["design", "dataset", "forecaster", "comparator"]
    rep = df[df["reported"]].set_index(key)
    for m in ("width", "score", "coverage"):
        col = f"{m}_verdict"
        df[f"{m}_verdict_reported"] = [rep.loc[tuple(r), col] for r in df[key].to_numpy()]
        df[f"{m}_differs_from_reported"] = df[col] != df[f"{m}_verdict_reported"]
    # source of variation relative to the reported configuration
    bb = df["dataset"].map(BASE_BLOCK)
    df["variation"] = np.where(df["reported"], "reported",
                       np.where(df["block_len"] == bb, "seed_only",
                       np.where(df["seed"] == 0, "block_only", "block_and_seed")))
    return df


def main(argv):
    mode = argv[0] if argv else "repro"
    designs = ["A", "B", "C"]
    tasks = [(d, ds, fc, mode) for d in designs for ds, fc in CELLS]
    t0 = time.time()
    all_rows, all_repro = [], []
    with Pool(2) as pool:
        for k, (rows, repro) in enumerate(pool.imap_unordered(one_cell_design, tasks)):
            all_rows += rows
            all_repro += repro
            print(f"[{k + 1}/{len(tasks)}] {rows[0]['design']} {rows[0]['dataset']}/{rows[0]['forecaster']} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    rp = pd.DataFrame(all_repro).sort_values(["design", "dataset", "forecaster", "comparator"])
    rp.to_csv(RESULTS / ("blockseed_repro.csv" if mode == "run" else "blockseed_repro_only.csv"), index=False)
    print("reproduction: rows", len(rp), "all pass 1e-9:", bool(rp["pass_1e-9"].all()),
          "max abs diff", rp.max_abs_diff.max(), flush=True)
    if mode == "run":
        df = pd.DataFrame(all_rows).sort_values(["design", "dataset", "forecaster", "comparator", "block_rank", "seed"])
        df = add_verdicts(df)
        df.to_csv(RESULTS / "blockseed_cells.csv", index=False)
        print("rows", len(df), flush=True)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
