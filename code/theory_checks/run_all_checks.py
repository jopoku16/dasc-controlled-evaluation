"""Run every theory check in order and keep one log.  python -X utf8 run_all_checks.py"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "check_perstep_bound.py",
    "check_longrun.py",
    "check_grid_lemma.py",
    "check_mixing_bound.py",
    "check_drift_lemma.py",
    "check_assumption_diagnostic.py",
    "check_assumption_diagnostic_bonf.py",
    "check_combination.py",
]

if __name__ == "__main__":
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    with open(out / "_run_all.log", "w", encoding="utf-8") as log:
        for s in SCRIPTS:
            log.write(f"===== {s} =====\n")
            log.flush()
            r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / s)], cwd=HERE,
                               capture_output=True, text=True, encoding="utf-8")
            log.write(r.stdout)
            log.write(r.stderr)
            log.write(f"exit {r.returncode}\n\n")
            print(s, "exit", r.returncode)
