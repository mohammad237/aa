"""
Run the whole pipeline end to end — one command to see everything work.

    python3 run_all.py

Steps, in order:
  1. self-test            (tests.py)
  2. generate sample data (make_sample_history.py -> history.csv)
  3. single-match demo    (example.py)
  4. matchday + parlays   (batch.py)
  5. tournament odds      (simulate.py, small run)
  6. naive walk-forward   (loses — the cautionary baseline)
  7. calibrated walk-fwd  (with CLV)
  8. train/test calibrate (calibrate.py)
  9. cross-validation     (crossval.py)

It's just a convenience wrapper; each script also runs on its own. Use this to
confirm a fresh checkout works before swapping in real data.
"""

import subprocess
import sys

PY = sys.executable


def step(title: str, cmd: list, capture_tail: int = 0) -> None:
    print("\n" + "#" * 64)
    print(f"#  {title}")
    print("#" * 64)
    if capture_tail:
        out = subprocess.run(cmd, capture_output=True, text=True)
        lines = (out.stdout + out.stderr).splitlines()
        print("\n".join(lines[-capture_tail:]))
    else:
        subprocess.run(cmd)


def main() -> None:
    step("1/9  Self-test", [PY, "tests.py"], capture_tail=4)

    print("\n" + "#" * 64)
    print("#  2/9  Generate synthetic sample history -> history.csv")
    print("#" * 64)
    with open("history.csv", "w") as fh:
        subprocess.run([PY, "make_sample_history.py"], stdout=fh)
    print("  wrote history.csv")

    step("3/9  Single-match demo", [PY, "example.py"], capture_tail=12)
    step("4/9  Matchday + parlays", [PY, "batch.py"], capture_tail=14)
    step("5/9  Tournament odds (3,000 sims)",
         [PY, "simulate.py", "3000"], capture_tail=12)
    step("6/9  Walk-forward — NAIVE (no blending: the losing baseline)",
         [PY, "walkforward.py", "--blend", "0", "--edge", "0.03"],
         capture_tail=11)
    step("7/9  Walk-forward — CALIBRATED (blend 0.6, edge 0.08) + CLV",
         [PY, "walkforward.py", "--blend", "0.6", "--edge", "0.08"],
         capture_tail=14)
    step("8/9  Calibration with train/test split",
         [PY, "calibrate.py"], capture_tail=10)
    step("9/9  Expanding-window cross-validation",
         [PY, "crossval.py"], capture_tail=14)

    print("\n" + "=" * 64)
    print("  DONE. Everything above ran on SYNTHETIC data with a planted edge.")
    print("  Swap in real results+odds (fetch_data.py) before betting anything.")
    print("=" * 64)


if __name__ == "__main__":
    main()
