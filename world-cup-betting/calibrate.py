"""
Calibrate the strategy honestly, with a time-based train/test split.

Fitting parameters and then quoting how they did on the *same* data is the
classic way to fool yourself: you pick the settings that got luckiest, and they
fall apart live. A professional avoids this by choosing parameters on past
("train") data and judging them only on later data they were never fit on
("test").

This script:
  1. splits the dated history chronologically (default first 65% train, rest
     test);
  2. grid-searches market_blend x edge_threshold on the TRAIN window;
  3. picks the best train setting (subject to a minimum bet count);
  4. reports that setting's OUT-OF-SAMPLE performance on the TEST window.

Throughout, ratings keep updating across the whole timeline in date order — only
the *betting* is restricted to the relevant window — so the test period is
evaluated with ratings as they actually stood going into it.

    python3 calibrate.py                      # history.csv + styles.csv
    python3 calibrate.py history.csv styles.csv 0.65
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from walkforward import run, load_history


BLENDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
EDGES = [0.02, 0.04, 0.06, 0.08, 0.10]
MIN_BETS_TRAIN = 20    # a setting must bet enough on train to be trustworthy


def _fmt(m: dict) -> str:
    brier = f"{m['brier']:.3f}" if m["brier"] is not None else "  -  "
    return (f"bets {m['n_bets']:>4}  hit {m['hit']:>5.1%}  "
            f"ROI {m['roi']:>+6.1%}  bankroll {m['bankroll']:>6.1f}  "
            f"Brier {brier}")


def main() -> None:
    history_path = sys.argv[1] if len(sys.argv) > 1 else "history.csv"
    styles_path = sys.argv[2] if len(sys.argv) > 2 else "styles.csv"
    train_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.65

    n = len(load_history(history_path))
    split = int(n * train_frac)
    print(f"History: {n} matches.  Train = first {split} (by date), "
          f"Test = last {n - split}.\n")

    # ---- 1-2. Grid search on TRAIN only ----
    print(f"TRAIN grid search (betting only on matches 0..{split}):")
    print(f"{'blend':>6}{'edge':>7}{'bets':>7}{'hit':>7}{'ROI':>9}{'Brier':>9}")
    print("-" * 45)
    results: List[Tuple[float, float, dict]] = []
    for blend in BLENDS:
        for edge in EDGES:
            m = run(history_path, styles_path, edge_threshold=edge,
                    market_blend=blend, eval_start=0, eval_end=split,
                    quiet=True)
            results.append((blend, edge, m))
            brier = f"{m['brier']:.3f}" if m["brier"] is not None else "  -  "
            flag = "" if m["n_bets"] >= MIN_BETS_TRAIN else "  (thin)"
            print(f"{blend:>6.1f}{edge:>7.2f}{m['n_bets']:>7}{m['hit']:>7.1%}"
                  f"{m['roi']:>+9.1%}{brier:>9}{flag}")

    eligible = [r for r in results if r[2]["n_bets"] >= MIN_BETS_TRAIN]
    if not eligible:
        print("\nNot enough bets on train to choose a setting. Need more data.")
        return

    # ---- 3. Pick best on TRAIN ----
    blend, edge, train_m = max(eligible, key=lambda r: r[2]["roi"])

    # ---- 4. Evaluate that setting OUT-OF-SAMPLE on TEST ----
    test_m = run(history_path, styles_path, edge_threshold=edge,
                 market_blend=blend, eval_start=split, eval_end=n, quiet=True)

    print("\n" + "=" * 60)
    print("  CHOSEN ON TRAIN, JUDGED ON TEST (out-of-sample)")
    print("=" * 60)
    print(f"  Best train setting : market_blend={blend:.1f}, "
          f"edge_threshold={edge:.2f}")
    print(f"  TRAIN  : {_fmt(train_m)}")
    print(f"  TEST   : {_fmt(test_m)}")
    print()
    if test_m["n_bets"] < 10:
        print("  ! Test window produced very few bets — treat as inconclusive.")
    elif test_m["roi"] > 0:
        print("  The edge survived out-of-sample. Promising, but keep")
        print("  re-validating on genuinely new matches before scaling stakes.")
    else:
        print("  The train edge did NOT survive out-of-sample — it was likely")
        print("  overfitting / noise. This is the honest answer: don't bet it.")
    print("\n  Plug the chosen values into analyze_match(edge_threshold=, "
          "market_blend=).")


if __name__ == "__main__":
    main()
