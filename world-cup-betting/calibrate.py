"""
Calibrate the strategy on historical data.

The raw model is overconfident — left alone it bets its own noise against a sharp
market and loses (run walkforward.py with --blend 0 to see it). Professionals fix
this by (a) shrinking model probabilities toward the market and (b) demanding a
real edge before staking. This script grid-searches those two levers on the
walk-forward (out-of-sample) backtest and reports the settings that would have
performed best, so the knobs are chosen by data, not by feel.

    python3 calibrate.py                      # history.csv + styles.csv
    python3 calibrate.py history.csv styles.csv

It optimizes for ROI subject to a minimum sample of bets (so it doesn't "win" by
making one lucky wager). Always re-validate on data the calibrator did not see —
in-sample best is an upper bound, not a promise.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from walkforward import run


BLENDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
EDGES = [0.02, 0.04, 0.06, 0.08, 0.10]
MIN_BETS = 25          # ignore settings that barely bet (unreliable ROI)


def main() -> None:
    history = sys.argv[1] if len(sys.argv) > 1 else "history.csv"
    styles = sys.argv[2] if len(sys.argv) > 2 else "styles.csv"

    print(f"Grid-searching blend x edge on {history} "
          f"(out-of-sample walk-forward)...\n")
    print(f"{'blend':>6}{'edge':>7}{'bets':>7}{'hit':>7}{'ROI':>9}"
          f"{'bankroll':>10}{'Brier':>9}")
    print("-" * 55)

    results: List[Tuple[float, float, dict]] = []
    for blend in BLENDS:
        for edge in EDGES:
            m = run(history, styles, edge_threshold=edge, market_blend=blend,
                    quiet=True)
            results.append((blend, edge, m))
            brier = f"{m['brier']:.3f}" if m["brier"] is not None else "  -  "
            flag = "" if m["n_bets"] >= MIN_BETS else "  (thin)"
            print(f"{blend:>6.1f}{edge:>7.2f}{m['n_bets']:>7}"
                  f"{m['hit']:>7.1%}{m['roi']:>9.1%}"
                  f"{m['bankroll']:>10.1f}{brier:>9}{flag}")

    eligible = [r for r in results if r[2]["n_bets"] >= MIN_BETS]
    if not eligible:
        print("\nNo setting placed enough bets to judge. Need more data.")
        return
    best = max(eligible, key=lambda r: r[2]["roi"])
    blend, edge, m = best
    print("\n" + "=" * 55)
    print("  BEST OUT-OF-SAMPLE SETTING")
    print("=" * 55)
    print(f"  market_blend   : {blend:.1f}")
    print(f"  edge_threshold : {edge:.2f}")
    print(f"  -> {m['n_bets']} bets, hit {m['hit']:.1%}, "
          f"ROI {m['roi']:+.1%}, bankroll -> {m['bankroll']:.1f}")
    print("\n  Use these in analyze_match(... edge_threshold=, market_blend=).")
    print("  Re-validate on fresh data before trusting them.")


if __name__ == "__main__":
    main()
