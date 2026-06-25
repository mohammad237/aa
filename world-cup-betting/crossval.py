"""
Expanding-window walk-forward cross-validation.

A single train/test split (calibrate.py) judges one set of parameters on one
held-out period — better than in-sample, but still one roll of the dice. The
professional standard for time-series strategies is *expanding-window* (a.k.a.
anchored) cross-validation:

    fold 1:  train on [0 .. t1)         then bet [t1 .. t2)   out-of-sample
    fold 2:  train on [0 .. t2)         then bet [t2 .. t3)   out-of-sample
    fold 3:  train on [0 .. t3)         then bet [t3 .. t4)   out-of-sample
    ...

Each fold re-picks parameters using only the past, then is judged on the next
unseen block. We then pool every out-of-sample bet across all folds into one
honest performance figure. This mirrors live use: you periodically recalibrate
on everything so far, then bet forward.

    python3 crossval.py                       # history.csv + styles.csv
    python3 crossval.py history.csv styles.csv 5    # 5 test folds
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from walkforward import run, load_history
from calibrate import BLENDS, EDGES, MIN_BETS_TRAIN


INITIAL_TRAIN_FRAC = 0.40   # first chunk is train-only (need history to learn)


def best_on_train(history_path: str, styles_path: str,
                  train_end: int) -> Tuple[float, float, dict]:
    """Grid-search on [0, train_end) and return the best (blend, edge, metrics)."""
    best = None
    for blend in BLENDS:
        for edge in EDGES:
            m = run(history_path, styles_path, edge_threshold=edge,
                    market_blend=blend, eval_start=0, eval_end=train_end,
                    quiet=True)
            if m["n_bets"] < MIN_BETS_TRAIN:
                continue
            if best is None or m["roi"] > best[2]["roi"]:
                best = (blend, edge, m)
    # Fallback if nothing bet enough: defer hard to the market.
    return best or (0.6, 0.08, {"roi": 0.0, "n_bets": 0})


def main() -> None:
    history_path = sys.argv[1] if len(sys.argv) > 1 else "history.csv"
    styles_path = sys.argv[2] if len(sys.argv) > 2 else "styles.csv"
    n_folds = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    n = len(load_history(history_path))
    start = int(n * INITIAL_TRAIN_FRAC)
    if start >= n - n_folds:
        print("Not enough matches for that many folds.")
        return
    # Fold boundaries across the remaining timeline.
    bounds = [start + round((n - start) * k / n_folds) for k in range(n_folds + 1)]

    print(f"History: {n} matches.  Initial train: first {start}.  "
          f"{n_folds} expanding-window test folds.\n")
    print(f"{'fold':>4}{'train<':>8}{'test':>12}{'blend':>7}{'edge':>6}"
          f"{'bets':>6}{'ROI':>8}{'CLV':>8}")
    print("-" * 59)

    tot_bets = tot_wins = tot_losses = 0
    tot_profit = 0.0
    clv_weighted = 0.0
    clv_total = 0
    clv_beat_weighted = 0.0

    for k in range(n_folds):
        t0, t1 = bounds[k], bounds[k + 1]
        blend, edge, _ = best_on_train(history_path, styles_path, t0)
        m = run(history_path, styles_path, edge_threshold=edge,
                market_blend=blend, eval_start=t0, eval_end=t1, quiet=True)

        bets = m["n_bets"]
        profit = m["roi"] * bets            # flat 1u stakes => staked == bets
        tot_bets += bets
        tot_wins += m["wins"]
        tot_losses += m["losses"]
        tot_profit += profit
        if m["clv"] is not None:
            clv_weighted += m["clv"] * m["clv_n"]
            clv_beat_weighted += m["clv_beat_rate"] * m["clv_n"]
            clv_total += m["clv_n"]

        clv_str = f"{m['clv']:+.1%}" if m["clv"] is not None else "  -  "
        print(f"{k+1:>4}{t0:>8}{f'[{t0},{t1})':>12}{blend:>7.1f}{edge:>6.2f}"
              f"{bets:>6}{m['roi']:>+8.1%}{clv_str:>8}")

    print("-" * 59)
    settled = tot_wins + tot_losses
    agg_roi = tot_profit / tot_bets if tot_bets else 0.0
    agg_hit = tot_wins / settled if settled else 0.0
    agg_clv = clv_weighted / clv_total if clv_total else None
    agg_beat = clv_beat_weighted / clv_total if clv_total else None

    print("\n" + "=" * 59)
    print("  POOLED OUT-OF-SAMPLE RESULT (all folds, never fit on)")
    print("=" * 59)
    print(f"  Bets        : {tot_bets}  (W {tot_wins} / L {tot_losses})")
    print(f"  Hit rate    : {agg_hit:.1%}")
    print(f"  Flat ROI    : {agg_roi:+.1%}  ({tot_profit:+.1f} units)")
    if agg_clv is not None:
        print(f"  CLV         : {agg_clv:+.2%} avg, beat close "
              f"{agg_beat:.0%} of the time")
    print()
    if tot_bets < 15:
        print("  Too few pooled bets to conclude — gather more history.")
    elif agg_roi > 0 and (agg_clv or 0) > 0:
        print("  Positive ROI AND positive CLV across folds is the strongest")
        print("  evidence this pipeline can find a real edge in this data.")
    elif (agg_clv or 0) > 0:
        print("  ROI is shaky but CLV is positive — the bet selection is")
        print("  sound; ROI variance should converge with more bets.")
    else:
        print("  No out-of-sample edge here. The honest conclusion: don't bet")
        print("  this market/data with these features.")


if __name__ == "__main__":
    main()
