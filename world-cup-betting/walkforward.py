"""
Walk-forward backtest — the honest one, with no look-ahead.

Processes a dated match history in chronological order. For each match it:

  1. builds each team's rating/form snapshot from games played *earlier only*
     (via the Elo engine in ratings.py),
  2. runs the model and places any +EV value bets at that match's odds,
  3. settles them against the actual result,
  4. THEN feeds the result into the engine to update ratings.

Because step 1 never sees the current (or any future) result, the ROI here is a
genuine out-of-sample test of the strategy — the thing the static backtest could
not give you.

Usage:
    python3 walkforward.py                       # history.csv + styles.csv
    python3 walkforward.py history.csv styles.csv
    python3 walkforward.py history.csv styles.csv --min-games 5 --warmup 30
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Optional

from ratings import EloEngine, EloConfig
from model import analyze_match
from backtest import settle
from data_io import ODDS_COLUMNS


def load_styles(path: str) -> Dict[str, str]:
    styles: Dict[str, str] = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                nm = (row.get("name") or "").strip()
                if nm:
                    styles[nm] = (row.get("style") or "balanced").strip()
    except FileNotFoundError:
        pass
    return styles


def load_history(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    # Chronological order (date is ISO so a string sort is fine).
    rows.sort(key=lambda r: r.get("date", ""))
    return rows


def _odds(row: dict) -> Dict[str, float]:
    book: Dict[str, float] = {}
    for col, market in ODDS_COLUMNS.items():
        v = row.get(col, "")
        if v not in (None, "") and str(v).strip():
            try:
                book[market] = float(v)
            except ValueError:
                pass
    return book


def run(history_path: str, styles_path: str,
        min_games: int = 4, warmup: int = 0,
        edge_threshold: float = 0.03, market_blend: float = 0.0,
        eval_start: int = 0, eval_end: Optional[int] = None,
        start_bankroll: float = 100.0, quiet: bool = False) -> dict:
    styles = load_styles(styles_path)
    engine = EloEngine(EloConfig(), styles=styles)
    history = load_history(history_path)
    if eval_end is None:
        eval_end = len(history)

    n_bets = wins = losses = pushes = 0
    skipped_warmup = 0
    staked = profit = 0.0
    bankroll = start_bankroll
    brier_sum = 0.0
    brier_n = 0
    buckets: Dict[int, List[int]] = defaultdict(list)
    by_market: Dict[str, List[float]] = defaultdict(list)

    for idx, row in enumerate(history):
        home, away = row["home"].strip(), row["away"].strip()
        try:
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
        except (KeyError, ValueError):
            continue
        neutral = str(row.get("neutral", "1")).strip() in ("1", "true", "yes", "")
        importance = float(row.get("importance", 1.0) or 1.0)

        in_window = eval_start <= idx < eval_end
        enough = (in_window and
                  engine.games_played(home) >= min_games and
                  engine.games_played(away) >= min_games and idx >= warmup)
        book = _odds(row)

        if enough and book:
            home_t = engine.as_of_team(home)
            away_t = engine.as_of_team(away)
            _, bets = analyze_match(home_t, away_t, book_odds=book,
                                    neutral=neutral,
                                    edge_threshold=edge_threshold,
                                    market_blend=market_blend)
            for b in bets:
                pl = settle(b.market, hg, ag, b.book_odds)
                if pl is None:
                    continue
                n_bets += 1
                staked += 1.0
                profit += pl
                bankroll += bankroll * b.kelly_stake * pl
                by_market[b.market].append(pl)
                if pl > 1e-9:
                    wins += 1
                elif pl < -1e-9:
                    losses += 1
                else:
                    pushes += 1
                if abs(pl) > 1e-9:
                    outcome = 1 if pl > 0 else 0
                    brier_sum += (b.model_prob - outcome) ** 2
                    brier_n += 1
                    buckets[int(min(0.999, b.model_prob) * 10)].append(outcome)
        elif book and in_window:
            skipped_warmup += 1

        # Update ratings AFTER betting (no look-ahead).
        engine.update(home, away, hg, ag, neutral=neutral, importance=importance)

    settled = wins + losses
    metrics = {
        "n_bets": n_bets,
        "wins": wins, "losses": losses, "pushes": pushes,
        "roi": (profit / staked) if staked else 0.0,
        "hit": (wins / settled) if settled else 0.0,
        "bankroll": bankroll,
        "brier": (brier_sum / brier_n) if brier_n else None,
    }
    if quiet:
        return metrics

    # ----- Report ----- #
    print("=" * 60)
    print("  WALK-FORWARD BACKTEST (out-of-sample, no look-ahead)")
    print("=" * 60)
    print(f"  Matches processed : {len(history)}")
    print(f"  Skipped (warm-up) : {skipped_warmup} fixtures had odds but too "
          f"little history")
    if n_bets == 0:
        print("  No value bets triggered once warm-up is excluded.")
    else:
        roi = metrics["roi"]
        hit = metrics["hit"]
        print(f"  Value bets placed : {n_bets}  "
              f"(W {wins} / L {losses} / Push {pushes})")
        print(f"  Hit rate          : {hit:.1%}")
        print(f"  Flat-stake P/L    : {profit:+.2f} on {staked:.0f} staked")
        print(f"  Flat-stake ROI    : {roi:+.1%}")
        print(f"  Kelly bankroll    : {start_bankroll:.0f} -> {bankroll:.2f}  "
              f"({bankroll/start_bankroll-1:+.1%})")
        if brier_n:
            print(f"  Brier score       : {brier_sum/brier_n:.4f}  "
                  f"(0.25 = coin-flip; lower is better)")

        print("\n  ROI by market:")
        for mkt, pls in sorted(by_market.items(),
                               key=lambda kv: sum(kv[1]) / len(kv[1]),
                               reverse=True):
            r = sum(pls) / len(pls)
            print(f"    {mkt:<10} n={len(pls):>3}  ROI {r:+6.1%}")

        print("\n  Calibration (predicted vs actual win-rate):")
        for b in range(10):
            outs = buckets.get(b, [])
            if not outs:
                continue
            lo, hi = b * 10, b * 10 + 10
            print(f"    {lo:>2}-{hi:<3}%  n={len(outs):>3}  "
                  f"predicted {(lo+hi)/2/100:>4.0%}  "
                  f"actual {sum(outs)/len(outs):>4.0%}")

    # Final learned ratings — should track the hidden true strengths.
    print("\n  Final learned Elo (top of the table):")
    for nm, elo, games in engine.ratings_table()[:8]:
        print(f"    {nm:<13} {elo:7.0f}   ({games} games)")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("history", nargs="?", default="history.csv")
    ap.add_argument("styles", nargs="?", default="styles.csv")
    ap.add_argument("--min-games", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--edge", type=float, default=0.03,
                    help="minimum edge to place a bet")
    ap.add_argument("--blend", type=float, default=0.0,
                    help="market-blend weight [0..1]; shrink model toward "
                         "the de-vigged market price")
    args = ap.parse_args()
    run(args.history, args.styles, min_games=args.min_games,
        warmup=args.warmup, edge_threshold=args.edge, market_blend=args.blend)
