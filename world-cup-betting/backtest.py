"""
Backtest the betting strategy against historical results.

Feed in past matches with their actual scoreline and the odds you could have
taken, and this replays the model: for each fixture it finds the +EV value bets
(exactly what batch.py would have flagged), "places" them at the recorded odds,
settles them against the real result, and reports how the strategy actually did.

Metrics
-------
* ROI (flat stake)   - profit per unit staked, 1 unit on every value bet.
* ROI (Kelly)        - bankroll growth using the model's fractional-Kelly stakes.
* Hit rate           - share of settled bets that won (pushes excluded).
* Brier score        - probability calibration error (lower is better).
* Reliability table  - predicted vs actual win-rate, bucketed.
* Closing-line value  - if the CSV has *_close odds, did we beat the close?
                       (Beating the closing line is the strongest predictor of
                       long-term edge.)

Usage:
    python3 backtest.py                       # teams.csv + results.csv
    python3 backtest.py teams.csv history.csv

results.csv = fixtures.csv columns + actual `home_goals`,`away_goals`, and
optionally closing-odds columns suffixed `_close` (e.g. home_win_close).

IMPORTANT caveat: this uses the *current* teams.csv as the rating snapshot for
every past match, so it carries look-ahead bias. For an honest backtest, supply
ratings/form as they were *before* each match (point-in-time). Treat the numbers
here as a sanity check on the pipeline, not proof of a live edge.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from data_io import load_teams, ODDS_COLUMNS
from model import analyze_match, ValueBet, Team


# --------------------------------------------------------------------------- #
# Settlement: profit per 1 unit staked on a market, given the final score.
# --------------------------------------------------------------------------- #

def _ah_unit(margin: int, line: float) -> float:
    """+1 win / -1 loss / 0 push for a single (non-quarter) handicap line."""
    adjusted = margin + line
    if adjusted > 1e-9:
        return 1.0
    if adjusted < -1e-9:
        return -1.0
    return 0.0


def settle(market: str, hg: int, ag: int, odds: float) -> Optional[float]:
    """
    Net profit for a 1-unit stake on `market` at `odds` given final score hg-ag.
    Returns None if the market name isn't understood.
    """
    def payout(win: bool) -> float:
        return (odds - 1.0) if win else -1.0

    total = hg + ag

    if market == "Home win":
        return payout(hg > ag)
    if market == "Draw":
        return payout(hg == ag)
    if market == "Away win":
        return payout(ag > hg)
    if market == "DC 1X":
        return payout(hg >= ag)
    if market == "DC 12":
        return payout(hg != ag)
    if market == "DC X2":
        return payout(ag >= hg)
    if market == "DNB Home":
        return 0.0 if hg == ag else payout(hg > ag)
    if market == "DNB Away":
        return 0.0 if hg == ag else payout(ag > hg)
    if market == "BTTS Yes":
        return payout(hg >= 1 and ag >= 1)
    if market == "BTTS No":
        return payout(not (hg >= 1 and ag >= 1))
    if market.startswith("Over "):
        return payout(total > float(market.split()[1]))
    if market.startswith("Under "):
        return payout(total < float(market.split()[1]))
    if market.startswith("CS "):
        i, j = market.split()[1].split("-")
        return payout(hg == int(i) and ag == int(j))
    if market.startswith("AH "):
        parts = market.split()
        side, line = parts[1].lower(), float(parts[2])
        margin = (hg - ag) if side == "home" else (ag - hg)
        # Quarter lines split the stake across two neighbouring lines.
        frac = round(abs(line) * 4) % 4
        if frac in (1, 3):
            lo = math.floor(line * 2) / 2
            hi = math.ceil(line * 2) / 2
            profit = 0.0
            for half_line in (lo, hi):
                u = _ah_unit(margin, half_line)
                profit += 0.5 * ((odds - 1.0) if u > 0 else (0.0 if u == 0 else -1.0))
            return profit
        u = _ah_unit(margin, line)
        return (odds - 1.0) if u > 0 else (0.0 if u == 0 else -1.0)
    return None


# --------------------------------------------------------------------------- #
# Loading the results file.
# --------------------------------------------------------------------------- #

class HistoricMatch:
    def __init__(self, home, away, neutral, hg, ag,
                 book_odds, close_odds):
        self.home = home
        self.away = away
        self.neutral = neutral
        self.hg = hg
        self.ag = ag
        self.book_odds: Dict[str, float] = book_odds
        self.close_odds: Dict[str, float] = close_odds


def load_results(path: str) -> List[HistoricMatch]:
    matches: List[HistoricMatch] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            home = (row.get("home") or "").strip()
            away = (row.get("away") or "").strip()
            if not home or not away:
                continue
            try:
                hg = int(row["home_goals"])
                ag = int(row["away_goals"])
            except (KeyError, ValueError):
                continue  # unplayed / missing result
            neutral = str(row.get("neutral", "1")).strip().lower() in (
                "1", "true", "yes", "y", "")
            book, close = {}, {}
            for col, market in ODDS_COLUMNS.items():
                v = row.get(col, "")
                if v not in (None, "") and str(v).strip():
                    try:
                        book[market] = float(v)
                    except ValueError:
                        pass
                vc = row.get(f"{col}_close", "")
                if vc not in (None, "") and str(vc).strip():
                    try:
                        close[market] = float(vc)
                    except ValueError:
                        pass
            matches.append(HistoricMatch(home, away, neutral, hg, ag,
                                         book, close))
    return matches


# --------------------------------------------------------------------------- #
# The backtest.
# --------------------------------------------------------------------------- #

def run(teams_path: str, results_path: str,
        start_bankroll: float = 100.0) -> None:
    teams = load_teams(teams_path)
    matches = load_results(results_path)

    n_bets = wins = losses = pushes = 0
    staked = profit = 0.0
    bankroll = start_bankroll
    brier_sum = 0.0
    brier_n = 0
    clv_sum = 0.0
    clv_n = 0
    # Calibration buckets: 0-10%, 10-20%, ...
    buckets: Dict[int, List[int]] = defaultdict(list)   # bin -> [outcomes 0/1]

    print(f"{'Match':<26}{'Bet':<12}{'Odds':>6}{'Res':>6}{'P/L':>8}")
    print("-" * 58)

    for m in matches:
        if m.home not in teams or m.away not in teams:
            continue
        probs, bets = analyze_match(teams[m.home], teams[m.away],
                                    book_odds=m.book_odds, neutral=m.neutral)
        for b in bets:
            pl = settle(b.market, m.hg, m.ag, b.book_odds)
            if pl is None:
                continue
            n_bets += 1
            staked += 1.0
            profit += pl
            # Kelly bankroll: stake a fraction of the *current* bankroll.
            bankroll += bankroll * b.kelly_stake * (pl)  # pl already per-unit

            if pl > 1e-9:
                wins += 1
                res = "W"
            elif pl < -1e-9:
                losses += 1
                res = "L"
            else:
                pushes += 1
                res = "P"

            # Calibration (skip pushes; treat half-results by sign).
            if abs(pl) > 1e-9:
                outcome = 1 if pl > 0 else 0
                brier_sum += (b.model_prob - outcome) ** 2
                brier_n += 1
                buckets[int(min(0.999, b.model_prob) * 10)].append(outcome)

            # Closing-line value.
            close = m.close_odds.get(b.market)
            if close and close > 1.0:
                clv_sum += (b.book_odds / close - 1.0)
                clv_n += 1

            print(f"{m.home[:11]+' v '+m.away[:11]:<26}{b.market:<12}"
                  f"{b.book_odds:>6.2f}{res:>6}{pl:>+8.2f}")

    # ----- Summary ----- #
    print("\n" + "=" * 58)
    print("  BACKTEST SUMMARY")
    print("=" * 58)
    if n_bets == 0:
        print("  No value bets were triggered on this dataset.")
        return

    settled = wins + losses
    roi = profit / staked if staked else 0.0
    hit = wins / settled if settled else 0.0
    print(f"  Value bets placed : {n_bets}  "
          f"(W {wins} / L {losses} / Push {pushes})")
    print(f"  Hit rate          : {hit:.1%}")
    print(f"  Flat-stake P/L    : {profit:+.2f} units on {staked:.0f} staked")
    print(f"  Flat-stake ROI    : {roi:+.1%}")
    print(f"  Kelly bankroll    : {start_bankroll:.0f} -> {bankroll:.2f}  "
          f"({bankroll/start_bankroll-1:+.1%})")
    if brier_n:
        print(f"  Brier score       : {brier_sum/brier_n:.4f}  "
              f"(lower is better; 0.25 = coin-flip)")
    if clv_n:
        print(f"  Closing-line value: {clv_sum/clv_n:+.2%} avg  "
              f"(positive = we beat the close)")

    # Reliability table.
    print("\n  Calibration (predicted vs actual win-rate):")
    print(f"  {'bucket':<12}{'n':>5}{'predicted':>12}{'actual':>10}")
    for b in range(10):
        outs = buckets.get(b, [])
        if not outs:
            continue
        lo, hi = b * 10, b * 10 + 10
        actual = sum(outs) / len(outs)
        mid = (lo + hi) / 2 / 100
        print(f"  {lo:>2}-{hi:<3}%{'':<4}{len(outs):>5}"
              f"{mid:>11.0%}{actual:>10.0%}")
    print("\n  (See the look-ahead caveat in the file header.)")


if __name__ == "__main__":
    teams_path = sys.argv[1] if len(sys.argv) > 1 else "teams.csv"
    results_path = sys.argv[2] if len(sys.argv) > 2 else "results.csv"
    run(teams_path, results_path)
