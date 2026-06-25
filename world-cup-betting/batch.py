"""
Batch-analyze a whole matchday from CSV.

Usage:
    python3 batch.py                      # uses teams.csv + fixtures.csv
    python3 batch.py teams.csv games.csv  # custom files

For each fixture it prints model probabilities and any +EV value bets, then a
ranked summary of every value bet found across the matchday (best edge first).
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from data_io import load_teams, load_fixtures, Fixture
from model import analyze_match, ValueBet, Team


def run(teams_path: str, fixtures_path: str) -> None:
    teams = load_teams(teams_path)
    fixtures = load_fixtures(fixtures_path)

    all_value: List[Tuple[str, ValueBet]] = []

    for fx in fixtures:
        if fx.home not in teams or fx.away not in teams:
            missing = fx.home if fx.home not in teams else fx.away
            print(f"  ! skipping {fx.home} vs {fx.away}: '{missing}' "
                  f"not in team database")
            continue

        home: Team = teams[fx.home]
        away: Team = teams[fx.away]
        probs, bets = analyze_match(
            home, away, book_odds=fx.book_odds, neutral=fx.neutral)

        print(f"\n{fx.home} vs {fx.away}"
              f"{'  (neutral)' if fx.neutral else '  (home edge)'}")
        print(f"  xG {probs.lambda_home:.2f}-{probs.lambda_away:.2f}   "
              f"1 {probs.home_win:.0%} | X {probs.draw:.0%} | "
              f"2 {probs.away_win:.0%}   "
              f"O2.5 {probs.over_2_5:.0%}  BTTS {probs.btts_yes:.0%}")
        if bets:
            for b in bets:
                print(f"    VALUE  {b.market:<10} @ {b.book_odds:>5.2f}  "
                      f"(fair {b.fair_odds:>5.2f})  edge {b.edge:+5.1%}  "
                      f"stake {b.kelly_stake:.2%}")
                all_value.append((f"{fx.home} v {fx.away}", b))
        else:
            print("    no value")

    print(f"\n{'='*64}")
    print("  MATCHDAY VALUE BOARD (ranked by edge)")
    print(f"{'='*64}")
    if not all_value:
        print("  No +EV bets across the slate.")
        return
    all_value.sort(key=lambda x: x[1].edge, reverse=True)
    for fixture_name, b in all_value:
        print(f"  {b.edge:+5.1%}  {fixture_name:<22} {b.market:<10} "
              f"@ {b.book_odds:.2f}  stake {b.kelly_stake:.2%}")


if __name__ == "__main__":
    teams_path = sys.argv[1] if len(sys.argv) > 1 else "teams.csv"
    fixtures_path = sys.argv[2] if len(sys.argv) > 2 else "fixtures.csv"
    run(teams_path, fixtures_path)
