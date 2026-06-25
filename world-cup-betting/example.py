"""
Worked example: analyze a World Cup fixture end to end.

Run with:  python3 example.py

The numbers below are illustrative (loosely 2022-era profiles) to show how to
populate teams, recent form, underlying stats, and bookmaker odds. Replace them
with current data before betting anything real.
"""

from model import (
    Team,
    TeamStats,
    MatchResult,
    analyze_match,
)


def make_teams():
    brazil = Team(
        name="Brazil",
        elo=2050,
        attack=1.35,
        defense=0.80,
        style="possession",
        recent=[
            MatchResult(2, 0, opponent_elo=1850),
            MatchResult(4, 1, opponent_elo=1700),
            MatchResult(1, 0, opponent_elo=1950),
            MatchResult(3, 0, opponent_elo=1650),
            MatchResult(1, 1, opponent_elo=2000),
        ],
        stats=TeamStats(
            shots_for=16.5, shots_on_target_for=6.4, big_chances_for=2.9,
            offsides_for=2.3, possession=58.0,
            shots_against=8.0, shots_on_target_against=2.6,
            big_chances_against=1.0, tackles=17.0, interceptions=9.5,
        ),
    )

    morocco = Team(
        name="Morocco",
        elo=1820,
        attack=0.92,
        defense=0.70,            # very solid defensively
        style="counter",
        recent=[
            MatchResult(2, 0, opponent_elo=1950),
            MatchResult(1, 0, opponent_elo=2000),
            MatchResult(0, 0, opponent_elo=1900),
            MatchResult(2, 1, opponent_elo=1750),
            MatchResult(1, 0, opponent_elo=1800),
        ],
        stats=TeamStats(
            shots_for=9.5, shots_on_target_for=3.4, big_chances_for=1.3,
            offsides_for=1.6, possession=44.0,
            shots_against=9.5, shots_on_target_against=2.8,
            big_chances_against=1.1, tackles=21.0, interceptions=11.0,
        ),
    )
    return brazil, morocco


def main():
    home, away = make_teams()

    # Bookmaker decimal odds for this fixture (what you'd see at a sportsbook).
    book_odds = {
        "Home win": 1.70,
        "Draw": 3.80,
        "Away win": 6.00,
        "Over 2.5": 2.10,
        "Under 2.5": 1.75,
        "BTTS Yes": 2.05,
        "BTTS No": 1.78,
    }

    probs, bets = analyze_match(home, away, book_odds=book_odds, neutral=True)

    print(f"\n{'='*56}")
    print(f"  {home.name}  vs  {away.name}   (neutral venue)")
    print(f"{'='*56}")
    print(f"  Expected goals:  {home.name} {probs.lambda_home:.2f}"
          f"  -  {probs.lambda_away:.2f} {away.name}")
    print()
    print(f"  {home.name} win : {probs.home_win:6.1%}   "
          f"(fair {1/probs.home_win:.2f})")
    print(f"  Draw       : {probs.draw:6.1%}   (fair {1/probs.draw:.2f})")
    print(f"  {away.name} win: {probs.away_win:6.1%}   "
          f"(fair {1/probs.away_win:.2f})")
    print()
    print(f"  Over 2.5   : {probs.over_2_5:6.1%}      "
          f"Under 2.5 : {probs.under_2_5:6.1%}")
    print(f"  BTTS Yes   : {probs.btts_yes:6.1%}      "
          f"BTTS No   : {probs.btts_no:6.1%}")
    print()
    print("  Most likely scorelines:")
    for score, p in probs.top_scores:
        print(f"     {score}   {p:5.1%}")

    print(f"\n{'-'*56}")
    print("  VALUE BETS (model edge over the bookmaker)")
    print(f"{'-'*56}")
    if not bets:
        print("  No +EV bets at these odds. Sit this one out.")
    else:
        for b in bets:
            print(f"  {b.market:<10}  book {b.book_odds:>5.2f}  "
                  f"fair {b.fair_odds:>5.2f}  "
                  f"edge {b.edge:+6.1%}  stake {b.kelly_stake:5.2%} bankroll")
    print()


if __name__ == "__main__":
    main()
