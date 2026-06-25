"""
CSV input/output helpers: load a team database and a matchday of fixtures.

Two files drive a batch run:

  teams.csv     - one row per team (ratings, style, underlying stats, form)
  fixtures.csv  - one row per fixture (the two teams + bookmaker odds)

See the sample CSVs in this folder for the exact columns.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional

from model import Team, TeamStats, MatchResult


def _f(row: Dict[str, str], key: str, default: float) -> float:
    """Read a float column, falling back to a default when blank/missing."""
    val = row.get(key, "")
    if val is None or str(val).strip() == "":
        return default
    return float(val)


def _parse_form(spec: str) -> List[MatchResult]:
    """
    Parse a form string like "2-0@1850;1-0@1900;0-0".

    Each entry is "goals_for-goals_against" with an optional "@opponentElo".
    Most-recent first. Empty string -> no form data.
    """
    results: List[MatchResult] = []
    spec = (spec or "").strip()
    if not spec:
        return results
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        opp_elo = 1500.0
        if "@" in chunk:
            chunk, opp = chunk.split("@", 1)
            try:
                opp_elo = float(opp)
            except ValueError:
                opp_elo = 1500.0
        gf, ga = chunk.split("-")
        results.append(MatchResult(int(gf), int(ga), opponent_elo=opp_elo))
    return results


def load_teams(path: str) -> Dict[str, Team]:
    """Load teams.csv into a {name: Team} dict."""
    teams: Dict[str, Team] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["name"].strip()
            if not name:
                continue
            stats = TeamStats(
                shots_for=_f(row, "shots_for", 12.0),
                shots_on_target_for=_f(row, "sot_for", 4.2),
                big_chances_for=_f(row, "big_for", 1.8),
                offsides_for=_f(row, "offsides_for", 2.0),
                possession=_f(row, "possession", 50.0),
                shots_against=_f(row, "shots_against", 12.0),
                shots_on_target_against=_f(row, "sot_against", 4.2),
                big_chances_against=_f(row, "big_against", 1.8),
                tackles=_f(row, "tackles", 18.0),
                interceptions=_f(row, "interceptions", 9.0),
            )
            teams[name] = Team(
                name=name,
                elo=_f(row, "elo", 1700.0),
                attack=_f(row, "attack", 1.0),
                defense=_f(row, "defense", 1.0),
                style=(row.get("style") or "balanced").strip() or "balanced",
                recent=_parse_form(row.get("form", "")),
                stats=stats,
            )
    return teams


# Map fixtures.csv odds columns -> model market names.
ODDS_COLUMNS = {
    "home_win": "Home win",
    "draw": "Draw",
    "away_win": "Away win",
    "over25": "Over 2.5",
    "under25": "Under 2.5",
    "btts_yes": "BTTS Yes",
    "btts_no": "BTTS No",
    "dc_1x": "DC 1X",
    "dc_12": "DC 12",
    "dc_x2": "DC X2",
    "dnb_home": "DNB Home",
    "dnb_away": "DNB Away",
}


class Fixture:
    def __init__(self, home: str, away: str, neutral: bool,
                 book_odds: Dict[str, float]):
        self.home = home
        self.away = away
        self.neutral = neutral
        self.book_odds = book_odds


def load_fixtures(path: str) -> List[Fixture]:
    """Load fixtures.csv into a list of Fixture objects."""
    fixtures: List[Fixture] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            home = (row.get("home") or "").strip()
            away = (row.get("away") or "").strip()
            if not home or not away:
                continue
            neutral = str(row.get("neutral", "1")).strip().lower() in (
                "1", "true", "yes", "y", "")
            book_odds: Dict[str, float] = {}
            for col, market in ODDS_COLUMNS.items():
                val = row.get(col, "")
                if val is not None and str(val).strip() != "":
                    try:
                        book_odds[market] = float(val)
                    except ValueError:
                        pass
            # Free-form extra markets: any column named "ah:AH Home -1.0=1.95"
            # style is overkill, so we also accept generic "odds_<Market>" cols.
            fixtures.append(Fixture(home, away, neutral, book_odds))
    return fixtures
