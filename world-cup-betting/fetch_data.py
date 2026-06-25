"""
Auto-fill teams.csv from a live football data API.

Provider: API-Football (https://www.api-sports.io/ ; also on RapidAPI). Its free
tier exposes national-team fixtures, results, and per-match statistics (shots,
shots on target, possession, ...), which is exactly what this model needs.

Setup
-----
1. Get a free key at dashboard.api-football.com (or via RapidAPI).
2. Export it:           export APIFOOTBALL_KEY=your_key_here
   (RapidAPI users also: export APIFOOTBALL_HOST=api-football-v1.p.rapidapi.com)
3. Run, e.g. the 2022 World Cup (league 1, season 2022):

       python3 fetch_data.py --league 1 --season 2022 \
           --teams "Brazil,Argentina,France,Spain,Morocco" --out teams.csv

What it does
------------
For each named team it pulls the most recent fixtures in that league/season,
derives recent form, and aggregates per-match attacking/defensive stats from the
fixture statistics endpoint. Anything the API doesn't provide falls back to the
model's league-average defaults, so a partial pull still yields a usable row.

Notes
-----
* Elo isn't provided by this API, so a light rating proxy is derived from goal
  difference and results (you can hand-edit the `elo` column afterwards, or drop
  in FIFA/World-Football-Elo numbers).
* Style is left as "balanced" — tactical style is a judgement call; set it by
  hand in the CSV. (possession / high_press / counter / low_block / direct).
* Network access goes through the environment's proxy automatically (urllib
  honours HTTPS_PROXY). The script never disables TLS verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Dict, List, Optional, Tuple


DEFAULT_HOST = "v3.football.api-sports.io"


# --------------------------------------------------------------------------- #
# Thin HTTP layer.
# --------------------------------------------------------------------------- #

class APIError(RuntimeError):
    pass


def _api_get(path: str, params: Dict[str, str], key: str, host: str,
             retries: int = 3) -> dict:
    """GET https://{host}/{path}?{params} with the API-Football auth headers."""
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"https://{host}/{path}?{qs}"
    # RapidAPI and direct API-Sports use different header names; send both.
    headers = {
        "x-apisports-key": key,
        "x-rapidapi-key": key,
        "x-rapidapi-host": host,
    }
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("errors"):
                raise APIError(f"API errors: {data['errors']}")
            return data
        except (urllib.error.URLError, APIError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
    raise APIError(f"request failed after {retries} tries: {last_err}")


# --------------------------------------------------------------------------- #
# Endpoint wrappers.
# --------------------------------------------------------------------------- #

def find_team_id(name: str, key: str, host: str) -> Optional[Tuple[int, str]]:
    data = _api_get("teams", {"search": name}, key, host)
    items = data.get("response", [])
    if not items:
        return None
    team = items[0]["team"]
    return team["id"], team["name"]


def recent_fixtures(team_id: int, league: int, season: int,
                    last: int, key: str, host: str) -> List[dict]:
    data = _api_get("fixtures", {
        "team": team_id, "league": league, "season": season,
        "last": last,
    }, key, host)
    return data.get("response", [])


def fixture_stats(fixture_id: int, team_id: int,
                  key: str, host: str) -> Dict[str, float]:
    """Return this team's stat line for a fixture as {stat_type: value}."""
    data = _api_get("fixtures/statistics",
                    {"fixture": fixture_id, "team": team_id}, key, host)
    out: Dict[str, float] = {}
    for block in data.get("response", []):
        for s in block.get("statistics", []):
            val = s.get("value")
            if isinstance(val, str) and val.endswith("%"):
                val = val.rstrip("%")
            try:
                out[s["type"]] = float(val)
            except (TypeError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------- #
# Aggregation -> a teams.csv row.
# --------------------------------------------------------------------------- #

def _avg(values: List[float], default: float) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else default


def build_team_row(name: str, league: int, season: int, last: int,
                   key: str, host: str, pause: float = 0.3) -> Optional[dict]:
    found = find_team_id(name, key, host)
    if not found:
        print(f"  ! '{name}' not found on the API — skipping", file=sys.stderr)
        return None
    team_id, api_name = found

    fixtures = recent_fixtures(team_id, league, season, last, key, host)
    if not fixtures:
        print(f"  ! no fixtures for {api_name} — writing defaults",
              file=sys.stderr)

    form_parts: List[str] = []
    shots_f, sot_f, poss, shots_a, sot_a = [], [], [], [], []
    gd_total = 0
    counted = 0

    for fx in fixtures:
        teams = fx["teams"]
        goals = fx["goals"]
        is_home = teams["home"]["id"] == team_id
        gf = goals["home"] if is_home else goals["away"]
        ga = goals["away"] if is_home else goals["home"]
        if gf is None or ga is None:
            continue
        form_parts.append(f"{gf}-{ga}")
        gd_total += gf - ga
        counted += 1

        # Per-match stats (own + opponent line) when available.
        try:
            mine = fixture_stats(fx["fixture"]["id"], team_id, key, host)
            time.sleep(pause)  # be gentle on rate limits
        except APIError:
            mine = {}
        if mine:
            shots_f.append(mine.get("Total Shots"))
            sot_f.append(mine.get("Shots on Goal"))
            poss.append(mine.get("Ball Possession"))
            # Shots faced approximated by opponent's total shots:
            opp_id = (teams["away"]["id"] if is_home else teams["home"]["id"])
            try:
                opp = fixture_stats(fx["fixture"]["id"], opp_id, key, host)
                time.sleep(pause)
            except APIError:
                opp = {}
            shots_a.append(opp.get("Total Shots"))
            sot_a.append(opp.get("Shots on Goal"))

    avg_gd = gd_total / counted if counted else 0.0
    # Light Elo proxy: centre at 1700, +/-55 per goal of average margin, capped.
    elo = max(1500, min(2100, 1700 + 55 * avg_gd))

    return {
        "name": api_name,
        "elo": round(elo),
        "attack": 1.0,
        "defense": 1.0,
        "style": "balanced",
        "shots_for": round(_avg(shots_f, 12.0), 1),
        "sot_for": round(_avg(sot_f, 4.2), 1),
        "big_for": "",                       # not provided by this endpoint
        "offsides_for": "",
        "possession": round(_avg(poss, 50.0), 1),
        "shots_against": round(_avg(shots_a, 12.0), 1),
        "sot_against": round(_avg(sot_a, 4.2), 1),
        "big_against": "",
        "tackles": "",
        "interceptions": "",
        "form": ";".join(form_parts),
    }


CSV_COLUMNS = [
    "name", "elo", "attack", "defense", "style",
    "shots_for", "sot_for", "big_for", "offsides_for", "possession",
    "shots_against", "sot_against", "big_against", "tackles",
    "interceptions", "form",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-fill teams.csv from API-Football")
    ap.add_argument("--teams", required=True,
                    help="comma-separated team names")
    ap.add_argument("--league", type=int, default=1,
                    help="league id (World Cup = 1)")
    ap.add_argument("--season", type=int, default=2022)
    ap.add_argument("--last", type=int, default=6,
                    help="number of recent fixtures to pull per team")
    ap.add_argument("--out", default="teams.csv")
    args = ap.parse_args()

    key = os.environ.get("APIFOOTBALL_KEY")
    host = os.environ.get("APIFOOTBALL_HOST", DEFAULT_HOST)
    if not key:
        sys.exit(
            "No API key. Set APIFOOTBALL_KEY (free key at "
            "dashboard.api-football.com), then re-run. "
            "See the header of fetch_data.py for details."
        )

    names = [t.strip() for t in args.teams.split(",") if t.strip()]
    rows: List[dict] = []
    for name in names:
        print(f"Fetching {name} ...", file=sys.stderr)
        row = build_team_row(name, args.league, args.season, args.last,
                             key, host)
        if row:
            rows.append(row)

    if not rows:
        sys.exit("Nothing fetched — check the team names / league / season.")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} teams to {args.out}.")
    print("Now set each team's `style` (and ideally `elo`) by hand, then run "
          "batch.py / simulate.py.")


if __name__ == "__main__":
    main()
