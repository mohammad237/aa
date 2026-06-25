"""
Generate a SYNTHETIC chronological match history for demos/tests.

This fabricates a believable run of national-team fixtures with results and
bookmaker odds, so ratings.py / walkforward.py / calibrate.py have something to
chew on without a paid data feed. It is NOT real data — replace history.csv with
genuine results + odds before drawing any conclusions.

How it works: each team has a hidden "true" strength; match scores are sampled
from the model's Poisson with those true strengths; odds are the true fair odds
plus a bookmaker margin and a bit of noise (so there is real, beatable value to
find). Seeded, so it is reproducible.

    python3 make_sample_history.py > history.csv
"""

from __future__ import annotations

import math
import random
import sys
from typing import Dict, List, Tuple

from model import scoreline_matrix

SEED = 7
random.seed(SEED)

# Hidden "true" parameters per team: (elo, attack, defense, style).
TRUE: Dict[str, Tuple[float, float, float, str]] = {
    "Brazil":      (2050, 1.35, 0.80, "possession"),
    "Argentina":   (2080, 1.30, 0.78, "possession"),
    "France":      (2040, 1.32, 0.85, "direct"),
    "Spain":       (1990, 1.25, 0.88, "possession"),
    "England":     (1970, 1.22, 0.86, "balanced"),
    "Germany":     (1950, 1.24, 0.95, "high_press"),
    "Netherlands": (1930, 1.18, 0.84, "possession"),
    "Portugal":    (1960, 1.26, 0.90, "possession"),
    "Croatia":     (1880, 1.05, 0.82, "possession"),
    "Morocco":     (1820, 0.92, 0.70, "counter"),
    "Japan":       (1810, 0.98, 0.88, "high_press"),
    "USA":         (1800, 1.00, 0.92, "high_press"),
    "Senegal":     (1790, 0.96, 0.86, "direct"),
    "Mexico":      (1780, 0.94, 0.90, "balanced"),
    "Australia":   (1730, 0.85, 0.95, "low_block"),
    "Ghana":       (1740, 0.95, 1.05, "direct"),
}

LEAGUE_AVG = 1.35
ELO_BETA = 0.0016
MARGIN = 0.06        # bookmaker overround (~6%)
NOISE = 0.05         # multiplicative noise on each price

# Realistic soft-book weaknesses the model can exploit: the bookmaker in this
# sample prices every game as if it were at a neutral venue (it ignores home
# advantage). Reality includes a genuine home edge, so backing home sides in
# non-neutral games carries real value. (Set HOME_ADV_ELO=0 for a perfectly
# efficient book, in which case nothing is beatable — see calibrate.py.)
HOME_ADV_ELO = 80.0


def true_lambdas(home: str, away: str, neutral: bool = True) -> Tuple[float, float]:
    eh, ah, dh, _ = TRUE[home]
    ea, aa, da, _ = TRUE[away]
    diff = (eh + (0.0 if neutral else HOME_ADV_ELO)) - ea
    lam_h = LEAGUE_AVG * ah * da * math.exp(ELO_BETA * diff)
    lam_a = LEAGUE_AVG * aa * dh * math.exp(-ELO_BETA * diff)
    return max(0.2, lam_h), max(0.2, lam_a)


def sample_poisson(lam: float) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def fair_market_probs(lam_h: float, lam_a: float) -> Dict[str, float]:
    grid = scoreline_matrix(lam_h, lam_a)
    p_h = p_d = p_a = p_over = p_btts = 0.0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            p = grid[i][j]
            if i > j:
                p_h += p
            elif i == j:
                p_d += p
            else:
                p_a += p
            if i + j >= 3:
                p_over += p
            if i >= 1 and j >= 1:
                p_btts += p
    return {
        "home_win": p_h, "draw": p_d, "away_win": p_a,
        "over25": p_over, "under25": 1 - p_over,
        "btts_yes": p_btts, "btts_no": 1 - p_btts,
    }


def priced(prob: float) -> str:
    """Fair odds with margin + noise applied -> a sportsbook-like price."""
    if prob <= 1e-6:
        return ""
    p = prob * (1 + MARGIN) * (1 + random.uniform(-NOISE, NOISE))
    p = min(0.99, p)
    return f"{1.0 / p:.2f}"


def main() -> None:
    teams = list(TRUE.keys())
    cols = ["date", "home", "away", "neutral", "importance",
            "home_goals", "away_goals",
            "home_win", "draw", "away_win",
            "over25", "under25", "btts_yes", "btts_no"]
    out: List[str] = [",".join(cols)]

    # A schedule: several "windows", each a random round-robin-ish slate.
    year, month, day = 2021, 1, 5
    windows = 22
    for _ in range(windows):
        random.shuffle(teams)
        # pair teams up for this window
        for i in range(0, len(teams) - 1, 2):
            home, away = teams[i], teams[i + 1]
            neutral = random.random() < 0.5
            importance = random.choice([1.0, 1.0, 1.2, 1.6])
            # Results follow the TRUE process (with home advantage)...
            lam_h, lam_a = true_lambdas(home, away, neutral=neutral)
            hg, ag = sample_poisson(lam_h), sample_poisson(lam_a)
            # ...but the bookmaker prices the game as if it were neutral,
            # leaving exploitable value for a model that accounts for venue.
            blam_h, blam_a = true_lambdas(home, away, neutral=True)
            mp = fair_market_probs(blam_h, blam_a)
            row = [
                f"{year:04d}-{month:02d}-{day:02d}", home, away,
                "1" if neutral else "0", f"{importance:.1f}",
                str(hg), str(ag),
                priced(mp["home_win"]), priced(mp["draw"]), priced(mp["away_win"]),
                priced(mp["over25"]), priced(mp["under25"]),
                priced(mp["btts_yes"]), priced(mp["btts_no"]),
            ]
            out.append(",".join(row))
        # advance the calendar ~3 weeks
        day += 21
        while day > 28:
            day -= 28
            month += 1
            if month > 12:
                month = 1
                year += 1

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
