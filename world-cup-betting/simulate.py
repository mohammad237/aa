"""
Monte-Carlo World Cup tournament simulator.

Plays the whole tournament thousands of times using the same expected-goals
engine, sampling each match's scoreline from the model, and reports each team's
probability of advancing from the group, reaching each round, and winning it.

Usage:
    python3 simulate.py            # default 8-group bracket from groups.csv
    python3 simulate.py 20000      # custom number of simulations

It reads team data from teams.csv and the group draw from groups.csv.
Outright "to win the tournament" odds let you spot value on the futures market.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from data_io import load_teams
from model import Team, expected_goals, CONFIG


# --------------------------------------------------------------------------- #
# Sampling a single match from the model.
# --------------------------------------------------------------------------- #

def _sample_poisson(lam: float, rng: random.Random) -> int:
    """Knuth's algorithm for a Poisson sample."""
    L = pow(2.718281828459045, -lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def sample_score(home: Team, away: Team, rng: random.Random,
                 neutral: bool = True) -> Tuple[int, int]:
    lam_h, lam_a = expected_goals(home, away, neutral=neutral)
    return _sample_poisson(lam_h, rng), _sample_poisson(lam_a, rng)


def sample_knockout(home: Team, away: Team, rng: random.Random) -> Team:
    """A knockout match: draws are broken by ET (extra goal weight) then pens."""
    h, a = sample_score(home, away, rng, neutral=True)
    if h > a:
        return home
    if a > h:
        return away
    # Extra time: ~1/3 of a match of additional expected goals.
    lam_h, lam_a = expected_goals(home, away, neutral=True)
    eh = _sample_poisson(lam_h / 3.0, rng)
    ea = _sample_poisson(lam_a / 3.0, rng)
    if eh > ea:
        return home
    if ea > eh:
        return away
    # Penalties: slight edge to the stronger (higher-Elo) side.
    p_home = 1.0 / (1.0 + 10 ** ((away.elo - home.elo) / 600.0))
    return home if rng.random() < p_home else away


# --------------------------------------------------------------------------- #
# Group + bracket structure.
# --------------------------------------------------------------------------- #

def load_groups(path: str) -> Dict[str, List[str]]:
    """groups.csv: columns group,team (one team per row)."""
    groups: Dict[str, List[str]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            g = (row.get("group") or "").strip()
            t = (row.get("team") or "").strip()
            if g and t:
                groups[g].append(t)
    return dict(groups)


def play_group(group_teams: List[Team], rng: random.Random
               ) -> List[Team]:
    """Round-robin; return teams ranked (winner, runner-up, ...)."""
    pts: Dict[str, int] = defaultdict(int)
    gd: Dict[str, int] = defaultdict(int)
    gf: Dict[str, int] = defaultdict(int)
    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            a, b = group_teams[i], group_teams[j]
            ga, gb = sample_score(a, b, rng, neutral=True)
            gf[a.name] += ga
            gf[b.name] += gb
            gd[a.name] += ga - gb
            gd[b.name] += gb - ga
            if ga > gb:
                pts[a.name] += 3
            elif gb > ga:
                pts[b.name] += 3
            else:
                pts[a.name] += 1
                pts[b.name] += 1
    ranked = sorted(
        group_teams,
        key=lambda t: (pts[t.name], gd[t.name], gf[t.name],
                       rng.random()),  # random tiebreak as a last resort
        reverse=True,
    )
    return ranked


def build_bracket(qualifiers: Dict[str, List[Team]],
                  group_order: List[str]) -> List[Team]:
    """
    Standard cross-over seeding for any even number of groups: the winner of
    each group meets the runner-up of its neighbour. Works for 4 groups
    (-> quarter-finals) or 8 groups (-> round of 16).
    """
    order = group_order
    n = len(order)
    pairs: List[Tuple[int, int]] = []
    # Front half: 1A-2B, 1C-2D, ...
    for k in range(0, n, 2):
        pairs.append((k, k + 1))
    # Back half (keeps group neighbours apart): 1B-2A, 1D-2C, ...
    for k in range(0, n, 2):
        pairs.append((k + 1, k))

    bracket: List[Team] = []
    for wi, ri in pairs:
        bracket.append(qualifiers[order[wi]][0])   # winner of group wi
        bracket.append(qualifiers[order[ri]][1])   # runner-up of group ri
    return bracket


def play_knockout_round(teams: List[Team], rng: random.Random) -> List[Team]:
    winners = []
    for i in range(0, len(teams), 2):
        winners.append(sample_knockout(teams[i], teams[i + 1], rng))
    return winners


# --------------------------------------------------------------------------- #
# The simulation loop.
# --------------------------------------------------------------------------- #

def simulate(teams_path="teams.csv", groups_path="groups.csv",
             n_sims=10000, seed=None) -> None:
    teams = load_teams(teams_path)
    groups = load_groups(groups_path)
    group_order = sorted(groups.keys())
    rng = random.Random(seed)

    # Validate every team exists.
    for g, names in groups.items():
        for nm in names:
            if nm not in teams:
                raise SystemExit(f"Team '{nm}' (group {g}) not in {teams_path}")

    advance = defaultdict(int)                 # reached the knockout stage
    reached = defaultdict(lambda: defaultdict(int))  # reached[size][name]
    champions = defaultdict(int)

    for _ in range(n_sims):
        qualifiers: Dict[str, List[Team]] = {}
        for g, names in groups.items():
            ranked = play_group([teams[n] for n in names], rng)
            qualifiers[g] = ranked[:2]
            for t in ranked[:2]:
                advance[t.name] += 1

        if len(group_order) % 2 == 0 and all(len(qualifiers[g]) >= 2
                                             for g in group_order):
            bracket = build_bracket(qualifiers, group_order)
        else:
            bracket = [t for g in group_order for t in qualifiers[g]]
        # Bracket must be a power of two to play a clean single elimination.
        if len(bracket) < 2 or (len(bracket) & (len(bracket) - 1)) != 0:
            continue

        while len(bracket) > 1:
            for t in bracket:
                reached[len(bracket)][t.name] += 1
            bracket = play_knockout_round(bracket, rng)
        champions[bracket[0].name] += 1

    # Determine which round sizes actually occurred (e.g. 16, 8, 4, 2).
    sizes = sorted(reached.keys(), reverse=True)
    labels = {16: "R16", 8: "QF", 4: "SF", 2: "Final"}

    header = f"\n{'Team':<14}{'Adv':>7}"
    for s in sizes:
        header += f"{labels.get(s, str(s)):>7}"
    header += f"{'Win':>7}{'Fair':>9}"
    print(header)
    print("-" * len(header.strip("\n")))

    all_names = sorted(teams.keys(), key=lambda n: champions[n], reverse=True)
    for nm in all_names:
        if advance[nm] == 0 and champions[nm] == 0:
            continue
        win_p = champions[nm] / n_sims
        fair = f"{1/win_p:.1f}" if win_p > 0 else "—"
        row = f"{nm:<14}{advance[nm]/n_sims:>7.1%}"
        for s in sizes:
            row += f"{reached[s][nm]/n_sims:>7.1%}"
        row += f"{win_p:>7.1%}{fair:>9}"
        print(row)
    print(f"\n({n_sims:,} simulations)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    simulate(n_sims=n, seed=42)
