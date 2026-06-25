"""
Point-in-time rating engine (the professional core).

The whole point: to bet without cheating, every match must be evaluated using
only information available *before* kickoff. This module ingests match results
in chronological order and maintains, for each team:

  * an Elo rating, updated after every game (World-Football-Elo style: a goal-
    difference multiplier and a match-importance weight), and
  * a rolling attack / defense strength from recent goals, and
  * a recent-results list for the form model,

all computed strictly from games played *earlier* than the match in question.
`as_of_team()` then hands back a model.Team frozen at that moment, which you
feed straight into the existing expected-goals / betting pipeline.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from model import Team, MatchResult


@dataclass
class EloConfig:
    base: float = 1500.0        # rating every team starts at
    k: float = 40.0             # base update speed (World Cup / big tournaments)
    home_adv: float = 65.0      # Elo points added to a non-neutral home side
    rolling_window: int = 10    # matches used for attack/defense & form
    league_avg_goals: float = 1.35


@dataclass
class _TeamState:
    elo: float
    results: Deque[MatchResult] = field(default_factory=lambda: deque(maxlen=20))
    goals_for: Deque[int] = field(default_factory=lambda: deque(maxlen=20))
    goals_against: Deque[int] = field(default_factory=lambda: deque(maxlen=20))
    games: int = 0


class EloEngine:
    """Maintains evolving ratings as matches are fed in chronological order."""

    def __init__(self, cfg: EloConfig = EloConfig(),
                 styles: Optional[Dict[str, str]] = None):
        self.cfg = cfg
        self.styles = styles or {}
        self.state: Dict[str, _TeamState] = {}

    # --- helpers ---------------------------------------------------------- #
    def _team(self, name: str) -> _TeamState:
        if name not in self.state:
            self.state[name] = _TeamState(elo=self.cfg.base)
        return self.state[name]

    @staticmethod
    def expected(rating_a: float, rating_b: float) -> float:
        """Expected score (win prob + half draw prob) for A vs B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def goal_multiplier(margin: int) -> float:
        """World-Football-Elo goal-difference weighting."""
        margin = abs(margin)
        if margin <= 1:
            return 1.0
        if margin == 2:
            return 1.5
        return (11.0 + margin) / 8.0

    def rating(self, name: str) -> float:
        return self._team(name).elo

    # --- snapshot: a Team as it looked BEFORE a given match --------------- #
    def as_of_team(self, name: str) -> Team:
        st = self._team(name)
        w = self.cfg.rolling_window
        lg = self.cfg.league_avg_goals

        recent_gf = list(st.goals_for)[-w:]
        recent_ga = list(st.goals_against)[-w:]
        if recent_gf:
            attack = (sum(recent_gf) / len(recent_gf)) / lg
            defense = (sum(recent_ga) / len(recent_ga)) / lg
        else:
            attack = defense = 1.0
        # Shrink toward 1.0 when the sample is thin (regularization).
        n = len(recent_gf)
        shrink = n / (n + 5.0)
        attack = 1.0 + shrink * (attack - 1.0)
        defense = 1.0 + shrink * (defense - 1.0)

        return Team(
            name=name,
            elo=st.elo,
            attack=max(0.5, min(1.8, attack)),
            defense=max(0.5, min(1.8, defense)),
            style=self.styles.get(name, "balanced"),
            recent=list(st.results)[:w],   # deque is most-recent-first
            stats=None,
        )

    def games_played(self, name: str) -> int:
        return self._team(name).games

    # --- update after a result ------------------------------------------- #
    def update(self, home: str, away: str, hg: int, ag: int,
               neutral: bool = True, importance: float = 1.0) -> None:
        h, a = self._team(home), self._team(away)

        adj = 0.0 if neutral else self.cfg.home_adv
        we_home = self.expected(h.elo + adj, a.elo)

        if hg > ag:
            w_home = 1.0
        elif hg < ag:
            w_home = 0.0
        else:
            w_home = 0.5

        k = self.cfg.k * importance * self.goal_multiplier(hg - ag)
        delta = k * (w_home - we_home)
        h.elo += delta
        a.elo -= delta

        # Record rolling goals + results (most-recent appended on the right).
        for st, gf, ga in ((h, hg, ag), (a, ag, hg)):
            st.goals_for.append(gf)
            st.goals_against.append(ga)
            st.results.appendleft(
                MatchResult(gf, ga, opponent_elo=(a.elo if st is h else h.elo),
                            importance=importance)
            )
            st.games += 1

    def ratings_table(self) -> List[tuple]:
        return sorted(
            ((nm, st.elo, st.games) for nm, st in self.state.items()),
            key=lambda r: r[1], reverse=True,
        )
