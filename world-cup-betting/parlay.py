"""
Accumulator / parlay expected-value builder.

A parlay combines several legs into one bet: all must win to pay out. The odds
multiply, and so do the probabilities — which is exactly why parlays are
seductive and usually -EV. This tool only ever builds parlays out of legs the
model already rates as +EV singles, and it tells you the combined edge so you
can see whether stacking them still clears the bar.

Two important realities baked in:

  * Independence. Legs from DIFFERENT matches are treated as independent, so the
    combined probability is the product. Legs from the SAME match are usually
    correlated (e.g. "Home win" and "Over 2.5"), so by default we refuse to put
    two legs from one fixture in the same parlay. You can override that, but the
    EV will be wrong if you do.
  * Edge compounds AND erodes. A parlay of +EV legs keeps positive edge only if
    each leg is genuinely +EV; the bookmaker's margin is multiplied too, so
    longer parlays need bigger per-leg edges to survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from model import ValueBet, fair_odds


@dataclass
class ParlayLeg:
    fixture: str           # which match this leg comes from
    market: str
    model_prob: float
    book_odds: float


@dataclass
class Parlay:
    legs: List[ParlayLeg]
    combined_prob: float
    combined_odds: float
    edge: float            # EV per unit staked
    kelly_stake: float

    @property
    def fair_odds(self) -> float:
        return fair_odds(self.combined_prob)


def build_parlay(
    legs: List[ParlayLeg],
    kelly_fraction: float = 0.25,
    max_stake: float = 0.03,
    allow_same_fixture: bool = False,
) -> Optional[Parlay]:
    """
    Combine legs into one parlay and compute its combined probability, odds,
    edge, and a (small) fractional-Kelly stake. Returns None if the legs are
    invalid (e.g. two legs from the same fixture when that isn't allowed).
    """
    if not legs:
        return None
    if not allow_same_fixture:
        fixtures = [leg.fixture for leg in legs]
        if len(set(fixtures)) != len(fixtures):
            return None

    combined_prob = 1.0
    combined_odds = 1.0
    for leg in legs:
        combined_prob *= leg.model_prob
        combined_odds *= leg.book_odds

    edge = combined_prob * combined_odds - 1.0
    # Parlay stakes are tiny: variance is the product of the legs' variance.
    full_kelly = edge / (combined_odds - 1.0) if combined_odds > 1.0 else 0.0
    stake = max(0.0, min(max_stake, full_kelly * kelly_fraction))
    return Parlay(
        legs=legs,
        combined_prob=combined_prob,
        combined_odds=combined_odds,
        edge=edge,
        kelly_stake=stake,
    )


def value_bets_to_legs(
    value_by_fixture: List[Tuple[str, ValueBet]]
) -> List[ParlayLeg]:
    """Convert (fixture_name, ValueBet) pairs from a batch run into legs."""
    return [
        ParlayLeg(fixture=name, market=b.market,
                  model_prob=b.model_prob, book_odds=b.book_odds)
        for name, b in value_by_fixture
    ]


def best_parlays(
    legs: List[ParlayLeg],
    size: int = 3,
    top_n: int = 10,
    min_edge: float = 0.05,
    min_leg_edge: float = 0.0,
    **kwargs,
) -> List[Parlay]:
    """
    Enumerate all parlays of `size` legs (one leg per fixture) and return the
    `top_n` by combined edge that clear `min_edge`. `min_leg_edge` optionally
    filters the input legs to only sufficiently strong singles first.
    """
    pool = [
        leg for leg in legs
        if (leg.model_prob * leg.book_odds - 1.0) >= min_leg_edge
    ]
    out: List[Parlay] = []
    for combo in combinations(pool, size):
        # One leg per fixture (avoid correlated same-match legs).
        if len({leg.fixture for leg in combo}) != size:
            continue
        parlay = build_parlay(list(combo), **kwargs)
        if parlay and parlay.edge >= min_edge:
            out.append(parlay)
    out.sort(key=lambda p: p.edge, reverse=True)
    return out[:top_n]
