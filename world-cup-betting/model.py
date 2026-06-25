"""
World Cup match betting model.

A transparent, dependency-free model that turns team data (ratings, recent
form, and tactical style) into match outcome probabilities, fair odds, and
value-bet / staking recommendations.

The pipeline, end to end:

    ratings + home edge + form + tactics
        -> expected goals (lambda) for each side
        -> Dixon-Coles corrected Poisson scoreline grid
        -> P(home win / draw / away win), Over/Under, BTTS
        -> fair odds, then compare to bookmaker odds
        -> expected value (edge) and fractional-Kelly stake

Everything is pure-Python / stdlib so it runs anywhere with `python3`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Tunable global parameters (the "knobs" of the model).
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    # Baseline goals an average team scores against an average team (per side).
    league_avg_goals: float = 1.35

    # Home-field advantage. At a World Cup most games are neutral venues, so
    # this is small and is meant for the host nation or "home continent" edge.
    # Expressed as Elo points added to the nominal home side.
    home_field_elo: float = 35.0

    # How strongly the Elo gap bends the goal expectation. Kept modest so it
    # *modulates* the attack/defense model rather than overriding it.
    elo_beta: float = 0.0016

    # How strongly recent form swings a team's attack/defense (per unit of the
    # normalized form score, which lives in roughly [-1, 1]).
    form_strength: float = 0.18

    # How strongly a favorable/unfavorable style matchup swings attack output.
    tactics_strength: float = 0.12

    # How much to trust underlying stats (shots/SoT/big chances/tackles/
    # offsides) vs the manual attack/defense ratings. 0 = ignore stats,
    # 1 = ratings come entirely from stats. Only applies to teams that have
    # a TeamStats object attached.
    stats_blend: float = 0.5

    # Dixon-Coles low-score dependence parameter (rho). Negative inflates draws
    # and 0-0 / 1-1 scorelines, matching real football where independence
    # over-predicts certain low scores.
    dc_rho: float = -0.08

    # Largest goal total considered when building the scoreline grid.
    max_goals: int = 10


CONFIG = ModelConfig()


# --------------------------------------------------------------------------- #
# Tactical style model.
# --------------------------------------------------------------------------- #
#
# Each team is assigned a primary playing style. The matrix below encodes how
# much of an *attacking* edge a row-style gets when it faces a column-style.
# Positive => the row style tends to create more than expected against that
# opponent; negative => it is blunted. Values are small (≈ +/-1) and are scaled
# by CONFIG.tactics_strength before being applied, so tactics nudge rather than
# dominate.
#
# Styles:
#   possession      - patient build-up, dominate the ball (e.g. Spain)
#   high_press      - aggressive pressing, win ball high (e.g. Germany'10s)
#   counter         - sit deep, break fast (e.g. Morocco 2022)
#   low_block       - defensive, compact, absorb pressure
#   direct          - vertical, long balls, second balls, set pieces
#   balanced        - no strong identity / adaptable

STYLES = ["possession", "high_press", "counter", "low_block", "direct", "balanced"]

# Read as: STYLE_MATCHUP[my_style][their_style] = attacking edge for "my_style".
STYLE_MATCHUP: Dict[str, Dict[str, float]] = {
    "possession": {
        "possession": 0.0, "high_press": -0.6, "counter": -0.4,
        "low_block": -0.3, "direct": 0.3, "balanced": 0.2,
    },
    "high_press": {
        "possession": 0.6, "high_press": 0.0, "counter": -0.3,
        "low_block": 0.1, "direct": -0.2, "balanced": 0.2,
    },
    "counter": {
        "possession": 0.5, "high_press": 0.2, "counter": 0.0,
        "low_block": -0.4, "direct": 0.0, "balanced": 0.1,
    },
    "low_block": {
        "possession": 0.3, "high_press": -0.2, "counter": 0.2,
        "low_block": 0.0, "direct": -0.3, "balanced": 0.0,
    },
    "direct": {
        "possession": 0.2, "high_press": 0.3, "counter": 0.0,
        "low_block": 0.4, "direct": 0.0, "balanced": 0.1,
    },
    "balanced": {
        "possession": 0.0, "high_press": -0.1, "counter": -0.1,
        "low_block": 0.0, "direct": 0.0, "balanced": 0.0,
    },
}


def style_edge(my_style: str, opp_style: str) -> float:
    """Attacking edge for `my_style` vs `opp_style` (0 if unknown styles)."""
    return STYLE_MATCHUP.get(my_style, {}).get(opp_style, 0.0)


# --------------------------------------------------------------------------- #
# Team and recent-results data.
# --------------------------------------------------------------------------- #

@dataclass
class MatchResult:
    """A single past result from the team's perspective."""
    goals_for: int
    goals_against: int
    opponent_elo: float = 1500.0   # strength of the opponent faced
    importance: float = 1.0        # 1.0 friendly ... up to ~1.6 knockout


@dataclass
class TeamStats:
    """
    Underlying performance stats, expressed as per-match averages over the
    recent sample. These describe *how* a team generates and concedes chances,
    which is more stable than goals alone (goals are noisy; shot volume and
    shot quality regress less).

    "for" = the team's own output; "against" = what they allow opponents.
    Leave any field at its league-average default if you don't have the data.
    """
    # Attacking output (per match)
    shots_for: float = 12.0
    shots_on_target_for: float = 4.2
    big_chances_for: float = 1.8
    offsides_for: float = 2.0          # proxy for an aggressive, high line / risk
    possession: float = 50.0           # percent

    # Defensive / what they concede (per match)
    shots_against: float = 12.0
    shots_on_target_against: float = 4.2
    big_chances_against: float = 1.8
    tackles: float = 18.0              # engagement / ball-winning volume
    interceptions: float = 9.0


# League-average reference values used to normalize TeamStats into multipliers.
STATS_BASELINE = TeamStats()


@dataclass
class Team:
    name: str
    elo: float                      # overall strength
    attack: float = 1.0             # >1 = scores more than an average side
    defense: float = 1.0            # >1 = concedes more than an average side (leaky)
    style: str = "balanced"
    recent: List[MatchResult] = field(default_factory=list)
    stats: Optional["TeamStats"] = None   # underlying shot/tackle/offside data


# --------------------------------------------------------------------------- #
# Form: turn the last N results into a single normalized score in ~[-1, 1].
# --------------------------------------------------------------------------- #

def form_score(team: Team, half_life: float = 3.0) -> float:
    """
    Weighted recent-form score.

    More recent matches count more (exponential decay with the given
    half-life, measured in games). Each game contributes a blend of:
      * the result (win/draw/loss -> +1 / 0 / -1), and
      * the goal difference (capped), and
      * an opponent-quality bonus (beating strong sides counts more).
    Returns a value normalized to roughly [-1, 1]; 0.0 when there is no data.
    """
    if not team.recent:
        return 0.0

    decay = math.log(2) / half_life
    weighted_sum = 0.0
    weight_total = 0.0

    # team.recent is assumed most-recent-first.
    for i, m in enumerate(team.recent):
        w = math.exp(-decay * i)

        if m.goals_for > m.goals_against:
            result = 1.0
        elif m.goals_for < m.goals_against:
            result = -1.0
        else:
            result = 0.0

        gd = max(-3, min(3, m.goals_for - m.goals_against)) / 3.0

        # Opponent quality: facing a 1700-rated side scales toward +0.5.
        opp_q = max(-0.5, min(0.5, (m.opponent_elo - 1500.0) / 400.0))
        # A good result vs a strong team is worth more.
        quality_bonus = opp_q * result

        game_score = (0.6 * result + 0.3 * gd + 0.1 * quality_bonus) * m.importance
        weighted_sum += w * game_score
        weight_total += w

    return weighted_sum / weight_total if weight_total else 0.0


# --------------------------------------------------------------------------- #
# Underlying-stats layer: shots / shots on target / big chances / tackles /
# offsides -> attacking and defensive performance multipliers.
# --------------------------------------------------------------------------- #
#
# Goals are noisy; shot *volume* and shot *quality* are far more stable and
# predictive. We build a stats-based "expected output" index for attack and a
# "solidity" index for defense, each centered on 1.0 (league average), then
# blend it into the ratings inside expected_goals().

def stats_attack_index(stats: TeamStats, b: TeamStats = STATS_BASELINE) -> float:
    """
    Attacking strength implied by underlying numbers (1.0 = average).

    Weighted toward shot quality: shots on target and big chances predict goals
    much better than raw shot count. Possession and offsides add a small signal
    (territorial control / willingness to attack in behind), but are capped so a
    sterile possession side isn't overrated.
    """
    sot_ratio = stats.shots_on_target_for / b.shots_on_target_for
    bigch_ratio = stats.big_chances_for / b.big_chances_for
    shots_ratio = stats.shots_for / b.shots_for
    poss_ratio = stats.possession / b.possession
    # Offsides: mild positive (attacking intent), but the cap below prevents
    # too many offsides from inflating attack — they are wasted chances.
    offside_signal = 1.0 + 0.04 * (stats.offsides_for - b.offsides_for)

    index = (
        0.42 * sot_ratio
        + 0.30 * bigch_ratio
        + 0.16 * shots_ratio
        + 0.07 * poss_ratio
        + 0.05 * offside_signal
    )
    return max(0.6, min(1.6, index))


def stats_defense_index(stats: TeamStats, b: TeamStats = STATS_BASELINE) -> float:
    """
    Defensive *leakiness* implied by underlying numbers (1.0 = average,
    >1 = concedes more than average). Driven by the quality of chances allowed;
    tackles and interceptions reduce leakiness (active ball-winning), with a cap
    so a foul-happy side that tackles a lot but still concedes isn't flattered.
    """
    sot_against_ratio = stats.shots_on_target_against / b.shots_on_target_against
    bigch_against_ratio = stats.big_chances_against / b.big_chances_against
    shots_against_ratio = stats.shots_against / b.shots_against

    leak = (
        0.45 * sot_against_ratio
        + 0.33 * bigch_against_ratio
        + 0.22 * shots_against_ratio
    )

    # Ball-winning volume relative to baseline trims leakiness slightly.
    winning_actions = (stats.tackles + stats.interceptions)
    baseline_actions = (b.tackles + b.interceptions)
    win_factor = 1.0 - 0.10 * max(-1.0, min(1.0,
                 (winning_actions - baseline_actions) / baseline_actions))

    return max(0.6, min(1.6, leak * win_factor))


def effective_ratings(team: Team, stats_blend: float = 0.5) -> Tuple[float, float]:
    """
    Blend the team's nominal attack/defense ratings with the values implied by
    underlying stats. `stats_blend` in [0, 1] sets how much to trust the stats
    (0.5 = weight them equally with the manual ratings). If the team has no
    stats attached, the nominal ratings are returned unchanged.
    """
    if team.stats is None:
        return team.attack, team.defense
    att = (1 - stats_blend) * team.attack + stats_blend * stats_attack_index(team.stats)
    deff = (1 - stats_blend) * team.defense + stats_blend * stats_defense_index(team.stats)
    return att, deff


# --------------------------------------------------------------------------- #
# Expected goals (the heart of the formula).
# --------------------------------------------------------------------------- #

def expected_goals(
    home: Team,
    away: Team,
    neutral: bool = True,
    cfg: ModelConfig = CONFIG,
) -> Tuple[float, float]:
    """
    Compute (lambda_home, lambda_away): the expected goals for each side.

    For the home side:

        lambda_home = league_avg
                      * home.attack * away.defense          # ratings
                      * elo_factor                          # overall strength gap
                      * form_factor_home                    # recent form
                      * style_factor_home                   # tactical matchup

    The away side is symmetric with the Elo factor inverted and no home edge.
    All modulating factors are exp(coef * signal), so they compose
    multiplicatively and stay strictly positive.
    """
    home_elo = home.elo + (0.0 if neutral else cfg.home_field_elo)
    # For a neutral World Cup we still let the *host* be passed as non-neutral.
    elo_diff = home_elo - away.elo

    elo_factor_home = math.exp(cfg.elo_beta * elo_diff)
    elo_factor_away = math.exp(-cfg.elo_beta * elo_diff)

    form_h = math.exp(cfg.form_strength * form_score(home))
    form_a = math.exp(cfg.form_strength * form_score(away))

    style_h = math.exp(cfg.tactics_strength * style_edge(home.style, away.style))
    style_a = math.exp(cfg.tactics_strength * style_edge(away.style, home.style))

    # Ratings, blended with underlying shot/tackle/offside stats when available.
    att_h, def_h = effective_ratings(home, cfg.stats_blend)
    att_a, def_a = effective_ratings(away, cfg.stats_blend)

    lam_home = (
        cfg.league_avg_goals
        * att_h * def_a
        * elo_factor_home * form_h * style_h
    )
    lam_away = (
        cfg.league_avg_goals
        * att_a * def_h
        * elo_factor_away * form_a * style_a
    )

    # Guard rails: keep expected goals in a sane football range.
    lam_home = max(0.15, min(5.0, lam_home))
    lam_away = max(0.15, min(5.0, lam_away))
    return lam_home, lam_away


# --------------------------------------------------------------------------- #
# Scoreline distribution: Dixon-Coles corrected Poisson.
# --------------------------------------------------------------------------- #

def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _dc_tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score correction term."""
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def scoreline_matrix(
    lam_home: float, lam_away: float, cfg: ModelConfig = CONFIG
) -> List[List[float]]:
    """Probability grid P[i][j] = P(home scores i, away scores j)."""
    n = cfg.max_goals + 1
    grid = [[0.0] * n for _ in range(n)]
    total = 0.0
    for i in range(n):
        for j in range(n):
            p = (
                _poisson_pmf(i, lam_home)
                * _poisson_pmf(j, lam_away)
                * _dc_tau(i, j, lam_home, lam_away, cfg.dc_rho)
            )
            grid[i][j] = p
            total += p
    # Renormalize (tau and the goal cap perturb the total slightly).
    if total:
        for i in range(n):
            for j in range(n):
                grid[i][j] /= total
    return grid


# --------------------------------------------------------------------------- #
# Market probabilities derived from the scoreline grid.
# --------------------------------------------------------------------------- #

@dataclass
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    under_2_5: float
    btts_yes: float
    btts_no: float
    lambda_home: float
    lambda_away: float
    top_scores: List[Tuple[str, float]]
    grid: List[List[float]] = field(default_factory=list)

    # --- Derived secondary markets (computed from the grid) ----------------- #
    @property
    def double_chance_1x(self) -> float:   # home win or draw
        return self.home_win + self.draw

    @property
    def double_chance_12(self) -> float:   # either team wins (no draw)
        return self.home_win + self.away_win

    @property
    def double_chance_x2(self) -> float:   # away win or draw
        return self.away_win + self.draw

    @property
    def dnb_home(self) -> float:           # draw-no-bet: stake refunded on draw
        denom = self.home_win + self.away_win
        return self.home_win / denom if denom else 0.0

    @property
    def dnb_away(self) -> float:
        denom = self.home_win + self.away_win
        return self.away_win / denom if denom else 0.0


def match_probabilities(
    home: Team, away: Team, neutral: bool = True, cfg: ModelConfig = CONFIG
) -> MatchProbabilities:
    lam_h, lam_a = expected_goals(home, away, neutral=neutral, cfg=cfg)
    grid = scoreline_matrix(lam_h, lam_a, cfg)

    p_home = p_draw = p_away = 0.0
    p_over = p_btts = 0.0
    scores: List[Tuple[str, float]] = []

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            p = grid[i][j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i + j >= 3:
                p_over += p
            if i >= 1 and j >= 1:
                p_btts += p
            scores.append((f"{i}-{j}", p))

    scores.sort(key=lambda s: s[1], reverse=True)
    return MatchProbabilities(
        home_win=p_home,
        draw=p_draw,
        away_win=p_away,
        over_2_5=p_over,
        under_2_5=1.0 - p_over,
        btts_yes=p_btts,
        btts_no=1.0 - p_btts,
        lambda_home=lam_h,
        lambda_away=lam_a,
        top_scores=scores[:5],
        grid=grid,
    )


# --------------------------------------------------------------------------- #
# Secondary markets computed from the scoreline grid.
# --------------------------------------------------------------------------- #

def total_goals_prob(grid: List[List[float]], line: float, over: bool) -> float:
    """P(total goals over/under `line`). Use .5 lines to avoid pushes."""
    p = 0.0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            total = i + j
            if (over and total > line) or (not over and total < line):
                p += grid[i][j]
    return p


def correct_score_probs(
    grid: List[List[float]], top_n: int = 8
) -> List[Tuple[str, float]]:
    """All correct-score probabilities, sorted, truncated to top_n."""
    scores = [
        (f"{i}-{j}", grid[i][j])
        for i in range(len(grid))
        for j in range(len(grid[i]))
    ]
    scores.sort(key=lambda s: s[1], reverse=True)
    return scores[:top_n]


def asian_handicap_outcomes(
    grid: List[List[float]], line: float, side: str = "home"
) -> Tuple[float, float, float]:
    """
    Probabilities of (win, push, loss) when backing one side at an Asian
    handicap `line` (the handicap added to that side's goals). Handles half,
    whole, and quarter lines. `side` is "home" or "away".

    Example: side="home", line=-1.0  -> home must win by 2+ to win, by exactly
    1 is a push (stake refunded), otherwise loss.
    """
    # Quarter lines (e.g. -0.75) split the stake across the two neighbouring
    # half/whole lines.
    frac = round(abs(line) * 4) % 4
    if frac in (1, 3):  # quarter line
        lo = math.floor(line * 2) / 2
        hi = math.ceil(line * 2) / 2
        w1, p1, l1 = asian_handicap_outcomes(grid, lo, side)
        w2, p2, l2 = asian_handicap_outcomes(grid, hi, side)
        return ((w1 + w2) / 2, (p1 + p2) / 2, (l1 + l2) / 2)

    win = push = loss = 0.0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            p = grid[i][j]
            margin = (i - j) if side == "home" else (j - i)
            adjusted = margin + line
            if adjusted > 1e-9:
                win += p
            elif adjusted < -1e-9:
                loss += p
            else:
                push += p
    return win, push, loss


# --------------------------------------------------------------------------- #
# Betting layer: fair odds, value detection, Kelly staking.
# --------------------------------------------------------------------------- #

def fair_odds(prob: float) -> float:
    """Fair decimal odds for a probability (inf if prob is ~0)."""
    return 1.0 / prob if prob > 1e-9 else float("inf")


@dataclass
class ValueBet:
    market: str
    model_prob: float
    fair_odds: float
    book_odds: float
    edge: float          # expected value per unit staked, i.e. p*odds - 1
    kelly_stake: float   # fraction of bankroll (already fractional-Kelly scaled)


def _devigged_implied(book_odds: Dict[str, float]) -> Dict[str, float]:
    """
    Bookmaker-implied probabilities with the margin ("vig") removed, so they sum
    to 1 within each complementary market group. Markets outside a known group
    fall back to the raw 1/odds.
    """
    groups = [
        ("Home win", "Draw", "Away win"),
        ("BTTS Yes", "BTTS No"),
        ("DNB Home", "DNB Away"),
        ("Over 0.5", "Under 0.5"), ("Over 1.5", "Under 1.5"),
        ("Over 2.5", "Under 2.5"), ("Over 3.5", "Under 3.5"),
        ("Over 4.5", "Under 4.5"),
    ]
    implied: Dict[str, float] = {}
    used = set()
    for grp in groups:
        present = [m for m in grp if m in book_odds and book_odds[m] > 1.0]
        if len(present) >= 2:
            raw = {m: 1.0 / book_odds[m] for m in present}
            total = sum(raw.values())
            for m in present:
                implied[m] = raw[m] / total
                used.add(m)
    for m, o in book_odds.items():
        if m not in used and o > 1.0:
            implied[m] = 1.0 / o
    return implied


def evaluate_bets(
    probs: MatchProbabilities,
    book_odds: Dict[str, float],
    edge_threshold: float = 0.03,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.05,
    market_blend: float = 0.0,
) -> List[ValueBet]:
    """
    Compare model probabilities to bookmaker decimal odds and return only the
    +EV ("value") bets.

      edge  = model_prob * book_odds - 1      (expected profit per unit)
      kelly = edge / (book_odds - 1)          (full-Kelly fraction)
      stake = clamp(kelly * kelly_fraction, 0, max_stake)

    Using fractional Kelly (default 1/4) and a stake cap keeps variance and
    model-error risk in check — full Kelly is notoriously aggressive.

    `market_blend` in [0, 1] is the professional safeguard: instead of trusting
    the model outright, shrink each model probability toward the (de-vigged)
    market price before judging value. 0 = pure model (aggressive, noisy);
    higher = defer more to the sharp market, so only disagreements the model is
    confident about survive. Around 0.3-0.6 is typical once calibrated.
    """
    implied = _devigged_implied(book_odds) if market_blend > 0 else {}
    market_probs = {
        "Home win": probs.home_win,
        "Draw": probs.draw,
        "Away win": probs.away_win,
        "Over 2.5": probs.over_2_5,
        "Under 2.5": probs.under_2_5,
        "BTTS Yes": probs.btts_yes,
        "BTTS No": probs.btts_no,
        # Double chance
        "DC 1X": probs.double_chance_1x,
        "DC 12": probs.double_chance_12,
        "DC X2": probs.double_chance_x2,
        # Draw-no-bet (push refunds handled by the conditional probability)
        "DNB Home": probs.dnb_home,
        "DNB Away": probs.dnb_away,
    }
    # Extra totals lines straight off the grid (e.g. "Over 1.5", "Under 3.5").
    if probs.grid:
        for line in (0.5, 1.5, 3.5, 4.5):
            market_probs[f"Over {line}"] = total_goals_prob(probs.grid, line, True)
            market_probs[f"Under {line}"] = total_goals_prob(probs.grid, line, False)
        # Correct-score markets, keyed "CS i-j".
        for score, p in correct_score_probs(probs.grid, top_n=12):
            market_probs[f"CS {score}"] = p

    bets: List[ValueBet] = []
    for market, odds in book_odds.items():
        if odds <= 1.0:
            continue

        # Asian handicap markets are evaluated separately because of pushes.
        if market.startswith("AH "):
            ah = _evaluate_asian_handicap(
                probs.grid, market, odds, edge_threshold,
                kelly_fraction, max_stake,
            )
            if ah:
                bets.append(ah)
            continue

        if market not in market_probs:
            continue
        p = market_probs[market]
        # Professional safeguard: shrink toward the sharp, de-vigged market.
        if market_blend > 0 and market in implied:
            p = (1.0 - market_blend) * p + market_blend * implied[market]
        edge = p * odds - 1.0
        if edge <= edge_threshold:
            continue
        full_kelly = edge / (odds - 1.0)
        stake = max(0.0, min(max_stake, full_kelly * kelly_fraction))
        bets.append(
            ValueBet(
                market=market,
                model_prob=p,
                fair_odds=fair_odds(p),
                book_odds=odds,
                edge=edge,
                kelly_stake=stake,
            )
        )
    bets.sort(key=lambda b: b.edge, reverse=True)
    return bets


def _evaluate_asian_handicap(
    grid: List[List[float]],
    market: str,
    odds: float,
    edge_threshold: float,
    kelly_fraction: float,
    max_stake: float,
) -> Optional[ValueBet]:
    """
    Evaluate an Asian-handicap bet keyed like "AH Home -1.0" / "AH Away +0.5".

    A push refunds the stake, so the expected value of a unit stake is:
        EV = p_win * (odds - 1) + p_push * 0 - p_loss * 1
    and the edge (profit per unit) is EV. We convert the win/push split into an
    effective probability so Kelly staking still applies.
    """
    if not grid:
        return None
    parts = market.split()
    if len(parts) != 3:
        return None
    side = parts[1].lower()
    try:
        line = float(parts[2])
    except ValueError:
        return None
    if side not in ("home", "away"):
        return None

    p_win, p_push, p_loss = asian_handicap_outcomes(grid, line, side)
    edge = p_win * (odds - 1.0) - p_loss          # = EV per unit staked
    if edge <= edge_threshold:
        return None
    # Effective win prob for a push-refunding bet: solve p_eff*odds - 1 = edge.
    p_eff = (edge + 1.0) / odds
    full_kelly = edge / (odds - 1.0)
    stake = max(0.0, min(max_stake, full_kelly * kelly_fraction))
    return ValueBet(
        market=market,
        model_prob=p_eff,
        fair_odds=fair_odds(p_eff),
        book_odds=odds,
        edge=edge,
        kelly_stake=stake,
    )


# --------------------------------------------------------------------------- #
# Convenience: full report for one fixture.
# --------------------------------------------------------------------------- #

def analyze_match(
    home: Team,
    away: Team,
    book_odds: Optional[Dict[str, float]] = None,
    neutral: bool = True,
    cfg: ModelConfig = CONFIG,
    edge_threshold: float = 0.03,
    market_blend: float = 0.0,
) -> Tuple[MatchProbabilities, List[ValueBet]]:
    probs = match_probabilities(home, away, neutral=neutral, cfg=cfg)
    bets = (evaluate_bets(probs, book_odds, edge_threshold=edge_threshold,
                          market_blend=market_blend)
            if book_odds else [])
    return probs, bets
