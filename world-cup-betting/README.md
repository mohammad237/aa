# World Cup Match Betting Model

A transparent, dependency-free model that turns team data into match-outcome
probabilities, fair odds, and value-bet / staking recommendations. It accounts
for the things you asked about:

- **Overall ratings** (Elo)
- **Recent games / form** (weighted last-N results)
- **Tactics & playing style** (a style-matchup matrix)
- **Underlying match stats** — shots, shots on target, big chances, tackles,
  interceptions, offsides, possession

No external libraries. Run it with `python3`.

```bash
cd world-cup-betting
python3 example.py
```

---

## The formula, layer by layer

The engine is an **expected-goals model**. For each side we compute an expected
number of goals (lambda), then turn the two lambdas into a full distribution of
scorelines, and from that every market probability.

### 1. Expected goals

For the "home" side (venue is neutral by default at a World Cup):

```
lambda_home = league_avg_goals
            * attack_home * defense_away      # ratings (blended with stats)
            * elo_factor                       # overall strength gap
            * form_factor_home                 # recent form
            * style_factor_home                # tactical matchup
```

The away side is symmetric (Elo factor inverted, no home edge). Every
modulating factor is `exp(coefficient * signal)`, so they multiply together
cleanly and can never produce a negative goal rate.

### 2. Ratings + underlying stats

Each team has nominal `attack` and `defense` ratings (1.0 = average,
attack > 1 scores more, defense > 1 is leakier). If you attach a `TeamStats`
object, the model derives **independent attack/defense indices from the
underlying numbers** and blends them with the manual ratings
(`stats_blend`, default 0.5):

- **Attack index** is weighted toward *shot quality* — shots on target (0.42)
  and big chances (0.30) matter far more than raw shot count (0.16), with small
  contributions from possession and offsides (attacking intent in behind).
- **Defense (leakiness) index** is driven by the *quality of chances conceded*
  — shots on target against, big chances against, shots against — and is
  reduced by ball-winning volume (tackles + interceptions).

Why blend stats in? Goals are noisy over a short World Cup sample; shot volume
and shot quality are much more stable and predictive, so they keep the model
from over-reacting to a lucky / unlucky goal tally.

### 3. Form (previous games)

`form_score` reads each team's recent results (most-recent first) and produces a
single number in roughly `[-1, 1]`:

- **Recency weighting** — exponential decay (default 3-game half-life), so last
  week counts more than a month ago.
- **Result + margin** — win/draw/loss blended with capped goal difference.
- **Opponent quality** — beating a strong side (high opponent Elo) is worth
  more than beating a weak one.
- **Match importance** — knockout games can be weighted above friendlies.

### 4. Tactics & playing style

Each team is tagged with a style: `possession`, `high_press`, `counter`,
`low_block`, `direct`, or `balanced`. A **matchup matrix** encodes how styles
interact — e.g. a `counter` side gets an attacking edge against `possession`
(space in behind), while `high_press` punishes slow build-up but is vulnerable
to `direct`. The edge is small and scaled by `tactics_strength` so it nudges
rather than dominates.

### 5. Scoreline distribution (Dixon–Coles Poisson)

The two lambdas feed independent Poisson distributions to build a grid of
`P(home i goals, away j goals)`, with the **Dixon–Coles low-score correction**
(`rho`) that fixes the well-known flaw of plain Poisson under-counting draws and
0-0 / 1-1 results. Summing the grid gives:

- 1X2: home win / draw / away win
- Over/Under 2.5 goals
- Both teams to score (BTTS)
- Most-likely scorelines

### 6. Betting layer — value & staking

Each model probability is compared to the bookmaker's decimal odds:

```
edge  = model_prob * book_odds - 1        # expected profit per unit staked
kelly = edge / (book_odds - 1)            # full-Kelly fraction of bankroll
stake = clamp(kelly * kelly_fraction, 0, max_stake)
```

Only **+EV bets** above an edge threshold are returned. Staking uses
**fractional Kelly** (default 1/4) with a hard cap, because full Kelly is far
too aggressive once you account for model error.

---

## Tuning

All knobs live in `ModelConfig` at the top of `model.py`:

| Knob | Meaning | Default |
|------|---------|---------|
| `league_avg_goals` | baseline goals per side | 1.35 |
| `home_field_elo` | host/home edge in Elo points | 35 |
| `elo_beta` | how hard the Elo gap bends goals | 0.0016 |
| `form_strength` | swing from recent form | 0.18 |
| `tactics_strength` | swing from style matchup | 0.12 |
| `stats_blend` | trust in underlying stats vs manual ratings | 0.5 |
| `dc_rho` | Dixon–Coles draw correction | −0.08 |

Betting knobs (`evaluate_bets`): `edge_threshold` (0.03), `kelly_fraction`
(0.25), `max_stake` (0.05).

---

## How to use it for a real fixture

1. Fill in each `Team`: Elo, attack/defense, style.
2. Add the last ~5 results as `MatchResult(goals_for, goals_against,
   opponent_elo, importance)`.
3. Attach a `TeamStats` with per-match averages (shots, SoT, big chances,
   tackles, interceptions, offsides, possession).
4. Paste the bookmaker's decimal odds into `book_odds`.
5. Run — read the probabilities, fair odds, and any flagged value bets.

See `example.py` for a complete worked fixture.

### Using it with Stake (decimal odds)

The model works directly with **decimal odds**, which is Stake's default display
format — so you can copy the prices straight off the Stake sportsbook into
`book_odds`. (If your account shows a different format, switch the odds display
to *Decimal* in settings first.) The market names map cleanly:

| `book_odds` key | Stake market |
|-----------------|--------------|
| `Home win` / `Draw` / `Away win` | Match Result / 1X2 (Full Time) |
| `Over 2.5` / `Under 2.5` | Total Goals Over/Under 2.5 |
| `BTTS Yes` / `BTTS No` | Both Teams To Score |

Use the **Full Time / 90-minute** market lines (not including extra time) so
they match the model, which predicts regulation-time goals. The model does not
connect to Stake or place bets — you read its output and decide for yourself.

---

## ⚠️ Reality check

This is a modeling tool, not a money printer. A model is only as good as its
inputs, the true outcome is genuinely uncertain, and bookmakers price sharply.
Treat any "edge" as an estimate, never stake more than you can lose, and be
aware of the law where you live. Gamble responsibly.
