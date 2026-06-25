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

### 5b. Markets covered

Everything below is derived from the same scoreline grid, so the prices are
internally consistent:

- **1X2** — home / draw / away
- **Double chance** — `DC 1X`, `DC 12`, `DC X2`
- **Draw-no-bet** — `DNB Home`, `DNB Away` (stake refunded on a draw)
- **Totals** — `Over/Under` at 0.5, 1.5, 2.5, 3.5, 4.5
- **BTTS** — both teams to score, yes / no
- **Correct score** — `CS 2-0`, `CS 1-1`, … (top scorelines)
- **Asian handicap** — `AH Home -1.0`, `AH Away +0.5`, quarter lines like
  `AH Home -0.75`; pushes (stake refunded) are handled correctly in the EV and
  Kelly math

To evaluate any of these, just add the key with the bookmaker's decimal odds to
`book_odds` (see `example.py`).

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

## Batch a whole matchday from CSV

Instead of editing Python, keep your teams and fixtures in spreadsheets:

- **`teams.csv`** — one row per team: `elo, attack, defense, style`, the
  underlying stats (`shots_for, sot_for, big_for, offsides_for, possession,
  shots_against, sot_against, big_against, tackles, interceptions`), and an
  optional `form` column written as `2-0@1850;1-0@1900;0-0` (goals for-against,
  optional `@opponentElo`, most recent first).
- **`fixtures.csv`** — one row per match: `home, away, neutral` plus odds
  columns (`home_win, draw, away_win, over25, under25, btts_yes, btts_no,
  dc_1x, dc_12, dc_x2, dnb_home, dnb_away`). Leave any odds cell blank to skip
  that market.

```bash
python3 batch.py                 # uses teams.csv + fixtures.csv
python3 batch.py teams.csv my_saturday.csv
```

It prints each fixture's probabilities and value bets, then a **matchday value
board** ranking every +EV bet across the slate by edge.

## Accumulators / parlays

`batch.py` also suggests **parlays** built only from legs the model already
rates as +EV singles, ranked by combined edge (`parlay.py` has the standalone
API). Two safeguards are baked in:

- **One leg per fixture.** Legs from the same match are correlated (e.g. "Home
  win" and "Over 2.5" tend to come together), which breaks the
  multiply-the-probabilities math, so the builder keeps legs independent by
  default.
- **Edge must survive stacking.** A parlay's combined edge is
  `(∏ model_prob) × (∏ odds) − 1`; the bookmaker margin multiplies too, so only
  parlays that still clear the threshold are shown, with a deliberately small
  fractional-Kelly stake (parlay variance is high).

Reality check: even +EV parlays are higher-variance than the equivalent singles.
They're a smaller-stake, higher-upside play, not a shortcut.

## Auto-fill the data from a live API

`fetch_data.py` populates `teams.csv` from **API-Football** (free tier) — recent
fixtures, results-based form, and per-match shot stats:

```bash
export APIFOOTBALL_KEY=your_key_here        # free at dashboard.api-football.com
python3 fetch_data.py --league 1 --season 2022 \
    --teams "Brazil,Argentina,France,Spain,Morocco" --out teams.csv
```

It writes a usable row per team, falling back to league-average defaults for any
stat the API doesn't return. Two fields are left for you to set by judgement
after the pull: **`style`** (the tactical identity) and **`elo`** (a light proxy
is derived from goal difference; swap in FIFA / World-Football-Elo numbers if you
have them). Without a key the script explains how to get one and exits cleanly;
network calls go through the environment proxy and never disable TLS.

## Backtest the strategy

`backtest.py` replays the model over historical matches to check whether the
"edges" actually hold up. Give it `results.csv` (fixtures + actual
`home_goals`/`away_goals` + the odds you could have taken), and for every match
it places the value bets the model would have flagged, settles them against the
real score, and reports:

```bash
python3 backtest.py               # teams.csv + results.csv
python3 backtest.py teams.csv history.csv
```

- **ROI (flat stake)** — profit per unit, 1 unit on every value bet
- **ROI (Kelly)** — bankroll growth using the model's fractional-Kelly stakes
- **Hit rate** — share of settled bets that won (pushes excluded)
- **Brier score** — probability calibration error (0.25 = a coin flip; lower is
  better)
- **Reliability table** — predicted vs actual win-rate, bucketed, so you can see
  if e.g. your "60%" bets actually win ~60%
- **Closing-line value** — if `results.csv` has `*_close` odds columns, whether
  you beat the closing price (the single best predictor of a real long-term edge)

Add `home_goals`/`away_goals` to mark a match as played; rows without a result
are skipped, so the same file can hold both upcoming and finished fixtures.

> ⚠️ **Look-ahead caveat.** The backtest uses the *current* `teams.csv` ratings
> for every past match. For an honest test, feed ratings/form *as they were
> before* each match (point-in-time). Treat the built-in run as a pipeline
> sanity check, not proof of a live edge.

## Simulate the whole tournament

`simulate.py` runs a Monte-Carlo of the entire World Cup — group stage round
robins, then a seeded knockout bracket — sampling each match's scoreline from
the model. It reports each team's chance of advancing, reaching each round, and
winning the title, with **fair outright odds** you can compare to the futures
market.

```bash
python3 simulate.py            # 10,000 simulations from teams.csv + groups.csv
python3 simulate.py 50000      # more sims = tighter estimates
```

`groups.csv` holds the draw (`group, team`); knockouts use extra time and a
slight Elo-weighted edge on penalties. It supports 4 groups of 4 (→ quarters)
or 8 groups of 4 (→ round of 16).

### Using it with Stake (decimal odds)

The model works directly with **decimal odds**, which is Stake's default display
format — so you can copy the prices straight off the Stake sportsbook into
`book_odds`. (If your account shows a different format, switch the odds display
to *Decimal* in settings first.) The market names map cleanly:

| `book_odds` key | Stake market |
|-----------------|--------------|
| `Home win` / `Draw` / `Away win` | Match Result / 1X2 (Full Time) |
| `DC 1X` / `DC 12` / `DC X2` | Double Chance |
| `DNB Home` / `DNB Away` | Draw No Bet |
| `Over/Under 2.5` (and 0.5–4.5) | Total Goals Over/Under |
| `BTTS Yes` / `BTTS No` | Both Teams To Score |
| `CS 2-0`, `CS 1-1`, … | Correct Score |
| `AH Home -1.0`, `AH Away +0.5` | Asian Handicap |

Use the **Full Time / 90-minute** market lines (not including extra time) so
they match the model, which predicts regulation-time goals. The model does not
connect to Stake or place bets — you read its output and decide for yourself.

---

## ⚠️ Reality check

This is a modeling tool, not a money printer. A model is only as good as its
inputs, the true outcome is genuinely uncertain, and bookmakers price sharply.
Treat any "edge" as an estimate, never stake more than you can lose, and be
aware of the law where you live. Gamble responsibly.
