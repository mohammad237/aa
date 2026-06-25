"""
Test suite for the betting model. Pure stdlib (unittest), no dependencies.

    python3 tests.py            # run everything
    python3 tests.py -v         # verbose

These cover the parts where a silent bug would quietly cost money: market
settlement, de-vigging, Asian-handicap pushes, probability sums, the Elo engine,
form, parlay math, and the end-to-end pipeline.
"""

import unittest

import model
from model import (
    Team, TeamStats, MatchResult,
    match_probabilities, evaluate_bets, asian_handicap_outcomes,
    _devigged_implied, total_goals_prob, expected_goals,
)
from ratings import EloEngine, EloConfig
from parlay import ParlayLeg, build_parlay
from backtest import settle


def make_match():
    home = Team("H", elo=1900, attack=1.2, defense=0.9, style="possession")
    away = Team("A", elo=1700, attack=0.95, defense=1.0, style="counter")
    return home, away


class TestSettlement(unittest.TestCase):
    def test_1x2(self):
        self.assertAlmostEqual(settle("Home win", 2, 0, 2.0), 1.0)
        self.assertAlmostEqual(settle("Home win", 0, 1, 2.0), -1.0)
        self.assertAlmostEqual(settle("Draw", 1, 1, 3.0), 2.0)
        self.assertAlmostEqual(settle("Away win", 0, 2, 4.0), 3.0)

    def test_double_chance_and_dnb(self):
        self.assertAlmostEqual(settle("DC 1X", 1, 1, 1.3), 0.3)   # draw counts
        self.assertAlmostEqual(settle("DC 12", 1, 1, 1.3), -1.0)  # draw loses
        self.assertEqual(settle("DNB Home", 1, 1, 1.8), 0.0)      # push on draw
        self.assertAlmostEqual(settle("DNB Home", 2, 0, 1.8), 0.8)

    def test_totals_and_btts(self):
        self.assertAlmostEqual(settle("Over 2.5", 2, 1, 2.0), 1.0)
        self.assertAlmostEqual(settle("Under 2.5", 1, 1, 2.0), 1.0)
        self.assertAlmostEqual(settle("BTTS Yes", 1, 1, 2.0), 1.0)
        self.assertAlmostEqual(settle("BTTS No", 2, 0, 1.8), 0.8)

    def test_correct_score(self):
        self.assertAlmostEqual(settle("CS 2-1", 2, 1, 8.0), 7.0)
        self.assertAlmostEqual(settle("CS 2-1", 1, 1, 8.0), -1.0)

    def test_asian_handicap_whole_line_push(self):
        # Home -1.0, home wins by exactly 1 -> push (stake returned).
        self.assertEqual(settle("AH Home -1.0", 1, 0, 1.95), 0.0)
        self.assertAlmostEqual(settle("AH Home -1.0", 2, 0, 1.95), 0.95)
        self.assertAlmostEqual(settle("AH Home -1.0", 0, 0, 1.95), -1.0)

    def test_asian_handicap_quarter_line(self):
        # Home -0.75 = half stake on -0.5, half on -1.0.
        # Win by 1: -0.5 wins, -1.0 pushes -> half win.
        pl = settle("AH Home -0.75", 1, 0, 2.0)
        self.assertAlmostEqual(pl, 0.5 * (2.0 - 1.0))
        # Draw: both lose -> full loss.
        self.assertAlmostEqual(settle("AH Home -0.75", 0, 0, 2.0), -1.0)

    def test_unknown_market(self):
        self.assertIsNone(settle("Nonsense", 1, 0, 2.0))


class TestProbabilities(unittest.TestCase):
    def test_probs_sum_to_one(self):
        home, away = make_match()
        p = match_probabilities(home, away)
        self.assertAlmostEqual(p.home_win + p.draw + p.away_win, 1.0, places=4)
        self.assertAlmostEqual(p.over_2_5 + p.under_2_5, 1.0, places=4)
        self.assertAlmostEqual(p.btts_yes + p.btts_no, 1.0, places=4)

    def test_favourite_has_higher_winprob(self):
        home, away = make_match()
        p = match_probabilities(home, away)
        self.assertGreater(p.home_win, p.away_win)

    def test_ah_outcomes_sum_to_one(self):
        home, away = make_match()
        p = match_probabilities(home, away)
        for line in (-1.0, -0.5, 0.5, 1.0):
            w, push, l = asian_handicap_outcomes(p.grid, line, "home")
            self.assertAlmostEqual(w + push + l, 1.0, places=4)

    def test_totals_monotonic(self):
        home, away = make_match()
        p = match_probabilities(home, away)
        # Over 0.5 must be >= Over 2.5 >= Over 4.5.
        o05 = total_goals_prob(p.grid, 0.5, True)
        o25 = total_goals_prob(p.grid, 2.5, True)
        o45 = total_goals_prob(p.grid, 4.5, True)
        self.assertGreaterEqual(o05, o25)
        self.assertGreaterEqual(o25, o45)


class TestDevig(unittest.TestCase):
    def test_devig_sums_to_one(self):
        odds = {"Home win": 1.70, "Draw": 3.80, "Away win": 6.00}
        implied = _devigged_implied(odds)
        self.assertAlmostEqual(sum(implied.values()), 1.0, places=6)

    def test_blend_reduces_edge(self):
        home, away = make_match()
        p = match_probabilities(home, away)
        odds = {"Home win": 1.0 / p.home_win * 1.10}  # 10% better than fair
        no_blend = evaluate_bets(p, odds, edge_threshold=-1, market_blend=0.0)
        blended = evaluate_bets(p, odds, edge_threshold=-1, market_blend=0.8)
        self.assertGreater(no_blend[0].edge, blended[0].edge)


class TestElo(unittest.TestCase):
    def test_expected_symmetry(self):
        self.assertAlmostEqual(EloEngine.expected(1500, 1500), 0.5)
        self.assertAlmostEqual(
            EloEngine.expected(1600, 1500) + EloEngine.expected(1500, 1600),
            1.0, places=6)

    def test_goal_multiplier(self):
        self.assertEqual(EloEngine.goal_multiplier(1), 1.0)
        self.assertEqual(EloEngine.goal_multiplier(2), 1.5)
        self.assertGreater(EloEngine.goal_multiplier(4), 1.5)

    def test_update_is_zero_sum_and_directional(self):
        eng = EloEngine(EloConfig())
        before_total = eng.rating("X") + eng.rating("Y")
        eng.update("X", "Y", 3, 0, neutral=True)
        after_total = eng.rating("X") + eng.rating("Y")
        self.assertAlmostEqual(before_total, after_total, places=6)  # zero-sum
        self.assertGreater(eng.rating("X"), eng.rating("Y"))         # winner up

    def test_as_of_team_uses_prior_only(self):
        eng = EloEngine(EloConfig(rolling_window=5))
        eng.update("X", "Y", 2, 0)
        eng.update("X", "Z", 1, 0)
        t = eng.as_of_team("X")
        self.assertEqual(len(t.recent), 2)
        self.assertGreater(t.elo, 1500)


class TestForm(unittest.TestCase):
    def test_form_sign(self):
        winner = Team("W", 1700, recent=[MatchResult(3, 0), MatchResult(2, 0)])
        loser = Team("L", 1700, recent=[MatchResult(0, 3), MatchResult(0, 2)])
        self.assertGreater(model.form_score(winner), 0)
        self.assertLess(model.form_score(loser), 0)

    def test_empty_form_is_zero(self):
        self.assertEqual(model.form_score(Team("E", 1700)), 0.0)


class TestParlay(unittest.TestCase):
    def test_combined_math(self):
        legs = [
            ParlayLeg("m1", "Home win", 0.5, 2.2),
            ParlayLeg("m2", "Over 2.5", 0.55, 2.0),
        ]
        p = build_parlay(legs)
        self.assertAlmostEqual(p.combined_odds, 4.4, places=6)
        self.assertAlmostEqual(p.combined_prob, 0.275, places=6)
        self.assertAlmostEqual(p.edge, 0.275 * 4.4 - 1.0, places=6)

    def test_same_fixture_rejected(self):
        legs = [
            ParlayLeg("same", "Home win", 0.5, 2.0),
            ParlayLeg("same", "Over 2.5", 0.5, 2.0),
        ]
        self.assertIsNone(build_parlay(legs))
        self.assertIsNotNone(build_parlay(legs, allow_same_fixture=True))


class TestEndToEnd(unittest.TestCase):
    def test_analyze_match_finds_value(self):
        home, away = make_match()
        p = match_probabilities(home, away)
        # Offer generous odds so a value bet must appear.
        odds = {"Home win": 1.0 / p.home_win * 1.20}
        _, bets = model.analyze_match(home, away, book_odds=odds)
        self.assertTrue(bets)
        self.assertEqual(bets[0].market, "Home win")

    def test_stats_blend_changes_lambda(self):
        home, away = make_match()
        leaky = TeamStats(shots_on_target_against=8.0, big_chances_against=4.0)
        away_leaky = Team("A", 1700, 0.95, 1.0, "counter", stats=leaky)
        lam_h1, _ = expected_goals(home, away)
        lam_h2, _ = expected_goals(home, away_leaky)
        self.assertGreater(lam_h2, lam_h1)  # leakier opponent -> more home goals


if __name__ == "__main__":
    unittest.main()
