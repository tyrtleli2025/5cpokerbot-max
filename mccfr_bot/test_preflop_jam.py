import unittest

from player import Player
from skeleton.actions import CallAction, FoldAction


class DummyRoundState:
    def __init__(self, pips, stacks):
        self.pips = pips
        self.stacks = stacks

    def legal_actions(self):
        return {CallAction, FoldAction}


class PreflopJamTests(unittest.TestCase):
    def setUp(self):
        self.player = Player()
        self.player.precomputed_strategy = None

    def test_villain_type_tight_pf_jammer(self):
        self.player.pf_jam_opportunities = 20
        self.player.pf_jams = 12
        self.player.pf_jam_showdowns = 5
        self.player.pf_jam_strong_showdowns = 4
        self.assertEqual(self.player.villain_type(), "tight_pf_jammer")

    def test_villain_type_likely_pf_jammer(self):
        self.player.pf_jam_opportunities = 10
        self.player.pf_jams = 8
        self.player.pf_jam_showdowns = 0
        self.player.pf_jam_strong_showdowns = 0
        self.assertEqual(self.player.villain_type(), "likely_pf_jammer")

    def test_jam_decision_premium_calls(self):
        self.player.pf_jam_opportunities = 20
        self.player.pf_jams = 12
        self.player.pf_jam_showdowns = 6
        self.player.pf_jam_strong_showdowns = 5
        round_state = DummyRoundState([0, 100], [300, 0])

        decision = self.player.decide_vs_preflop_jam(["As", "Ah", "2d"], round_state, 0)
        self.assertIsInstance(decision, CallAction)

        decision = self.player.decide_vs_preflop_jam(["As", "Ks", "7d"], round_state, 0)
        self.assertIsInstance(decision, CallAction)

    def test_jam_decision_trash_folds(self):
        self.player.pf_jam_opportunities = 20
        self.player.pf_jams = 12
        self.player.pf_jam_showdowns = 6
        self.player.pf_jam_strong_showdowns = 5
        round_state = DummyRoundState([0, 100], [300, 0])

        decision = self.player.decide_vs_preflop_jam(["2s", "7d", "9c"], round_state, 0)
        self.assertIsInstance(decision, FoldAction)


if __name__ == "__main__":
    unittest.main()
