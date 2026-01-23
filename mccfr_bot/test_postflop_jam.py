import random
import unittest

from jam_defense import decide_vs_postflop_jam, default_config
from player import evaluate_best, card_rank, card_suit


class PostflopJamTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)
        self.config = default_config(evaluate_best)
        self.config["mode"] = "rules"

    def test_river_weak_pair_folds(self):
        hole = ["3c", "7d"]
        board = ["Ah", "Kd", "7s", "2c", "9h", "5d"]
        call, _, _ = decide_vs_postflop_jam(
            hole,
            board,
            "river",
            pot=100,
            to_call=80,
            stack_eff=100,
            stats={},
            config=self.config,
            rng=self.rng,
            card_rank=card_rank,
            card_suit=card_suit,
        )
        self.assertFalse(call)

    def test_river_two_pair_calls(self):
        hole = ["As", "7c"]
        board = ["Ah", "Kd", "7s", "2c", "9h", "5d"]
        call, _, _ = decide_vs_postflop_jam(
            hole,
            board,
            "river",
            pot=100,
            to_call=40,
            stack_eff=100,
            stats={},
            config=self.config,
            rng=self.rng,
            card_rank=card_rank,
            card_suit=card_suit,
        )
        self.assertTrue(call)

    def test_turn_nut_draw_calls(self):
        hole = ["Ah", "Kh"]
        board = ["2h", "7h", "Qc", "9d"]
        call, _, _ = decide_vs_postflop_jam(
            hole,
            board,
            "turn",
            pot=120,
            to_call=20,
            stack_eff=200,
            stats={},
            config=self.config,
            rng=self.rng,
            card_rank=card_rank,
            card_suit=card_suit,
        )
        self.assertTrue(call)

    def test_turn_weak_draw_folds(self):
        hole = ["6h", "8h"]
        board = ["2h", "7h", "Qc", "9d"]
        call, _, _ = decide_vs_postflop_jam(
            hole,
            board,
            "turn",
            pot=120,
            to_call=40,
            stack_eff=200,
            stats={},
            config=self.config,
            rng=self.rng,
            card_rank=card_rank,
            card_suit=card_suit,
        )
        self.assertFalse(call)

    def test_mc_determinism(self):
        config = default_config(evaluate_best)
        config["mode"] = "mc"
        rng = random.Random(7)
        hole = ["Ah", "Kh"]
        board = ["2h", "7h", "Qc", "9d"]
        first, _, _ = decide_vs_postflop_jam(
            hole,
            board,
            "turn",
            pot=120,
            to_call=30,
            stack_eff=200,
            stats={},
            config=config,
            rng=rng,
            card_rank=card_rank,
            card_suit=card_suit,
        )
        rng = random.Random(7)
        second, _, _ = decide_vs_postflop_jam(
            hole,
            board,
            "turn",
            pot=120,
            to_call=30,
            stack_eff=200,
            stats={},
            config=config,
            rng=rng,
            card_rank=card_rank,
            card_suit=card_suit,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
