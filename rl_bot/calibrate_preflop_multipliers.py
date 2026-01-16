"""
Calibrate suitedness/connectivity multipliers for preflop scoring.

This script samples random 3-card hands, estimates their equity via Monte Carlo,
and performs a small grid search to find multipliers that best align the
preflop score with equity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_bot.player import Player, RANK_TO_INT  # noqa: E402


@dataclass(frozen=True)
class HandSample:
    ranks: tuple[int, int, int]
    suits: tuple[str, str, str]
    is_triple_suited: bool
    is_double_suited: bool
    is_conn_both: bool
    is_conn_one: bool
    base_points: float
    equity: float


def build_deck() -> list[str]:
    return [r + s for r in "23456789TJQKA" for s in "cdhs"]


def score_hand(points: float, sample: HandSample, suited_mults: tuple[float, float],
               connected_mults: tuple[float, float]) -> float:
    suited_triple, suited_double = suited_mults
    connected_both, connected_one = connected_mults

    if sample.is_triple_suited:
        points *= suited_triple
    elif sample.is_double_suited:
        points *= suited_double

    if sample.is_conn_both:
        points *= connected_both
    elif sample.is_conn_one:
        points *= connected_one

    return points


def normalize(values: list[float]) -> list[float]:
    v_min = min(values)
    v_max = max(values)
    span = max(1e-9, v_max - v_min)
    return [(v - v_min) / span for v in values]


def main() -> None:
    rng = random.Random(7)
    player = Player()

    total_samples = 120
    equity_iters = 60

    deck = build_deck()
    samples: list[HandSample] = []

    for _ in range(total_samples):
        rng.shuffle(deck)
        hand = deck[:3]

        ranks = sorted([RANK_TO_INT[c[0]] for c in hand], reverse=True)
        suits = [c[1] for c in hand]

        points = float(sum(ranks))
        if ranks[0] == ranks[1] or ranks[1] == ranks[2] or ranks[0] == ranks[2]:
            points += 20
            if ranks[0] == ranks[2]:
                points += 30

        gap1 = ranks[0] - ranks[1]
        gap2 = ranks[1] - ranks[2]

        is_triple = suits[0] == suits[1] == suits[2]
        is_double = not is_triple and (suits[0] == suits[1] or suits[1] == suits[2] or suits[0] == suits[2])

        is_conn_both = gap1 == 1 and gap2 == 1
        is_conn_one = (gap1 == 1 or gap2 == 1) and not is_conn_both

        equity = player.calculate_equity_python(hand, [], iterations=equity_iters)

        samples.append(
            HandSample(
                ranks=tuple(ranks),
                suits=tuple(suits),
                is_triple_suited=is_triple,
                is_double_suited=is_double,
                is_conn_both=is_conn_both,
                is_conn_one=is_conn_one,
                base_points=points,
                equity=equity,
            )
        )

    equities = [s.equity for s in samples]
    equities_norm = normalize(equities)

    suited_triple_grid = [1.12, 1.16, 1.20, 1.24]
    suited_double_grid = [1.04, 1.06, 1.08, 1.10]
    connected_both_grid = [1.12, 1.16, 1.18, 1.20]
    connected_one_grid = [1.04, 1.06, 1.07, 1.08]

    best = None
    best_mse = None

    for suited_triple in suited_triple_grid:
        for suited_double in suited_double_grid:
            for connected_both in connected_both_grid:
                for connected_one in connected_one_grid:
                    preds = [
                        score_hand(
                            s.base_points,
                            s,
                            (suited_triple, suited_double),
                            (connected_both, connected_one),
                        )
                        for s in samples
                    ]
                    preds_norm = normalize(preds)
                    mse = sum(
                        (pred - eq) ** 2
                        for pred, eq in zip(preds_norm, equities_norm)
                    ) / len(samples)

                    if best_mse is None or mse < best_mse:
                        best_mse = mse
                        best = (suited_triple, suited_double, connected_both, connected_one)

    print("Best multipliers:")
    print(f"  triple suited: {best[0]:.2f}")
    print(f"  double suited: {best[1]:.2f}")
    print(f"  connected both: {best[2]:.2f}")
    print(f"  connected one: {best[3]:.2f}")
    print(f"MSE: {best_mse:.6f}")


if __name__ == "__main__":
    main()
