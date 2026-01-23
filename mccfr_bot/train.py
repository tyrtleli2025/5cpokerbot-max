"""
Offline training utility for the MCCFR bot.

This script samples random abstract game states, updates regrets via the
Player's MCCFR logic, and writes an averaged strategy to strategy.pkl.
"""
import argparse
import os
import random

from player import Player, STRATEGY_PATH


class MockRoundState:
    def __init__(self, street, pips, stacks):
        self.street = street
        self.pips = pips
        self.stacks = stacks

    def legal_actions(self):
        if self.street in (2, 3):
            return set()
        continue_cost = self.pips[1] - self.pips[0]
        if continue_cost == 0:
            return {"check", "raise", "fold"}
        return {"call", "raise", "fold"}

    def raise_bounds(self):
        min_raise = self.pips[0] + 2
        max_raise = self.pips[0] + max(2, self.stacks[0])
        return min_raise, max_raise


def sample_state(rng):
    street = rng.choice([0, 4, 5, 6])
    pot = rng.randint(2, 60)
    stacks = [400 - pot // 2, 400 - pot // 2]
    if rng.random() < 0.5:
        pips = [0, 0]
    else:
        bet = rng.randint(2, 10)
        pips = [0, bet]
    return MockRoundState(street, pips, stacks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--output", type=str, default=STRATEGY_PATH)
    args = parser.parse_args()

    rng = random.Random(42)
    player = Player()
    player.precomputed_strategy = None

    for _ in range(args.iterations):
        mock_state = sample_state(rng)
        equity = rng.random()
        info_set = player.build_infoset(mock_state, 0, equity)
        actions = ["fold", "call", "raise"]
        utilities = player.action_utilities(mock_state, 0, equity, actions)
        player.update_regrets(info_set, actions, utilities)

    output_path = os.path.abspath(args.output)
    player.save_strategy(output_path)
    print(f"Saved strategy to {output_path}")


if __name__ == "__main__":
    main()
