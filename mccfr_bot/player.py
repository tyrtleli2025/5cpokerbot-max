"""
MCCFR-inspired poker bot for the 3-card + discard variant.

This bot uses a lightweight Monte Carlo regret-matching loop to approximate a
GTO-like strategy in an abstracted state space. It samples hand equity with a
simple evaluator and updates per-information-set regrets each decision.
"""
import itertools
import os
import random
import sys
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKELETON_DIR = os.path.join(ROOT_DIR, "python_skeleton")
if SKELETON_DIR not in sys.path:
    sys.path.append(SKELETON_DIR)

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS, start=2)}

ALL_CARDS = [r + s for r in RANKS for s in SUITS]


class Card:
    def __init__(self, rank_char, suit_char):
        self.rank_char = rank_char
        self.suit_char = suit_char
        self.rank = RANK_TO_INT[rank_char]
        self.suit = suit_char

    def __repr__(self):
        return f"{self.rank_char}{self.suit_char}"

    def __str__(self):
        return f"{self.rank_char}{self.suit_char}"


def to_card(obj):
    if isinstance(obj, Card):
        return obj
    if isinstance(obj, str) and len(obj) >= 2:
        return Card(obj[0], obj[1])
    raise ValueError(f"Unsupported card format: {obj}")


def card_str(obj):
    if isinstance(obj, Card):
        return str(obj)
    return str(obj)


def evaluate_five(cards):
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    counts = Counter(ranks)
    unique_ranks = sorted(counts.keys(), reverse=True)

    is_flush = len(set(suits)) == 1

    is_straight = False
    straight_high = None
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    if is_flush and is_straight:
        return (8, [straight_high])

    if 4 in counts.values():
        four_rank = max(r for r, c in counts.items() if c == 4)
        kicker = max(r for r in ranks if r != four_rank)
        return (7, [four_rank, kicker])

    if sorted(counts.values()) == [2, 3]:
        trips_rank = max(r for r, c in counts.items() if c == 3)
        pair_rank = max(r for r, c in counts.items() if c == 2)
        return (6, [trips_rank, pair_rank])

    if is_flush:
        return (5, ranks)

    if is_straight:
        return (4, [straight_high])

    if 3 in counts.values():
        trips_rank = max(r for r, c in counts.items() if c == 3)
        kickers = [r for r in ranks if r != trips_rank]
        return (3, [trips_rank] + kickers)

    pairs = [r for r, c in counts.items() if c == 2]
    if len(pairs) == 2:
        pairs_sorted = sorted(pairs, reverse=True)
        kicker = max(r for r in ranks if r not in pairs_sorted)
        return (2, pairs_sorted + [kicker])

    if len(pairs) == 1:
        pair_rank = pairs[0]
        kickers = [r for r in ranks if r != pair_rank]
        return (1, [pair_rank] + kickers)

    return (0, ranks)


def evaluate_best(cards):
    if len(cards) < 5:
        ranks = sorted([c.rank for c in cards], reverse=True)
        return (0, ranks)
    best = None
    for combo in itertools.combinations(cards, 5):
        score = evaluate_five(combo)
        if best is None or score > best:
            best = score
    return best


def dealt_community_cards(street):
    if street == 0:
        return 0
    if street in (2, 3):
        return 3
    if street == 4:
        return 4
    return 5


def pick_best_keep(hand_cards, board_cards):
    best_keep = None
    best_discard = None
    best_score = None
    for idx in range(len(hand_cards)):
        discard = hand_cards[idx]
        keep = [c for j, c in enumerate(hand_cards) if j != idx]
        eval_cards = [to_card(c) for c in keep] + [to_card(c) for c in board_cards] + [to_card(discard)]
        score = evaluate_best(eval_cards)
        if best_score is None or score > best_score:
            best_score = score
            best_keep = keep
            best_discard = discard
    return best_keep, best_discard


class Player(Bot):
    def __init__(self):
        self.rng = random.Random(7)
        self.regrets = {}
        self.strategy_sum = {}
        self.equity_cache = {}
        self.mccfr_iterations = 4
        self.base_samples = 40

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.board

        if DiscardAction in legal:
            discard_idx = self.choose_discard(my_cards, board_cards)
            return DiscardAction(discard_idx)

        if street in (2, 3):
            return CheckAction()

        action_labels = self.available_action_labels(legal)
        if not action_labels:
            return CheckAction()

        equity = self.estimate_equity_cached(my_cards, board_cards, street)
        info_set = self.build_infoset(round_state, active, equity)

        for _ in range(self.mccfr_iterations):
            utilities = self.action_utilities(round_state, active, equity, action_labels)
            self.update_regrets(info_set, action_labels, utilities)

        strategy = self.get_strategy(info_set, action_labels)
        choice = self.sample_action(strategy)
        return self.to_action(choice, round_state, legal)

    def available_action_labels(self, legal):
        labels = []
        if FoldAction in legal:
            labels.append("fold")
        if CallAction in legal or CheckAction in legal:
            labels.append("call")
        if RaiseAction in legal:
            labels.append("raise")
        return labels

    def build_infoset(self, round_state, active, equity):
        pot = self.compute_pot(round_state)
        pot_bucket = min(20, pot // 10)
        equity_bucket = int(equity * 10)
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        pressure = "facing" if continue_cost > 0 else "free"
        return f"s{round_state.street}|p{active}|pb{pot_bucket}|e{equity_bucket}|{pressure}"

    def get_strategy(self, info_set, actions):
        regrets = self.regrets.setdefault(info_set, {a: 0.0 for a in actions})
        for action in actions:
            regrets.setdefault(action, 0.0)
        positive = [max(0.0, regrets[a]) for a in actions]
        normalizer = sum(positive)
        if normalizer > 0:
            strategy = {a: positive[i] / normalizer for i, a in enumerate(actions)}
        else:
            strategy = {a: 1.0 / len(actions) for a in actions}
        strat_sum = self.strategy_sum.setdefault(info_set, {a: 0.0 for a in actions})
        for action in actions:
            strat_sum[action] = strat_sum.get(action, 0.0) + strategy[action]
        return strategy

    def update_regrets(self, info_set, actions, utilities):
        strategy = self.get_strategy(info_set, actions)
        expected = sum(strategy[a] * utilities[a] for a in actions)
        regrets = self.regrets.setdefault(info_set, {a: 0.0 for a in actions})
        for action in actions:
            regrets[action] = regrets.get(action, 0.0) + (utilities[action] - expected)

    def sample_action(self, strategy):
        roll = self.rng.random()
        cumulative = 0.0
        for action, prob in strategy.items():
            cumulative += prob
            if roll <= cumulative:
                return action
        return next(iter(strategy))

    def action_utilities(self, round_state, active, equity, actions):
        my_contrib = STARTING_STACK - round_state.stacks[active]
        opp_contrib = STARTING_STACK - round_state.stacks[1 - active]
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        utilities = {}

        for action in actions:
            if action == "fold":
                utilities[action] = -my_contrib
                continue

            if action == "call":
                my_final = my_contrib + max(0, continue_cost)
                utilities[action] = (equity * opp_contrib) - ((1 - equity) * my_final)
                continue

            if action == "raise":
                raise_to = self.choose_raise_amount(round_state, active)
                my_cost = raise_to - round_state.pips[active]
                opp_cost = raise_to - round_state.pips[1 - active]
                my_final = my_contrib + my_cost
                opp_final = opp_contrib + opp_cost
                fold_prob = self.raise_fold_probability(equity)
                show_ev = (equity * opp_final) - ((1 - equity) * my_final)
                utilities[action] = fold_prob * opp_contrib + (1 - fold_prob) * show_ev

        return utilities

    def raise_fold_probability(self, equity):
        return max(0.1, min(0.7, 0.2 + (equity - 0.5) * 0.8))

    def choose_raise_amount(self, round_state, active):
        min_raise, max_raise = round_state.raise_bounds()
        pot = self.compute_pot(round_state)
        opp_pip = round_state.pips[1 - active]
        target = opp_pip + max(BIG_BLIND * 2, pot)
        return max(min_raise, min(max_raise, target))

    def compute_pot(self, round_state):
        return (STARTING_STACK * 2) - sum(round_state.stacks)

    def to_action(self, label, round_state, legal):
        if label == "raise" and RaiseAction in legal:
            return RaiseAction(self.choose_raise_amount(round_state, round_state.button % 2))
        if label == "call":
            if CallAction in legal:
                return CallAction()
            if CheckAction in legal:
                return CheckAction()
        if label == "fold":
            if CheckAction in legal:
                return CheckAction()
            return FoldAction()
        if CheckAction in legal:
            return CheckAction()
        return FoldAction()

    def choose_discard(self, my_cards, board_cards):
        if len(my_cards) <= 2:
            return 0
        best_idx = 0
        best_score = None
        for idx in range(len(my_cards)):
            keep = [c for i, c in enumerate(my_cards) if i != idx]
            discard = my_cards[idx]
            eval_cards = [to_card(c) for c in keep] + [to_card(c) for c in board_cards] + [to_card(discard)]
            score = evaluate_best(eval_cards)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def estimate_equity_cached(self, my_cards, board_cards, street):
        key = (tuple(sorted(map(card_str, my_cards))), tuple(sorted(map(card_str, board_cards))), street, len(my_cards))
        if key in self.equity_cache:
            return self.equity_cache[key]
        equity = self.estimate_equity(my_cards, board_cards, street, self.base_samples)
        self.equity_cache[key] = equity
        return equity

    def estimate_equity(self, my_cards, board_cards, street, samples):
        known_cards = {card_str(c) for c in my_cards} | {card_str(c) for c in board_cards}
        deck = [c for c in ALL_CARDS if c not in known_cards]

        community_dealt = dealt_community_cards(street)
        remaining_community = max(0, 5 - community_dealt)

        wins = 0
        ties = 0
        for _ in range(samples):
            draw = self.rng.sample(deck, 3 + remaining_community)
            opp_cards = draw[:3]
            community = draw[3:]
            full_board = list(board_cards) + community

            our_hand = list(my_cards)
            board_with_discards = list(full_board)
            if len(our_hand) == 3:
                our_keep, our_discard = pick_best_keep(our_hand, board_with_discards)
                board_with_discards.append(our_discard)
            else:
                our_keep = our_hand

            opp_hand = list(opp_cards)
            if len(opp_hand) == 3:
                opp_keep, opp_discard = pick_best_keep(opp_hand, board_with_discards)
                board_with_discards.append(opp_discard)
            else:
                opp_keep = opp_hand

            our_eval_cards = [to_card(c) for c in our_keep] + [to_card(c) for c in board_with_discards]
            opp_eval_cards = [to_card(c) for c in opp_keep] + [to_card(c) for c in board_with_discards]

            our_score = evaluate_best(our_eval_cards)
            opp_score = evaluate_best(opp_eval_cards)

            if our_score > opp_score:
                wins += 1
            elif our_score == opp_score:
                ties += 1

        return (wins + 0.5 * ties) / samples if samples > 0 else 0.0


if __name__ == "__main__":
    run_bot(Player(), parse_args())
