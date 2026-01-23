"""
MCCFR-inspired poker bot for the 3-card + discard variant.

This bot uses a lightweight Monte Carlo regret-matching loop to approximate a
GTO-like strategy in an abstracted state space. It samples hand equity with a
simple evaluator and updates per-information-set regrets each decision.
"""
import itertools
import os
import pickle
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
STRATEGY_PATH = os.path.join(os.path.dirname(__file__), "strategy.pkl")

# Preflop jammer detection + response config.
PF_JAM_RATE_THRESHOLD = 0.45
PF_JAM_OPPORTUNITY_MIN = 20
PF_JAM_SHOWDOWN_MIN = 5
PF_JAM_STRONG_RATE_THRESHOLD = 0.70
PF_LIKELY_JAM_RATE = 0.70
PF_LIKELY_JAM_OPPORTUNITY_MIN = 10
OPEN_FREQ_MULT = 1.20
THREEBET_FREQ_MULT = 1.25
PREMIUM_TIER1_MAX_REQUIRED = 0.38
PREMIUM_TIER2_MAX_REQUIRED = 0.33
DEBUG_PF = False


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


def is_big_card(rank):
    return rank >= 11


def has_rank(cards, rank):
    return any(card_rank(c) == rank for c in cards)


def suited_pair_count(cards):
    suits = [card_suit(c) for c in cards]
    suit_counts = Counter(suits)
    return max(suit_counts.values())


def classify_strong_jam_hand(hand3):
    ranks = sorted([card_rank(c) for c in hand3], reverse=True)
    suits = [card_suit(c) for c in hand3]
    suit_counts = Counter(suits)

    pair_rank = None
    for i in range(3):
        for j in range(i + 1, 3):
            if ranks[i] == ranks[j]:
                pair_rank = ranks[i]
    if pair_rank is not None and pair_rank >= 10:
        return True
    if pair_rank is not None and 7 <= pair_rank <= 9:
        if any(r >= 11 for r in ranks):
            return True

    suited = suit_counts.most_common(1)[0][1] >= 2
    if suited and has_rank(hand3, 14) and has_rank(hand3, 13):
        return True
    if suited and has_rank(hand3, 14) and has_rank(hand3, 12):
        return True
    big_cards = [r for r in ranks if r >= 12]
    if suited and len(big_cards) >= 2:
        return True
    return False


def classify_premium_tier(hand3):
    ranks = sorted([card_rank(c) for c in hand3], reverse=True)
    suits = [card_suit(c) for c in hand3]
    suit_counts = Counter(suits)

    pair_rank = None
    for i in range(3):
        for j in range(i + 1, 3):
            if ranks[i] == ranks[j]:
                pair_rank = ranks[i]

    suited = suit_counts.most_common(1)[0][1] >= 2
    has_ace = 14 in ranks
    has_king = 13 in ranks
    has_queen = 12 in ranks
    has_jack = 11 in ranks

    if pair_rank is not None and pair_rank >= 11:
        return "tier1"
    if suited and has_ace and has_king:
        return "tier1"
    if suited and has_ace and has_queen:
        return "tier1"
    if pair_rank == 10 and (has_ace or has_king or has_queen):
        return "tier2"
    if suited and sum(1 for r in ranks if r >= 11) >= 2 and max(ranks) >= 14:
        if min(ranks) >= 8:
            return "tier2"
    if pair_rank == 9 and (has_ace or has_king):
        return "tier2"
    return None


def card_rank(card):
    return to_card(card).rank


def card_suit(card):
    return to_card(card).suit


def rank_gap(card_a, card_b):
    return abs(card_rank(card_a) - card_rank(card_b))


def is_pair(cards):
    return card_rank(cards[0]) == card_rank(cards[1])


def is_suited(cards):
    return card_suit(cards[0]) == card_suit(cards[1])


def board_suit_count(board_cards, suit):
    return sum(1 for c in board_cards if card_suit(c) == suit)


def board_rank_count(board_cards, rank):
    return sum(1 for c in board_cards if card_rank(c) == rank)


def straight_window_count(ranks):
    unique = sorted(set(ranks))
    windows = 0
    for start in range(2, 11):
        window = set(range(start, start + 5))
        if window.intersection(unique):
            windows += 1
    return windows


def keep_strength(kept, board_cards):
    strength = 0
    ranks = [card_rank(c) for c in kept]
    if is_pair(kept):
        strength += 6
    if is_suited(kept):
        strength += 2
    gap = rank_gap(kept[0], kept[1])
    if gap == 0:
        strength += 2
    elif gap == 1:
        strength += 2
    elif gap == 2:
        strength += 1
    for r in ranks:
        if r >= 11:
            strength += 1
    for r in ranks:
        if board_rank_count(board_cards, r) > 0:
            strength += 2
    return strength


def keep_potential(kept, board_cards):
    potential = 0
    ranks = [card_rank(c) for c in kept]
    suited = is_suited(kept)
    if suited:
        suit = card_suit(kept[0])
        suit_count = board_suit_count(board_cards, suit)
        if suit_count >= 2:
            potential += 4
        elif suit_count == 1:
            potential += 2
        else:
            potential += 1
    gap = rank_gap(kept[0], kept[1])
    if gap <= 4:
        potential += 1
    if gap == 1:
        potential += 2
    board_ranks = [card_rank(c) for c in board_cards]
    combined = board_ranks + ranks
    if straight_window_count(combined) >= 3:
        potential += 2
    if any(board_rank_count(board_cards, r) > 0 for r in ranks):
        potential += 2
    if len(board_ranks) != len(set(board_ranks)):
        potential += 1
    return potential


def board_help(discarded, board_cards):
    penalty = 0
    disc_rank = card_rank(discarded)
    disc_suit = card_suit(discarded)
    suit_count = board_suit_count(board_cards, disc_suit)
    if suit_count >= 2:
        penalty += 6
    elif suit_count == 1:
        penalty += 3
    if board_rank_count(board_cards, disc_rank) > 0:
        penalty += 4
    board_ranks = [card_rank(c) for c in board_cards]
    if any(abs(disc_rank - r) == 1 for r in board_ranks):
        penalty += 2
    if straight_window_count(board_ranks + [disc_rank]) > straight_window_count(board_ranks):
        penalty += 1
    if disc_rank >= 11 and all(r <= 9 for r in board_ranks):
        penalty += 2
    return penalty


def reverse_implied_risk(kept, board_cards):
    risk = 0
    ranks = [card_rank(c) for c in kept]
    suited = is_suited(kept)
    if suited:
        suit = card_suit(kept[0])
        if board_suit_count(board_cards, suit) >= 2 and max(ranks) <= 9:
            risk += 2
    gap = rank_gap(kept[0], kept[1])
    board_ranks = [card_rank(c) for c in board_cards]
    if gap <= 2 and max(ranks) <= 8 and any(r >= 11 for r in board_ranks):
        risk += 2
    if gap >= 4 and not suited:
        risk += 1
    return risk


def choose_discard_index(hand3, board_cards, position):
    board_cards = list(board_cards)
    if len(hand3) <= 2:
        return 0

    A, B, C, D = 4, 3, 3, 2
    scored = []
    overrides = {}

    ranks = [card_rank(c) for c in hand3]
    suits = [card_suit(c) for c in hand3]
    board_ranks = [card_rank(c) for c in board_cards]

    # Override #1 + #5: never discard a card that matches the board.
    matching_board = {i for i, r in enumerate(ranks) if r in board_ranks}
    if matching_board:
        overrides["keep_board_match"] = matching_board

    # Override #2: if two cards share a suit, prefer discarding off-suit.
    suit_counts = Counter(suits)
    suited_suit = next((s for s, c in suit_counts.items() if c == 2), None)
    if suited_suit:
        off_suit_indices = [i for i, s in enumerate(suits) if s != suited_suit]
        if off_suit_indices:
            overrides["prefer_off_suit"] = set(off_suit_indices)

    # Override #3: if pair, keep pair unless board help would be massive.
    pair_rank = None
    for i in range(3):
        for j in range(i + 1, 3):
            if ranks[i] == ranks[j]:
                pair_rank = ranks[i]
    if pair_rank is not None:
        overrides["keep_pair"] = {i for i, r in enumerate(ranks) if r == pair_rank}

    # Override #4: avoid creating 3-flush on board.
    flush_sensitive = {i for i, s in enumerate(suits) if board_suit_count(board_cards, s) >= 2}
    if flush_sensitive:
        overrides["avoid_third_flush"] = flush_sensitive

    for idx in range(3):
        kept = [hand3[i] for i in range(3) if i != idx]
        discarded = hand3[idx]
        score = 0
        score += A * keep_strength(kept, board_cards)
        score += B * keep_potential(kept, board_cards)
        score -= C * board_help(discarded, board_cards)
        score -= D * reverse_implied_risk(kept, board_cards)
        if position == "sb":
            score -= board_help(discarded, board_cards) * 0.5
        scored.append(score)

    # Apply overrides by making disallowed discards very unattractive.
    for idx in range(3):
        if "keep_board_match" in overrides and idx in overrides["keep_board_match"]:
            scored[idx] -= 50
        if "avoid_third_flush" in overrides and idx in overrides["avoid_third_flush"]:
            scored[idx] -= 20
        if "keep_pair" in overrides and idx in overrides["keep_pair"]:
            scored[idx] -= 15
        if "prefer_off_suit" in overrides and idx not in overrides["prefer_off_suit"]:
            scored[idx] -= 5

    # Tiebreakers: discard least synergistic, then lowest rank.
    best = max(scored)
    candidates = [i for i, s in enumerate(scored) if s == best]
    if len(candidates) > 1:
        def synergy(idx):
            kept = [hand3[i] for i in range(3) if i != idx]
            suited = 1 if is_suited(kept) else 0
            gap = rank_gap(kept[0], kept[1])
            return suited * 2 - gap

        candidates.sort(key=lambda i: (synergy(i), -min(ranks[i], 14)))
    return candidates[0]


class Player(Bot):
    def __init__(self):
        self.rng = random.Random(7)
        self.regrets = {}
        self.strategy_sum = {}
        self.equity_cache = {}
        self.mccfr_iterations = 4
        self.base_samples = 40
        self.precomputed_strategy = self.load_strategy(STRATEGY_PATH)
        self.pf_jam_opportunities = 0
        self.pf_jams = 0
        self.pf_jam_showdowns = 0
        self.pf_jam_strong_showdowns = 0
        self.faced_preflop_jam = False
        self.counted_pf_opportunity = False

    def handle_new_round(self, game_state, round_state, active):
        self.faced_preflop_jam = False
        self.counted_pf_opportunity = False

    def handle_round_over(self, game_state, terminal_state, active):
        try:
            previous_state = terminal_state.previous_state
            if self.faced_preflop_jam and previous_state is not None:
                opp_cards = previous_state.hands[1 - active]
                if opp_cards:
                    self.pf_jam_showdowns += 1
                    if classify_strong_jam_hand(opp_cards):
                        self.pf_jam_strong_showdowns += 1
        except Exception:
            self.faced_preflop_jam = False

    def get_action(self, game_state, round_state, active):
        try:
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

            if street == 0:
                if not self.counted_pf_opportunity:
                    self.pf_jam_opportunities += 1
                    self.counted_pf_opportunity = True
                continue_cost = round_state.pips[1 - active] - round_state.pips[active]
                facing_jam = continue_cost > 0 and round_state.stacks[1 - active] == 0
                if facing_jam:
                    self.faced_preflop_jam = True
                    self.pf_jams += 1
                    if self.villain_type() in ("tight_pf_jammer", "likely_pf_jammer"):
                        decision = self.decide_vs_preflop_jam(my_cards, round_state, active)
                        if DEBUG_PF:
                            print(
                                f"[PF JAM] hand={my_cards} decision={decision} "
                                f"required={self.required_equity(round_state, active):.2f} "
                                f"villain={self.villain_type()}",
                            )
                        return decision

            equity = self.estimate_equity_cached(my_cards, board_cards, street)
            info_set = self.build_infoset(round_state, active, equity)

            if self.precomputed_strategy and info_set in self.precomputed_strategy:
                strategy = self.precomputed_strategy[info_set]
            else:
                for _ in range(self.mccfr_iterations):
                    utilities = self.action_utilities(round_state, active, equity, action_labels)
                    self.update_regrets(info_set, action_labels, utilities)
                strategy = self.get_strategy(info_set, action_labels)
            if street == 0 and self.villain_type() in ("tight_pf_jammer", "likely_pf_jammer"):
                strategy = self.adjust_preflop_strategy(strategy)
            choice = self.sample_action(strategy)
            return self.to_action(choice, round_state, legal)
        except Exception:
            legal = round_state.legal_actions()
            if CheckAction in legal:
                return CheckAction()
            if CallAction in legal:
                return CallAction()
            return FoldAction()

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

    def average_strategy(self):
        average = {}
        for info_set, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            if total <= 0:
                continue
            average[info_set] = {action: value / total for action, value in action_sums.items()}
        return average

    def villain_type(self):
        pf_jam_rate = self.pf_jams / max(1, self.pf_jam_opportunities)
        pf_jam_strong_rate = self.pf_jam_strong_showdowns / max(1, self.pf_jam_showdowns)
        if (
            self.pf_jam_opportunities >= PF_JAM_OPPORTUNITY_MIN
            and pf_jam_rate >= PF_JAM_RATE_THRESHOLD
            and self.pf_jam_showdowns >= PF_JAM_SHOWDOWN_MIN
            and pf_jam_strong_rate >= PF_JAM_STRONG_RATE_THRESHOLD
        ):
            return "tight_pf_jammer"
        if (
            self.pf_jam_opportunities >= PF_LIKELY_JAM_OPPORTUNITY_MIN
            and pf_jam_rate >= PF_LIKELY_JAM_RATE
        ):
            return "likely_pf_jammer"
        return "unknown"

    def required_equity(self, round_state, active):
        to_call = round_state.pips[1 - active] - round_state.pips[active]
        pot = self.compute_pot(round_state)
        if to_call <= 0:
            return 0.0
        return to_call / (pot + to_call)

    def decide_vs_preflop_jam(self, hand3, round_state, active):
        villain_type = self.villain_type()
        if villain_type not in ("tight_pf_jammer", "likely_pf_jammer"):
            return self.to_action("call", round_state, round_state.legal_actions())
        tier = classify_premium_tier(hand3)
        if tier is None:
            return FoldAction()
        required = self.required_equity(round_state, active)
        threshold = PREMIUM_TIER1_MAX_REQUIRED if tier == "tier1" else PREMIUM_TIER2_MAX_REQUIRED
        if required <= threshold:
            return self.to_action("call", round_state, round_state.legal_actions())
        return FoldAction()

    def adjust_preflop_strategy(self, strategy):
        adjusted = dict(strategy)
        if "raise" in adjusted:
            adjusted["raise"] *= OPEN_FREQ_MULT
        if "call" in adjusted:
            adjusted["call"] *= THREEBET_FREQ_MULT
        total = sum(adjusted.values())
        if total > 0:
            for action in adjusted:
                adjusted[action] /= total
        return adjusted

    def save_strategy(self, path):
        data = self.average_strategy()
        with open(path, "wb") as handle:
            pickle.dump(data, handle)
        return data

    def load_strategy(self, path):
        if os.path.exists(path):
            with open(path, "rb") as handle:
                return pickle.load(handle)
        return None

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
        return choose_discard_index(my_cards, board_cards, self.position_from_street(board_cards))

    def position_from_street(self, board_cards):
        return "sb" if len(board_cards) >= 3 else "bb"

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
