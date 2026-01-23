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
from skeleton.states import STARTING_STACK, BIG_BLIND, SMALL_BLIND, NUM_ROUNDS
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
DEBUG_JAM = False
JAM_DEFENSE_ENABLED = True
JAM_DEFENSE_MODE = "hybrid"
MC_SAMPLES_FLOP = 600
MC_SAMPLES_TURN = 500
MC_SAMPLES_RIVER = 400
SAFETY_MARGIN_FLOP = 0.03
SAFETY_MARGIN_TURN = 0.05
SAFETY_MARGIN_RIVER = 0.08
MIN_SHOWDOWNS_FOR_RANGE_UPDATE = 5
RANGE_PRIOR_ALLOW_TOP_PAIR = True
RANGE_PRIOR_ALLOW_DRAWS = True
RANGE_PRIOR_ALLOW_TURN_DRAWS = True
RANGE_PRIOR_ALLOW_RIVER_STRONG_PAIR = False

HAND_CLASSES = [
    "high_card",
    "one_pair",
    "two_pair",
    "trips",
    "straight",
    "flush",
    "full_house",
    "quads",
    "straight_flush",
]


def jam_defense_config(evaluate_best):
    return {
        "enabled": True,
        "mode": "hybrid",
        "mc_samples_flop": 600,
        "mc_samples_turn": 500,
        "mc_samples_river": 400,
        "safety_margin_flop": 0.03,
        "safety_margin_turn": 0.05,
        "safety_margin_river": 0.08,
        "min_showdowns": 5,
        "prior_allow_top_pair": True,
        "prior_allow_draws": True,
        "prior_allow_turn_draws": True,
        "prior_allow_river_strong_pair": False,
        "evaluate_best": evaluate_best,
        "debug": False,
    }


def street_key_from_value(street_value):
    if street_value <= 3:
        return "flop"
    if street_value == 4:
        return "turn"
    return "river"


def classify_hand(cards, evaluate_best):
    score = evaluate_best(cards)
    category = score[0]
    return HAND_CLASSES[category]


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


def count_suits(cards, card_suit):
    return Counter(card_suit(c) for c in cards)


def count_ranks(cards, card_rank):
    return Counter(card_rank(c) for c in cards)


def is_flush_draw(hole_cards, board_cards, card_suit):
    suits = count_suits(hole_cards + board_cards, card_suit)
    for suit, count in suits.items():
        hole_count = sum(1 for c in hole_cards if card_suit(c) == suit)
        if hole_count >= 1 and count >= 4 and count < 5:
            return True, suit
    return False, None


def is_nut_flush_draw(hole_cards, board_cards, card_suit, card_rank):
    has_draw, suit = is_flush_draw(hole_cards, board_cards, card_suit)
    if not has_draw:
        return False
    suited_hole = [card_rank(c) for c in hole_cards if card_suit(c) == suit]
    if not suited_hole:
        return False
    return max(suited_hole) >= 13


def straight_draw_outs(ranks):
    unique = sorted(set(ranks))
    outs = 0
    for start in range(2, 11):
        window = list(range(start, start + 5))
        missing = [r for r in window if r not in unique]
        if len(missing) == 1:
            outs += 1
    return outs


def is_straight_draw(hole_cards, board_cards, card_rank):
    ranks = [card_rank(c) for c in hole_cards + board_cards]
    return straight_draw_outs(ranks) > 0


def is_top_pair_good(hole_cards, board_cards, card_rank):
    if not board_cards:
        return False
    board_ranks = [card_rank(c) for c in board_cards]
    top_rank = max(board_ranks)
    hole_ranks = [card_rank(c) for c in hole_cards]
    if top_rank not in hole_ranks:
        return False
    kicker = max(hole_ranks)
    return kicker >= 12


def board_is_scary(board_cards, card_rank, card_suit):
    board_ranks = [card_rank(c) for c in board_cards]
    rank_counts = Counter(board_ranks)
    if any(v >= 2 for v in rank_counts.values()):
        return True
    suit_counts = Counter(card_suit(c) for c in board_cards)
    if any(v >= 4 for v in suit_counts.values()):
        return True
    return False


def villain_in_range(
    villain_hole,
    board_cards,
    street,
    card_rank,
    card_suit,
    evaluate_best,
    config,
):
    hand_class = classify_hand(villain_hole + board_cards, evaluate_best)
    class_rank = HAND_CLASSES.index(hand_class)
    if street == "river":
        if class_rank >= HAND_CLASSES.index("two_pair"):
            return True
        if config["prior_allow_river_strong_pair"] and hand_class == "one_pair":
            return is_top_pair_good(villain_hole, board_cards, card_rank)
        return False

    if class_rank >= HAND_CLASSES.index("two_pair"):
        return True

    if config["prior_allow_top_pair"] and hand_class == "one_pair":
        if is_top_pair_good(villain_hole, board_cards, card_rank):
            return True

    if config["prior_allow_draws"]:
        if is_nut_flush_draw(villain_hole, board_cards, card_suit, card_rank):
            return True
        if is_straight_draw(villain_hole, board_cards, card_rank):
            return True

    if street == "turn" and config["prior_allow_turn_draws"]:
        return is_nut_flush_draw(villain_hole, board_cards, card_suit, card_rank)

    return False


def equity_vs_range(
    hole_cards,
    board_cards,
    street,
    pot,
    to_call,
    card_rank,
    card_suit,
    evaluate_best,
    rng,
    config,
):
    deck = []
    used = {str(c) for c in hole_cards + board_cards}
    for rank in "23456789TJQKA":
        for suit in "shdc":
            card = f"{rank}{suit}"
            if card not in used:
                deck.append(card)

    remaining = max(0, 6 - len(board_cards))
    samples = {
        "flop": config["mc_samples_flop"],
        "turn": config["mc_samples_turn"],
        "river": config["mc_samples_river"],
    }[street]

    wins = 0
    ties = 0
    accepted = 0
    for _ in range(samples):
        if len(deck) < 2:
            break
        villain = rng.sample(deck, 2)
        if not villain_in_range(villain, board_cards, street, card_rank, card_suit, evaluate_best, config):
            continue
        accepted += 1
        remaining_deck = [c for c in deck if c not in villain]
        extra = rng.sample(remaining_deck, remaining) if remaining else []
        full_board = list(board_cards) + list(extra)
        our_score = evaluate_best(hole_cards + full_board)
        vill_score = evaluate_best(villain + full_board)
        if our_score > vill_score:
            wins += 1
        elif our_score == vill_score:
            ties += 1

    if accepted == 0:
        return 0.0, 0.0
    equity = (wins + 0.5 * ties) / accepted
    return equity, accepted / samples


def rules_based_call(
    hole_cards,
    board_cards,
    street,
    required,
    card_rank,
    card_suit,
    evaluate_best,
    config,
):
    hand_class = classify_hand(hole_cards + board_cards, evaluate_best)
    class_rank = HAND_CLASSES.index(hand_class)
    if class_rank >= HAND_CLASSES.index("full_house"):
        return True, 1.0
    if street == "river":
        if class_rank >= HAND_CLASSES.index("two_pair"):
            return required <= 0.55, 0.55
        return False, 0.0
    if class_rank >= HAND_CLASSES.index("two_pair"):
        return required <= 0.5, 0.5
    if hand_class == "one_pair":
        if is_top_pair_good(hole_cards, board_cards, card_rank) and not board_is_scary(board_cards, card_rank, card_suit):
            return required <= 0.35, 0.35
        return False, 0.0
    if hand_class == "high_card":
        if is_nut_flush_draw(hole_cards, board_cards, card_suit, card_rank):
            return required <= 0.33, 0.33
        return False, 0.0
    return False, 0.0


def decide_vs_postflop_jam(
    hole_cards,
    board_cards,
    street,
    pot,
    to_call,
    stack_eff,
    stats,
    config,
    rng,
    card_rank,
    card_suit,
):
    required = to_call / (pot + to_call) if to_call > 0 else 0.0
    safety = {
        "flop": config["safety_margin_flop"],
        "turn": config["safety_margin_turn"],
        "river": config["safety_margin_river"],
    }[street]
    if stats:
        showdowns = stats.get("jam_showdowns", {}).get(street, 0)
        counts = stats.get("jam_showdown_handclass_counts", {}).get(street, {})
        if showdowns >= config["min_showdowns"] and counts:
            value_count = sum(
                counts.get(cls, 0)
                for cls in ("two_pair", "trips", "straight", "flush", "full_house", "quads", "straight_flush")
            )
            value_rate = value_count / max(1, showdowns)
            if value_rate >= 0.7:
                safety += 0.05

    if config["mode"] in ("rules",):
        call, equity = rules_based_call(
            hole_cards,
            board_cards,
            street,
            required,
            card_rank,
            card_suit,
            config["evaluate_best"],
            config,
        )
        return call, equity, 0.0

    if config["mode"] in ("mc", "hybrid"):
        equity, acceptance = equity_vs_range(
            hole_cards,
            board_cards,
            street,
            pot,
            to_call,
            card_rank,
            card_suit,
            config["evaluate_best"],
            rng,
            config,
        )
        if equity == 0.0:
            call, equity = rules_based_call(
                hole_cards,
                board_cards,
                street,
                required,
                card_rank,
                card_suit,
                config["evaluate_best"],
                config,
            )
            return call, equity, acceptance
        return equity >= required + safety, equity, acceptance

    call, equity = rules_based_call(
        hole_cards,
        board_cards,
        street,
        required,
        card_rank,
        card_suit,
        config["evaluate_best"],
        config,
    )
    return call, equity, 0.0

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
        self.lockdown_mode = False
        self.counted_postflop_opps = set()
        self.jam_opps = {"flop": 0, "turn": 0, "river": 0}
        self.jams = {"flop": 0, "turn": 0, "river": 0}
        self.jam_showdowns = {"flop": 0, "turn": 0, "river": 0}
        self.jam_showdown_handclass_counts = {
            "flop": Counter(),
            "turn": Counter(),
            "river": Counter(),
        }
        self.last_postflop_jam_street = None
        self.jam_defense_config = jam_defense_config(evaluate_best)
        self.jam_defense_config.update(
            {
                "enabled": JAM_DEFENSE_ENABLED,
                "mode": JAM_DEFENSE_MODE,
                "mc_samples_flop": MC_SAMPLES_FLOP,
                "mc_samples_turn": MC_SAMPLES_TURN,
                "mc_samples_river": MC_SAMPLES_RIVER,
                "safety_margin_flop": SAFETY_MARGIN_FLOP,
                "safety_margin_turn": SAFETY_MARGIN_TURN,
                "safety_margin_river": SAFETY_MARGIN_RIVER,
                "min_showdowns": MIN_SHOWDOWNS_FOR_RANGE_UPDATE,
                "prior_allow_top_pair": RANGE_PRIOR_ALLOW_TOP_PAIR,
                "prior_allow_draws": RANGE_PRIOR_ALLOW_DRAWS,
                "prior_allow_turn_draws": RANGE_PRIOR_ALLOW_TURN_DRAWS,
                "prior_allow_river_strong_pair": RANGE_PRIOR_ALLOW_RIVER_STRONG_PAIR,
                "debug": DEBUG_JAM,
            }
        )

    def handle_new_round(self, game_state, round_state, active):
        self.faced_preflop_jam = False
        self.counted_pf_opportunity = False
        self.last_postflop_jam_street = None
        self.counted_postflop_opps = set()
        rounds_left = NUM_ROUNDS - game_state.round_num + 1
        fold_cost_per_round = (SMALL_BLIND + BIG_BLIND) / 2
        max_safe_loss = rounds_left * fold_cost_per_round
        self.lockdown_mode = game_state.bankroll > max_safe_loss

    def handle_round_over(self, game_state, terminal_state, active):
        try:
            previous_state = terminal_state.previous_state
            if self.faced_preflop_jam and previous_state is not None:
                opp_cards = previous_state.hands[1 - active]
                if opp_cards:
                    self.pf_jam_showdowns += 1
                    if classify_strong_jam_hand(opp_cards):
                        self.pf_jam_strong_showdowns += 1
            if self.last_postflop_jam_street and previous_state is not None:
                opp_cards = previous_state.hands[1 - active]
                if opp_cards:
                    street_key = self.last_postflop_jam_street
                    self.jam_showdowns[street_key] += 1
                    hand_class = classify_hand(opp_cards + previous_state.board, evaluate_best)
                    self.jam_showdown_handclass_counts[street_key][hand_class] += 1
        except Exception:
            self.faced_preflop_jam = False

    def get_action(self, game_state, round_state, active):
        try:
            legal = round_state.legal_actions()
            street = round_state.street
            my_cards = round_state.hands[active]
            board_cards = round_state.board
            continue_cost = round_state.pips[1 - active] - round_state.pips[active]

            if DiscardAction in legal:
                discard_idx = self.choose_discard(my_cards, board_cards)
                return DiscardAction(discard_idx)

            if street in (2, 3):
                return CheckAction()

            if self.lockdown_mode:
                if CheckAction in legal:
                    return CheckAction()
                if CallAction in legal:
                    return CallAction()
                return FoldAction()

            action_labels = self.available_action_labels(legal)
            if not action_labels:
                return CheckAction()

            if street == 0:
                if not self.counted_pf_opportunity:
                    self.pf_jam_opportunities += 1
                    self.counted_pf_opportunity = True
                facing_jam = continue_cost > 0 and round_state.stacks[1 - active] == 0
                facing_all_in = continue_cost > 0 and continue_cost >= round_state.stacks[active]
                if facing_jam or facing_all_in:
                    self.faced_preflop_jam = True
                    self.pf_jams += 1
                    decision = self.decide_vs_preflop_allin(my_cards, round_state, active)
                    if DEBUG_PF:
                        print(
                            f"[PF ALL-IN] hand={my_cards} decision={decision} "
                            f"required={self.required_equity(round_state, active):.2f} "
                            f"villain={self.villain_type()}",
                        )
                    return decision

            if street >= 4:
                street_key = street_key_from_value(street)
                if street_key not in self.counted_postflop_opps:
                    self.jam_opps[street_key] += 1
                    self.counted_postflop_opps.add(street_key)
                if continue_cost <= 0:
                    pass
                else:
                    facing_all_in = continue_cost >= round_state.stacks[active]
                    facing_jam = round_state.stacks[1 - active] == 0
                    if facing_all_in or facing_jam:
                        self.jams[street_key] += 1
                        self.last_postflop_jam_street = street_key
                        if self.jam_defense_config["enabled"]:
                            pot = self.compute_pot(round_state)
                            decision, equity, acceptance = decide_vs_postflop_jam(
                                my_cards,
                                board_cards,
                                street_key,
                                pot,
                                continue_cost,
                                min(round_state.stacks),
                                {
                                    "jam_opps": self.jam_opps,
                                    "jams": self.jams,
                                    "jam_showdowns": self.jam_showdowns,
                                    "jam_showdown_handclass_counts": self.jam_showdown_handclass_counts,
                                },
                                self.jam_defense_config,
                                self.rng,
                                card_rank,
                                card_suit,
                            )
                            if DEBUG_JAM or self.jam_defense_config.get("debug"):
                                required = self.required_equity(round_state, active)
                                print(
                                    f"[POSTFLOP JAM] street={street_key} pot={pot} to_call={continue_cost} "
                                    f"req={required:.2f} eq={equity:.2f} acc={acceptance:.2f} "
                                    f"decision={'call' if decision else 'fold'}",
                                )
                            return CallAction() if decision else FoldAction()

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
            return FoldAction()
        tier = classify_premium_tier(hand3)
        if tier is None:
            return FoldAction()
        required = self.required_equity(round_state, active)
        threshold = PREMIUM_TIER1_MAX_REQUIRED if tier == "tier1" else PREMIUM_TIER2_MAX_REQUIRED
        if required <= threshold:
            return self.to_action("call", round_state, round_state.legal_actions())
        return FoldAction()

    def decide_vs_preflop_allin(self, hand3, round_state, active):
        villain_type = self.villain_type()
        if villain_type in ("tight_pf_jammer", "likely_pf_jammer"):
            return self.decide_vs_preflop_jam(hand3, round_state, active)
        tier = classify_premium_tier(hand3)
        if tier != "tier1":
            return FoldAction()
        required = self.required_equity(round_state, active)
        if required <= PREMIUM_TIER1_MAX_REQUIRED:
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
