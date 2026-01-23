"""
Postflop anti-jam defender utilities.
"""
from collections import Counter
import random


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


def default_config(evaluate_best):
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
