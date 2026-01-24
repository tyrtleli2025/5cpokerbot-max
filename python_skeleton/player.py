'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
from collections import Counter

RANK_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANK_ORDER, start=2)}

def parse_card(card):
    """
    Adapts to common encodings: 'As', 'Td', etc.
    Returns (rank_int, suit_char).
    If your engine uses a different format, adjust here.
    """
    r = card[0]
    s = card[1]
    return RANK_TO_INT[r], s

def hole_card_score(cards):
    """
    Very lightweight preflop score.
    Higher = stronger.
    """
    (r1, s1) = parse_card(cards[0])
    (r2, s2) = parse_card(cards[1])
    hi, lo = max(r1, r2), min(r1, r2)
    suited = (s1 == s2)
    gap = hi - lo

    score = 0

    # Pairs are strong
    if r1 == r2:
        score += 40 + (hi * 2)  # AA highest
        return score

    # High cards matter
    score += hi
    score += lo * 0.5

    # Suited bonus
    if suited:
        score += 3

    # Connectivity bonus
    if gap == 1:
        score += 4
    elif gap == 2:
        score += 2
    elif gap == 3:
        score += 1

    # Broadways bonus
    broadways = sum(1 for r in (r1, r2) if r >= 10)
    score += broadways * 2

    # Ace + something
    if hi == 14:
        score += 2

    return score

def count_flush_draw(hole, board):
    """
    Returns True if we have a 4-card flush (flush draw) on flop/turn.
    """
    suits = [parse_card(c)[1] for c in hole + board]
    suit_counts = Counter(suits)
    return max(suit_counts.values()) == 4

def count_made_flush(hole, board):
    suits = [parse_card(c)[1] for c in hole + board]
    suit_counts = Counter(suits)
    return max(suit_counts.values()) >= 5

def count_straight_draw(hole, board):
    """
    Detects a simple open-ended straight draw proxy.
    Not perfect, but fast and better than nothing.
    """
    ranks = sorted({parse_card(c)[0] for c in hole + board})
    # handle wheel possibility by treating Ace as 1 too
    if 14 in ranks:
        ranks = sorted(set(ranks + [1]))

    best_run = 1
    run = 1
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i-1] + 1:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 1

    # If we have 4 in a row but not 5, that's a strong draw.
    return best_run == 4

def made_hand_tier(hole, board):
    """
    Returns an integer tier describing made hand strength:
    0 = high card
    1 = one pair
    2 = two pair
    3 = trips
    4 = straight
    5 = flush
    6 = full house+
    This is approximate (especially straight/full house), but works decently.
    """
    all_cards = hole + board
    ranks = [parse_card(c)[0] for c in all_cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    # Flush
    if count_made_flush(hole, board):
        return 5

    # Straight (approx)
    ranks_u = sorted(set(ranks))
    if 14 in ranks_u:
        ranks_u = sorted(set(ranks_u + [1]))
    run = 1
    best_run = 1
    for i in range(1, len(ranks_u)):
        if ranks_u[i] == ranks_u[i-1] + 1:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 1
    is_straight = best_run >= 5

    if counts[0] == 4:
        return 6
    if counts[0] == 3 and counts[1] >= 2:
        return 6
    if is_straight:
        return 4
    if counts[0] == 3:
        return 3
    if counts[0] == 2 and counts[1] == 2:
        return 2
    if counts[0] == 2:
        return 1
    return 0

def hand_strength_proxy(hole, board):
    """
    Produces a 0..1-ish proxy for equity/strength.
    - Strong made hands => high
    - Good draws => medium
    - Weak => low
    """
    tier = made_hand_tier(hole, board)

    if tier >= 6:
        return 0.95
    if tier == 5:  # flush
        return 0.90
    if tier == 4:  # straight
        return 0.85
    if tier == 3:  # trips
        return 0.80
    if tier == 2:  # two pair
        return 0.72
    if tier == 1:  # one pair
        # Pair strength: is it likely top-ish?
        hole_ranks = sorted([parse_card(c)[0] for c in hole], reverse=True)
        board_ranks = sorted([parse_card(c)[0] for c in board], reverse=True) if board else []
        top_board = board_ranks[0] if board_ranks else 0
        # if our highest card >= top board, pair might be decent
        bonus = 0.05 if hole_ranks[0] >= top_board else 0.0
        return 0.55 + bonus

    # High card: evaluate overcards + draws
    strength = 0.35
    if board:
        hole_ranks = sorted([parse_card(c)[0] for c in hole], reverse=True)
        board_ranks = sorted([parse_card(c)[0] for c in board], reverse=True)
        # overcards to board top
        if hole_ranks[0] > board_ranks[0]:
            strength += 0.05
        if hole_ranks[1] > board_ranks[0]:
            strength += 0.03

    if len(board) >= 3:
        if count_flush_draw(hole, board):
            strength += 0.18
        if count_straight_draw(hole, board):
            strength += 0.15

    return min(0.75, strength)

def clamp_raise(target_pip, min_raise, max_raise):
    return max(min_raise, min(max_raise, target_pip))

class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        pass

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # the total number of seconds your bot has left to play this game
        game_clock = game_state.game_clock
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0,2,3,4,5,6 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        # opponent's cards or [] if not revealed
        opp_cards = previous_state.hands[1-active]
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.board

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]

        continue_cost = opp_pip - my_pip
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        pot = my_contribution + opp_contribution

        # Discard: keep the better of the two hole cards (simple heuristic).
        if DiscardAction in legal_actions:
            (r1, s1) = parse_card(my_cards[0])
            (r2, s2) = parse_card(my_cards[1])
            # Discard the lower rank; if equal, discard non-suited preference irrelevant.
            discard_idx = 0 if r1 < r2 else 1
            return DiscardAction(discard_idx)

        # Helper: choose between check/call depending on cost
        def passive_action():
            if continue_cost == 0 and CheckAction in legal_actions:
                return CheckAction()
            if continue_cost > 0 and CallAction in legal_actions:
                return CallAction()
            # fallback
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction()

        # If we cannot fold (rare), just call/check
        can_fold = FoldAction in legal_actions

        # Pot odds for calling
        # If we call continue_cost, total pot becomes pot + continue_cost (ignoring future betting)
        call_pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0

        # Preflop strategy
        if street == 0:
            score = hole_card_score(my_cards)

            strong = score >= 55
            medium = 45 <= score < 55
            weak = score < 45

            # When facing a bet
            if continue_cost > 0:
                # If pot odds are good, call wider
                if strong:
                    # raise often for value
                    if RaiseAction in legal_actions:
                        min_raise, max_raise = round_state.raise_bounds()
                        # size: 3x the continue_cost added to current opp_pip baseline
                        target = opp_pip + 3 * continue_cost
                        target = clamp_raise(target, min_raise, max_raise)
                        # sometimes just call to avoid predictability
                        if random.random() < 0.75:
                            return RaiseAction(target)
                    return CallAction()

                if medium:
                    # call if cost isn't too large relative to pot
                    if call_pot_odds <= 0.33:
                        return CallAction()
                    return FoldAction() if can_fold else CallAction()

                # weak hands fold unless extremely cheap
                if call_pot_odds <= 0.20:
                    return CallAction()
                return FoldAction() if can_fold else CallAction()

            # If we can check (unopened)
            if RaiseAction in legal_actions and continue_cost == 0:
                min_raise, max_raise = round_state.raise_bounds()
                # open-raise with strong hands, occasionally with medium
                if strong or (medium and random.random() < 0.25):
                    # size: ~2.5bb equivalent proxy: choose a moderate raise inside bounds
                    target = clamp_raise(min_raise + int(0.6 * (max_raise - min_raise)), min_raise, max_raise)
                    # if bounds are tight, just min-raise
                    if random.random() < 0.85:
                        return RaiseAction(target)
                return CheckAction() if CheckAction in legal_actions else CallAction()

            return passive_action()

        # Postflop (flop/turn/river)
        strength = hand_strength_proxy(my_cards, board_cards)

        # Define thresholds by street (tighter on river)
        if street == 3:      # flop
            value_thresh = 0.70
            call_thresh = 0.45
        elif street == 4:    # turn
            value_thresh = 0.75
            call_thresh = 0.48
        else:                # river (5) or later if your engine uses 6
            value_thresh = 0.80
            call_thresh = 0.52

        # Facing a bet: call if strength beats pot odds with some margin
        if continue_cost > 0:
            # required equity roughly = pot odds
            required = call_pot_odds + 0.03
            if strength >= max(call_thresh, required):
                # Raise for value with very strong hands
                if RaiseAction in legal_actions and strength >= value_thresh:
                    min_raise, max_raise = round_state.raise_bounds()
                    # value raise size: ~70% pot, expressed as target pip
                    raise_amount = int(0.7 * (pot + continue_cost))
                    target = opp_pip + raise_amount
                    target = clamp_raise(target, min_raise, max_raise)
                    if random.random() < 0.65:
                        return RaiseAction(target)
                return CallAction()

            # Occasionally bluff-raise strong draws (semi-bluff) if we have them
            has_draw = (len(board_cards) >= 3) and (count_flush_draw(my_cards, board_cards) or count_straight_draw(my_cards, board_cards))
            if RaiseAction in legal_actions and has_draw and random.random() < 0.18:
                min_raise, max_raise = round_state.raise_bounds()
                raise_amount = int(0.55 * (pot + continue_cost))
                target = opp_pip + raise_amount
                target = clamp_raise(target, min_raise, max_raise)
                return RaiseAction(target)

            return FoldAction() if can_fold else CallAction()

        # No bet to face: choose between betting and checking
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

            # Value bet
            if strength >= value_thresh:
                bet_amount = int(0.7 * pot) if pot > 0 else (BIG_BLIND * 2)
                target = my_pip + bet_amount
                target = clamp_raise(target, min_raise, max_raise)
                if random.random() < 0.80:
                    return RaiseAction(target)

            # Semi-bluff with good draws
            has_draw = (len(board_cards) >= 3) and (count_flush_draw(my_cards, board_cards) or count_straight_draw(my_cards, board_cards))
            if has_draw and strength >= call_thresh and random.random() < 0.22:
                bet_amount = int(0.55 * pot) if pot > 0 else BIG_BLIND
                target = my_pip + bet_amount
                target = clamp_raise(target, min_raise, max_raise)
                return RaiseAction(target)

        # Default: check
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())