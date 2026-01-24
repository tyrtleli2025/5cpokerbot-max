'''
Improved Pokerbot v2 with:
1. Very tight pre-flop shoving (QQ+/trips ONLY - no TT/JJ shoves)
2. Caution mode when ahead (stop shoving, play small ball)
3. Larger post-flop bet sizing (minimum 6 chips)
4. Only bet with strong hands (0.50+ strength)
5. Pot odds-based calling decisions
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState, NUM_ROUNDS, BIG_BLIND, STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    def __init__(self):
        self.rank_map = {r: i for i, r in enumerate("23456789TJQKA")}

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board

        # ---------------------------------------------------------
        # 1. HANDLE DISCARD ACTION
        # ---------------------------------------------------------
        if DiscardAction in legal:
            return self.get_discard_action(my_cards, board)

        # ---------------------------------------------------------
        # 2. LOCKDOWN CHECK (Safety Mode)
        # ---------------------------------------------------------
        rounds_remaining = NUM_ROUNDS - game_state.round_num + 1
        secure_win_threshold = (rounds_remaining * 1.5) + 2

        if game_state.bankroll > secure_win_threshold:
            if CheckAction in legal:
                return CheckAction()
            return FoldAction()

        # ---------------------------------------------------------
        # 2b. CAUTION MODE (When ahead, stop shoving)
        # ---------------------------------------------------------
        # If we're ahead by 150+ chips, play more conservatively
        # Don't risk our lead with all-ins
        is_caution_mode = game_state.bankroll >= 150

        # ---------------------------------------------------------
        # 3. CALCULATE BETTING SITUATION
        # ---------------------------------------------------------
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]

        continue_cost = opp_pip - my_pip
        pot_total = my_pip + opp_pip

        is_opponent_allin = (opp_stack == 0)
        is_huge_bet = (continue_cost > my_stack * 0.5)

        # ---------------------------------------------------------
        # 4. PRE-FLOP STRATEGY
        # ---------------------------------------------------------
        if street == 0:
            hand_tier = self.get_preflop_tier(my_cards)

            # FACING ALL-IN OR HUGE BET
            if is_opponent_allin or is_huge_bet:
                # Only call with SUPER PREMIUM hands (QQ+, trips)
                # Don't call all-ins with TT/JJ - they lose too often
                if hand_tier == 1:
                    # Check if it's actually QQ+ or trips
                    ranks = [self.rank_map[c[0]] for c in my_cards]
                    rank_counts = {}
                    for r in ranks:
                        rank_counts[r] = rank_counts.get(r, 0) + 1

                    has_trips = any(c == 3 for c in rank_counts.values())
                    pair_rank = None
                    for r, c in rank_counts.items():
                        if c == 2:
                            pair_rank = r

                    # Only call with QQ+ (Q=10) or trips
                    if has_trips or (pair_rank is not None and pair_rank >= 10):
                        if CallAction in legal:
                            return CallAction()

                if CheckAction in legal:
                    return CheckAction()
                return FoldAction()

            # CAUTION MODE: Don't shove, just raise small
            if is_caution_mode:
                if hand_tier == 1:
                    if RaiseAction in legal:
                        min_raise, max_raise = round_state.raise_bounds()
                        # Raise 4-5x BB instead of all-in
                        raise_amount = min(min_raise + BIG_BLIND * 3, max_raise)
                        return RaiseAction(raise_amount)
                    if CallAction in legal:
                        return CallAction()
                elif hand_tier == 2:
                    if RaiseAction in legal:
                        min_raise, _ = round_state.raise_bounds()
                        return RaiseAction(min_raise)
                    if CallAction in legal:
                        return CallAction()
                if CheckAction in legal:
                    return CheckAction()
                return FoldAction()

            # NORMAL MODE: WE ACT FIRST OR FACING SMALL BET
            if hand_tier == 1:
                # Check if it's QQ+ or trips (worth shoving)
                ranks = [self.rank_map[c[0]] for c in my_cards]
                rank_counts = {}
                for r in ranks:
                    rank_counts[r] = rank_counts.get(r, 0) + 1

                has_trips = any(c == 3 for c in rank_counts.values())
                pair_rank = None
                for r, c in rank_counts.items():
                    if c == 2:
                        pair_rank = r

                if RaiseAction in legal:
                    min_raise, max_raise = round_state.raise_bounds()

                    # Only shove with QQ+ (Q=10) or trips
                    if has_trips or (pair_rank is not None and pair_rank >= 10):
                        return RaiseAction(max_raise)
                    else:
                        # TT/JJ: raise 5-6x BB, don't shove
                        raise_amount = min(min_raise + BIG_BLIND * 4, max_raise)
                        return RaiseAction(raise_amount)

                if CallAction in legal:
                    return CallAction()
                if CheckAction in legal:
                    return CheckAction()

            elif hand_tier == 2:
                # MEDIUM HAND -> Small raise or call, NOT all-in
                if RaiseAction in legal:
                    min_raise, max_raise = round_state.raise_bounds()
                    # Raise 3-4x the big blind, not all-in
                    raise_amount = min(min_raise + BIG_BLIND * 2, max_raise)
                    return RaiseAction(raise_amount)
                if CallAction in legal:
                    return CallAction()
                if CheckAction in legal:
                    return CheckAction()

            else:
                # WEAK HAND -> Check or fold
                if CheckAction in legal:
                    return CheckAction()
                # Only call the minimum blind
                if CallAction in legal and continue_cost <= BIG_BLIND:
                    return CallAction()
                return FoldAction()

        # ---------------------------------------------------------
        # 5. POST-FLOP STRATEGY
        # ---------------------------------------------------------
        hand_strength = self.evaluate_postflop_strength(my_cards, board)

        # FACING A BET
        if CallAction in legal and continue_cost > 0:
            pot_odds = continue_cost / (pot_total + continue_cost)

            # Against all-in: only call with very strong hands
            if is_opponent_allin or is_huge_bet:
                if hand_strength >= 0.70:  # Strong made hand
                    return CallAction()
                return FoldAction()

            # Normal pot odds decision
            if hand_strength > pot_odds:
                return CallAction()
            return FoldAction()

        # WE CAN BET OR CHECK
        if RaiseAction in legal:
            min_raise, max_raise = round_state.raise_bounds()

            # Minimum bet size of 6 chips to pressure opponent
            min_bet = max(min_raise, 6)

            # STRONG HAND (trips+, two pair, flush, straight) -> Bet for value
            if hand_strength >= 0.65:
                # Bet around 60-75% of pot, minimum 6 chips
                bet_size = min(max(min_bet, int(pot_total * 0.7)), max_raise)
                return RaiseAction(bet_size)

            # GOOD DRAW (flush draw or straight draw) -> Semi-bluff
            if self.has_strong_draw(my_cards, board):
                # Semi-bluff with decent sizing
                bet_size = min(max(min_bet, int(pot_total * 0.5)), max_raise)
                return RaiseAction(bet_size)

            # MEDIUM-STRONG HAND (good pair) -> Value bet
            # Only bet with strength >= 0.50 (not 0.40 - too weak)
            if hand_strength >= 0.50:
                bet_size = min(max(min_bet, int(pot_total * 0.5)), max_raise)
                return RaiseAction(bet_size)

        # Default: check with weak hands
        if CheckAction in legal:
            return CheckAction()

        return FoldAction()

    def get_discard_action(self, my_cards, board):
        '''Smart discard: keep high cards and flush potential'''
        board_suits = [c[1] for c in board]

        flush_suit = None
        if len(board_suits) >= 2:
            suit_counts = {}
            for s in board_suits:
                suit_counts[s] = suit_counts.get(s, 0) + 1
            for s, count in suit_counts.items():
                if count >= 2:
                    flush_suit = s
                    break

        best_discard_index = 0
        lowest_value = float('inf')

        for i, card in enumerate(my_cards):
            rank = self.rank_map[card[0]]
            suit = card[1]

            # Value = rank (higher is better to keep)
            value = rank

            # Bonus for matching flush suit on board
            if flush_suit and suit == flush_suit:
                value += 15

            # Bonus for having a pair in hand
            other_ranks = [self.rank_map[my_cards[j][0]] for j in range(len(my_cards)) if j != i]
            if rank in other_ranks:
                value += 20  # Keep paired cards

            if value < lowest_value:
                lowest_value = value
                best_discard_index = i

        return DiscardAction(best_discard_index)

    def get_preflop_tier(self, cards):
        '''
        Returns hand tier:
        1 = Premium (only hands worth shoving): High pairs (TT+), Trips
        2 = Medium (worth raising small): Medium pairs (55-99), High suited
        3 = Weak (check/fold)
        '''
        ranks = [self.rank_map[c[0]] for c in cards]
        suits = [c[1] for c in cards]

        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        # Check for trips - always premium
        for count in rank_counts.values():
            if count == 3:
                return 1

        # Check for pairs
        for r, count in rank_counts.items():
            if count == 2:
                if r >= 8:  # TT+ (T=8, J=9, Q=10, K=11, A=12)
                    return 1  # Premium
                elif r >= 3:  # 55-99
                    return 2  # Medium
                else:  # 22-44
                    return 3  # Weak

        # High cards (no pair) - suited with high cards is medium, otherwise weak
        if len(set(suits)) == 1:  # All suited
            high_count = sum(1 for r in ranks if r >= 9)  # J+
            if high_count >= 2:
                return 2  # Medium - suited with 2+ high cards

        # High cards without pair
        if sum(1 for r in ranks if r >= 11) >= 2:  # 2+ cards K or A
            return 2

        return 3  # Weak

    def evaluate_postflop_strength(self, my_cards, board):
        '''
        Returns strength score 0.0 to 1.0
        '''
        all_cards = list(my_cards) + list(board)
        ranks = [self.rank_map[c[0]] for c in all_cards]
        suits = [c[1] for c in all_cards]

        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1

        # Check for flush
        for count in suit_counts.values():
            if count >= 5:
                return 0.85

        # Check for quads
        for count in rank_counts.values():
            if count == 4:
                return 0.95

        # Check for full house (trips + pair)
        has_trips = any(c >= 3 for c in rank_counts.values())
        has_pair = any(c == 2 for c in rank_counts.values())
        if has_trips and has_pair:
            return 0.90

        # Check for trips
        if has_trips:
            return 0.75

        # Check for straight
        sorted_ranks = sorted(set(ranks))
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i + 4] - sorted_ranks[i] == 4:
                return 0.80
        # Wheel straight
        if set([0, 1, 2, 3, 12]).issubset(set(ranks)):
            return 0.80

        # Check for two pair
        pair_count = sum(1 for c in rank_counts.values() if c == 2)
        if pair_count >= 2:
            return 0.60

        # Check for one pair
        if pair_count == 1:
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            # Check if our hole cards made the pair
            my_ranks = [self.rank_map[c[0]] for c in my_cards]
            if my_ranks.count(pair_rank) >= 1:
                # We have a pair using our cards
                board_ranks = [self.rank_map[c[0]] for c in board]
                max_board = max(board_ranks) if board_ranks else 0

                if pair_rank > max_board:
                    # Overpair
                    return 0.55 + (pair_rank / 12) * 0.15
                elif pair_rank == max_board:
                    # Top pair
                    return 0.45 + (pair_rank / 12) * 0.10
                else:
                    # Underpair or middle pair
                    return 0.30 + (pair_rank / 12) * 0.10
            else:
                # Board paired, we don't have it
                return 0.25

        # High card
        my_ranks = [self.rank_map[c[0]] for c in my_cards]
        max_rank = max(my_ranks)
        return 0.15 + (max_rank / 12) * 0.10

    def has_strong_draw(self, my_cards, board):
        '''Returns True if we have a flush draw or open-ended straight draw'''
        all_cards = list(my_cards) + list(board)
        suits = [c[1] for c in all_cards]
        ranks = [self.rank_map[c[0]] for c in all_cards]

        # Flush draw (4 of same suit)
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        if any(c == 4 for c in suit_counts.values()):
            return True

        # Open-ended straight draw (4 consecutive)
        sorted_ranks = sorted(set(ranks))
        consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] - sorted_ranks[i-1] == 1:
                consecutive += 1
                if consecutive >= 4:
                    return True
            else:
                consecutive = 1

        return False

if __name__ == '__main__':
    run_bot(Player(), parse_args())
