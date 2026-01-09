'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random


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
        Improved strategy to fix the Pre-flop/Flop leak.
        '''
        legal_actions = round_state.legal_actions()  #
        street = round_state.street  #
        my_cards = round_state.hands[active]  #
        board_cards = round_state.board  #
        my_pip = round_state.pips[active]  #
        opp_pip = round_state.pips[1-active]  #

        # Helper: Map ranks to integers
        rank_map = {r: i for i, r in enumerate("23456789TJQKA", 2)}
        my_ranks = [rank_map[c[0]] for c in my_cards]
        my_suits = [c[1] for c in my_cards]

        # ---------------------------------------------------------
        # STREET 2 & 3: DISCARD PHASE
        # Logic: Always discard the lowest rank card.
        # ---------------------------------------------------------
        if DiscardAction in legal_actions:
            min_rank = min(my_ranks)
            discard_index = my_ranks.index(min_rank)
            return DiscardAction(discard_index)

        # ---------------------------------------------------------
        # STREET 0: PRE-FLOP
        # Fix: Distinguish between "Premium" (Raise) and "Speculative" (Call)
        # ---------------------------------------------------------
        if street == 0:
            # Premium: High Pairs (10+), or 3 Big Cards (10+), or High Suited (10+)
            is_premium = (
                (len(set(my_ranks)) <= 2 and max(my_ranks) >= 10) or
                (all(r >= 10 for r in my_ranks)) or
                (len(set(my_suits)) == 1 and max(my_ranks) >= 10)
            )
            
            # Speculative: Any Pair, Any Suited, or Connected (e.g. 5,6,7)
            my_ranks_sorted = sorted(my_ranks)
            is_connected = (my_ranks_sorted[0] + 1 == my_ranks_sorted[1] and my_ranks_sorted[1] + 1 == my_ranks_sorted[2])
            is_speculative = len(set(my_ranks)) <= 2 or len(set(my_suits)) == 1 or is_connected

            if is_premium:
                # Raise big with great hands
                if RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(max_raise)
                return CallAction()
            
            elif is_speculative:
                # Just Call (limp) with decent hands to see the flop cheaply
                if CallAction in legal_actions:
                    return CallAction()
                return CheckAction()
            
            else:
                # Trash hands -> Fold immediately
                if CheckAction in legal_actions:
                    return CheckAction()
                return FoldAction()

       # ---------------------------------------------------------
        # STREET 4, 5, 6: POST-FLOP
        # ---------------------------------------------------------
        all_cards = my_cards + board_cards
        all_ranks = [rank_map[c[0]] for c in all_cards]
        all_suits = [c[1] for c in all_cards]
        
        # --- 1. Check Pair / Trips / Full House ---
        has_pair_or_better = len(set(all_ranks)) < len(all_ranks)

        # --- 2. Check Flush (5+ cards of same suit) ---
        suit_counts = {}
        for s in all_suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        is_flush = any(count >= 5 for count in suit_counts.values())
        # (Optional: Keep the draw logic for semi-bluffing)
        is_flush_draw = any(count == 4 for count in suit_counts.values())

        # --- 3. Check Straight (5 consecutive ranks) ---
        unique_ranks = sorted(list(set(all_ranks)))
        is_straight = False
        # Check standard straight (e.g., 5,6,7,8,9)
        if len(unique_ranks) >= 5:
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    is_straight = True
                    break
        # Check Wheel straight (A, 2, 3, 4, 5) -> A is 14
        if not is_straight and 14 in unique_ranks:
            # If we have A, we need 2,3,4,5
            needed = {2, 3, 4, 5}
            if needed.issubset(set(unique_ranks)):
                is_straight = True

        # --- Final Decision Logic ---
        
        # Now "Strong" includes Straights and Flushes
        is_strong = has_pair_or_better or is_flush or is_straight

        # Pot Odds / Aggression Check
        pot_total = my_pip + opp_pip
        is_continuation_bet_opportunity = (pot_total > 20) 

        if is_strong:
            # Value Bet: Raise max with the goods
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(max_raise)
            return CallAction()
        
        elif is_flush_draw or is_continuation_bet_opportunity:
            # Semi-Bluff or C-Bet
            if RaiseAction in legal_actions:
                 min_raise, max_raise = round_state.raise_bounds()
                 return RaiseAction(int((min_raise + max_raise) / 2))
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction()

        # Weak hand -> Check/Fold
        if CheckAction in legal_actions:
            return CheckAction()
        return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())
