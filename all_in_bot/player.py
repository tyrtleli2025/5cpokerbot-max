'''
Simple All-In or Fold Pokerbot
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    '''
    A bot that goes All-In Pre-flop with pairs, suited cards, trips, or connectors.
    Otherwise, it folds.
    '''

    def __init__(self):
        pass

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
            # Simple Strategy: Discard the lowest rank card
            # Ranks: 2=0, 3=1, ... A=12
            rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
            my_ranks = [rank_map[c[0]] for c in my_cards]
            
            # Find index of lowest card
            min_rank = min(my_ranks)
            discard_index = my_ranks.index(min_rank)
            return DiscardAction(discard_index)

        # ---------------------------------------------------------
        # 2. PRE-FLOP STRATEGY (The "All-In" Logic)
        # ---------------------------------------------------------
        if street == 0:
            if self.is_good_preflop(my_cards):
                # GO ALL IN
                if RaiseAction in legal:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(max_raise)
                # If we can't raise (e.g. facing a shove), Call
                if CallAction in legal:
                    return CallAction()
                if CheckAction in legal:
                    return CheckAction()
            else:
                # BAD HAND -> FOLD
                # (Unless we can check for free)
                if CheckAction in legal:
                    return CheckAction()
                return FoldAction()

        # ---------------------------------------------------------
        # 3. POST-FLOP / WAITING TURNS
        # ---------------------------------------------------------
        # If we made it here, we either went all-in or checked through.
        # Just check/call to try and see the showdown.
        
        if CheckAction in legal:
            return CheckAction()
        if CallAction in legal:
            return CallAction()
        return FoldAction()

    def is_good_preflop(self, cards):
        '''
        Returns True if hand is Pair, Suited, 3-of-a-kind, or Connectors.
        '''
        rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
        ranks = [rank_map[c[0]] for c in cards]
        suits = [c[1] for c in cards]
        
        # 1. Pairs / 3-of-a-kind
        # If unique ranks < 3, we have duplicates (Pair or Trips)
        if len(set(ranks)) < 3:
            return True

        # 2. Same Suit (Flush potential)
        if len(set(suits)) == 1:
            return True

        # 3. Connectors (Straight potential)
        # Sort ranks and check if they are sequential (e.g. 5, 6, 7)
        sorted_ranks = sorted(ranks)
        if (sorted_ranks[1] == sorted_ranks[0] + 1) and (sorted_ranks[2] == sorted_ranks[1] + 1):
            return True
        
        # Special Case: Wheel Straight (A, 2, 3) -> Ranks [12, 0, 1]
        if set(sorted_ranks) == {0, 1, 12}:
            return True

        return False

if __name__ == '__main__':
    run_bot(Player(), parse_args())