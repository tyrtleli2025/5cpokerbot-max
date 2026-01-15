'''
Test Bot: "The Mouse"
1. NEVER calls a bet (Folds to all aggression).
2. Min-Bets the Flop if checked to.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class TestPlayer(Bot):

    def __init__(self):
        pass

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        
        # ---------------------------------------------------------
        # 1. HANDLE DISCARD (Required)
        # ---------------------------------------------------------
        if DiscardAction in legal:
            # Simple discard: just discard the first card (index 0)
            return DiscardAction(0)

        # ---------------------------------------------------------
        # 2. FLOP AGGRESSION (Street 4)
        # ---------------------------------------------------------
        # Street 4 is the betting round immediately following the discard phase (The Flop).
        # If we can Check (meaning opponent didn't bet), we seize initiative and Bet Small.
        if street == 4 and CheckAction in legal and RaiseAction in legal:
            min_raise, max_raise = round_state.raise_bounds()
            return RaiseAction(min_raise)

        # ---------------------------------------------------------
        # 3. "NEVER CALL" LOGIC
        # ---------------------------------------------------------
        # If we can Check, we Check (unless we triggered the Flop bet above).
        if CheckAction in legal:
            return CheckAction()
            
        # If we cannot Check, it means we are facing a bet.
        # We NEVER call. We Fold immediately.
        return FoldAction()

if __name__ == '__main__':
    run_bot(TestPlayer(), parse_args())