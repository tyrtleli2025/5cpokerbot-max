'''
SANITY CHECK BOT
No libraries. No files. Pure Python.
If this bot folds, the server or your account is broken.
'''
import random
import sys

# Standard skeleton imports only
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    def __init__(self):
        # Absolutely nothing here that can crash
        pass

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        try:
            legal = round_state.legal_actions()
            
            # ------------------------------------------------
            # 1. DISCARD LOGIC (Pure Python, No Treys)
            # ------------------------------------------------
            if DiscardAction in legal:
                # Simple Heuristic: Keep the highest rank cards
                # Ranks: 2=0, ... A=12
                ranks = "23456789TJQKA"
                my_cards = round_state.hands[active] # List of strings like ['As', '2h', 'Kd']
                
                # Find the card with the LOWEST rank index to discard
                worst_card_idx = 0
                min_rank_val = 100
                
                for i, card_str in enumerate(my_cards):
                    rank_char = card_str[0]
                    rank_val = ranks.index(rank_char)
                    if rank_val < min_rank_val:
                        min_rank_val = rank_val
                        worst_card_idx = i
                
                return DiscardAction(worst_card_idx)

            # ------------------------------------------------
            # 2. BETTING LOGIC (Blind Aggression)
            # ------------------------------------------------
            # We will ALWAYS Call or Check. 
            # If this bot folds, it means the code didn't run at all.
            
            if CheckAction in legal:
                return CheckAction()
            
            if CallAction in legal:
                return CallAction()
                
            # If we can't Check or Call (rare), we fold
            return FoldAction()

        except Exception as e:
            # If this prints, we have a syntax error
            print(f"FATAL ERROR: {e}")
            return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())