import sys
import pickle
import random
import os
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from hand_utils import get_abstraction, evaluate_strength

class Player(Bot):
    def __init__(self):
        self.strategy_map = {}
        if os.path.exists("strategy.pkl"):
            with open("strategy.pkl", "rb") as f:
                self.strategy_map = pickle.load(f)
                # print(f"Loaded strategy with {len(self.strategy_map)} states")

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        try:
            legal = round_state.legal_actions()
            my_cards = round_state.hands[active]
            board = round_state.board
            street = round_state.street

            # ---------------------------------------------------------
            # 1. HANDLE DISCARD PHASE (Street 2 & 3)
            # ---------------------------------------------------------
            
            # CASE A: It is our turn to Discard
            if DiscardAction in legal:
                # Hybrid: Use the Evaluator to drop the worst card
                best_idx = 0
                best_score = -1
                for i in range(3):
                    temp_hand = my_cards[:]
                    temp_hand.pop(i)
                    score = evaluate_strength(temp_hand, board)
                    if score > best_score:
                        best_score = score
                        best_idx = i
                return DiscardAction(best_idx)

            # CASE B: It is the Discard Phase, but we are waiting for opponent [FIX]
            if street in (2, 3):
                # We cannot Bet, Call, or Raise here. We MUST Check.
                return CheckAction()

            # ---------------------------------------------------------
            # 2. HANDLE BETTING (Pre-flop & Post-flop)
            # ---------------------------------------------------------
            
            # Build History String (Heuristic)
            history = ""
            if street == 0:
                pips = round_state.pips
                if pips[0] == 1 and pips[1] == 2: history = "" # Start
                elif pips[active] < pips[1-active]: history = "R" # Facing raise
            
            # Get Abstraction
            info_set = get_abstraction(my_cards, board, history)

            # Strategy Lookup
            if info_set in self.strategy_map:
                probs = self.strategy_map[info_set]
                # Map 0->Fold, 1->Call, 2->Raise
                action_idx = random.choices([0, 1, 2], weights=probs)[0]
                
                # --- SAFELY EXECUTE CHOSEN ACTION ---
                
                # Attempt RAISE
                if action_idx == 2:
                    if RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        # Raise 75% of pot
                        amt = int(min_r + (max_r - min_r) * 0.75)
                        return RaiseAction(amt)
                    # If we can't raise, fall through to Call
                    action_idx = 1
                
                # Attempt CALL
                if action_idx == 1:
                    if CallAction in legal: return CallAction()
                    if CheckAction in legal: return CheckAction()
                    # If we can't Call or Check, we must Fold
                    return FoldAction()
                
                # Attempt FOLD
                if action_idx == 0:
                    if CheckAction in legal: return CheckAction() # Never fold for free
                    return FoldAction()

            # ---------------------------------------------------------
            # 3. FALLBACK (Honest Play)
            # ---------------------------------------------------------
            strength = evaluate_strength(my_cards, board)
            
            # Strong: Raise if possible
            if strength > 80 and RaiseAction in legal:
                min_r, max_r = round_state.raise_bounds()
                return RaiseAction(max_r)
            
            # Decent: Call
            if strength > 50:
                if CallAction in legal: return CallAction()
                if CheckAction in legal: return CheckAction()
            
            # Weak: Check/Fold
            if CheckAction in legal: return CheckAction()
            return FoldAction()

        except Exception as e:
            # Emergency Safety Net
            # Do NOT just return CheckAction, because it might be illegal (e.g. facing a bet)
            legal = round_state.legal_actions()
            if CheckAction in legal: return CheckAction()
            if CallAction in legal: return CallAction()
            return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())