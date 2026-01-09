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

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        try:
            my_cards = round_state.hands[active]
            board = round_state.board
            
            # 1. Build History String
            # We need to reconstruct the simplified history string from round_state
            # Examples: "", "R", "RC/", "RC/d0/"
            # This is hard to do perfectly without tracking every move, 
            # so we'll use a simplified mapping based on Street + Pips.
            
            # Simple heuristic for history:
            history = ""
            if round_state.street == 0:
                pips = round_state.pips
                if pips[0] == 1 and pips[1] == 2: history = "" # Start
                elif pips[active] < pips[1-active]: history = "R" # Facing raise
            
            # 2. Get Abstraction
            info_set = get_abstraction(my_cards, board, history)

            # 3. Discard Phase Specifics
            if DiscardAction in round_state.legal_actions():
                # We need to decide which card to drop.
                # In the trainer, this was actions 3,4,5 (indices 0,1,2)
                # We can check the strategy for "InfoSet|d"
                # OR we can just use the Evaluator directly since Training discard is noisy.
                
                # HYBRID APPROACH: Use Evaluator for Discard (Safer)
                # Try dropping each card and see which remaining hand is strongest
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

            # 4. Betting Strategy Lookup
            if info_set in self.strategy_map:
                probs = self.strategy_map[info_set]
                # Map 0->Fold, 1->Call, 2->Raise
                action_idx = random.choices([0, 1, 2], weights=probs)[0]
                
                legal = round_state.legal_actions()
                if action_idx == 2 and RaiseAction in legal:
                    min_r, max_r = round_state.raise_bounds()
                    # Raise 75% of pot (standard sizing)
                    amt = int(min_r + (max_r - min_r) * 0.75)
                    return RaiseAction(amt)
                if action_idx == 1:
                    return CallAction() if CallAction in legal else CheckAction()
                if action_idx == 0:
                    return FoldAction() if FoldAction in legal else CheckAction()

            # 5. Fallback (If state not in strategy)
            # Play "Honest" Geometry
            strength = evaluate_strength(my_cards, board)
            legal = round_state.legal_actions()
            
            # If monster, raise
            if strength > 80 and RaiseAction in legal:
                min_r, max_r = round_state.raise_bounds()
                return RaiseAction(max_r)
            # If decent, call
            if strength > 50:
                return CallAction() if CallAction in legal else CheckAction()
            
            return CheckAction() if CheckAction in legal else FoldAction()

        except Exception as e:
            return CheckAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())