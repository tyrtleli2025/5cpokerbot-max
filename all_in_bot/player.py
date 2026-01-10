'''
Simple All-In or Fold Pokerbot with "Lockdown" Safety Mode
Updated to Call Big Blinds with weak hands.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState, NUM_ROUNDS, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    '''
    A bot that:
    1. Goes All-In Pre-flop with strong hands.
    2. Calls the Big Blind (limps) with weak hands, then Check/Folds.
    3. Uses smart discard logic (denying flushes/keeping connectivity).
    4. Enters "Lockdown Mode" (Check/Fold) if it has mathematically won.
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
        # 1. HANDLE DISCARD ACTION (Always required if legal)
        # ---------------------------------------------------------
        if DiscardAction in legal:
            # Smart Strategy: Avoid discarding flush connectivity or high cards
            rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
            
            # Analyze Board Texture
            board_suits = [c[1] for c in board]
            
            # Check if board has flush potential (2 of same suit)
            flush_suit = None
            if len(board_suits) == 2 and board_suits[0] == board_suits[1]:
                flush_suit = board_suits[0]
            
            best_discard_index = 0
            lowest_danger_score = float('inf')
            
            for i, card in enumerate(my_cards):
                card_rank = rank_map[card[0]]
                card_suit = card[1]
                
                # Base Danger Score = Rank (0-12)
                # Higher rank = Higher danger to leave on board for opponent
                danger_score = card_rank
                
                # Flush Danger Penalty
                if flush_suit and card_suit == flush_suit:
                    danger_score += 100
                
                if danger_score < lowest_danger_score:
                    lowest_danger_score = danger_score
                    best_discard_index = i
            
            return DiscardAction(best_discard_index)

        # ---------------------------------------------------------
        # 2. LOCKDOWN CHECK (Safety Mode)
        # ---------------------------------------------------------
        # Calculate rounds remaining (including current one)
        rounds_remaining = NUM_ROUNDS - game_state.round_num + 1
        
        # Average cost to fold every hand is 1.5 chips/round.
        # If we have more chips than we can possibly lose, stop betting.
        # We add a small buffer (+2) to handle the variance of ending on a Big Blind.
        secure_win_threshold = (rounds_remaining * 1.5) + 2
        
        if game_state.bankroll > secure_win_threshold:
            # We have won. Do not risk any chips.
            if CheckAction in legal:
                return CheckAction()
            return FoldAction()

        # ---------------------------------------------------------
        # 3. PRE-FLOP STRATEGY
        # ---------------------------------------------------------
        if street == 0:
            if self.is_good_preflop(my_cards):
                # GOOD HAND -> GO ALL IN
                if RaiseAction in legal:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(max_raise)
                if CallAction in legal:
                    return CallAction()
                if CheckAction in legal:
                    return CheckAction()
            else:
                # BAD HAND -> CHECK OR CALL BLIND
                
                # 1. If we can Check (we are BB and no raise), just Check.
                if CheckAction in legal:
                    return CheckAction()
                
                # 2. If we can Call, check if it's cheap (just the Big Blind).
                # If opp_pip > BIG_BLIND, they raised, so we should Fold.
                # If opp_pip <= BIG_BLIND, it's just the blind, so we Call.
                opp_pip = round_state.pips[1-active]
                
                if CallAction in legal and opp_pip <= BIG_BLIND:
                    return CallAction()
                
                # 3. Otherwise (facing a raise), Fold.
                return FoldAction()

        # ---------------------------------------------------------
        # 4. POST-FLOP STRATEGY
        # ---------------------------------------------------------
        # If we didn't fold pre-flop, we are likely All-In or checked through.
        
        # Always Check if possible
        if CheckAction in legal:
            return CheckAction()
        
        # Fold to any aggression (since we only have a bad hand here)
        return FoldAction()

    def is_good_preflop(self, cards):
        '''
        Returns True if hand is:
        - Trips
        - Pair (55+)
        - 3 of the same suit AND sum of card values >= 25
        '''
        rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
        ranks = [rank_map[c[0]] for c in cards]
        suits = [c[1] for c in cards]
        
        # Count rank frequencies
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
            
        is_pair = False
        is_trips = False
        pair_rank = -1
        
        for r, count in rank_counts.items():
            if count == 2:
                is_pair = True
                pair_rank = r
            elif count == 3:
                is_trips = True
        
        # 1. Pairs / Trips Logic
        if is_trips:
            return True
            
        if is_pair:
            # If pair is 2, 3, or 4 (indices 0, 1, 2), fold
            if pair_rank <= 2:
                return False
            # Otherwise (55+), it's good
            return True

        # 2. Suited Logic (Flush potential)
        if len(set(suits)) == 1:
            # Sum of face values (2=2 ... A=14)
            total_val = sum(r + 2 for r in ranks)
            if total_val >= 25:
                return True
            else:
                return False

        return False

if __name__ == '__main__':
    run_bot(Player(), parse_args())