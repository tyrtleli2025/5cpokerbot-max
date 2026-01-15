'''
Range-Merged "Balanced Bully" Bot
Protects against intelligent bots by disguising hand strength via unified bet sizing.
Includes original Lockdown Mode.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState, NUM_ROUNDS, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    def __init__(self):
        self.consecutive_uncalled_shoves = 0
        self.bully_threshold = 10 
        self.is_bully_mode = False
        self.did_shove_this_round = False
        self.starting_stack = 400 # Fixed: Removed citation tags

    def handle_new_round(self, game_state, round_state, active):
        self.did_shove_this_round = False

    def handle_round_over(self, game_state, terminal_state, active):
        # Update adaptive counters based on results
        my_delta = terminal_state.deltas[active]
        if self.did_shove_this_round:
            # If we won small (blind steal)
            if my_delta > 0 and my_delta < 10:
                self.consecutive_uncalled_shoves += 1
            else:
                # If we got called (win big or lose), reset.
                self.consecutive_uncalled_shoves = 0
                self.is_bully_mode = False

        if self.consecutive_uncalled_shoves >= self.bully_threshold:
            self.is_bully_mode = True

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
        # 2. LOCKDOWN CHECK (Safety Mode) - EXACT ORIGINAL LOGIC
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
            is_strong = self.is_good_preflop(my_cards)
            
            # --- BULLY MODE (Range Merging) ---
            if self.is_bully_mode:
                # Check for Aggression (Opponent raised?)
                opp_pip = round_state.pips[1-active]
                
                # CASE A: We face a Raise (Pip > 2)
                if opp_pip > 2:
                    if is_strong:
                        # TRAP SPRUNG: We have a monster and they raised!
                        # JAM ALL-IN over their raise.
                        if RaiseAction in legal:
                            min_r, max_r = round_state.raise_bounds()
                            self.did_shove_this_round = True
                            return RaiseAction(max_r)
                        return CallAction()
                    else:
                        # We were stealing with trash. Fold.
                        return FoldAction()
                
                # CASE B: Unopened Pot (We act first or they limped)
                # We Min-Raise with 100% of our range (Both Strong and Weak)
                # This disguises our hand.
                if RaiseAction in legal:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(min_raise)
                    
                # If we can't raise for some reason, check/call
                if CheckAction in legal: return CheckAction()
                return CallAction()

            # --- STANDARD MODE (GTO-ish Shoving) ---
            # Against unknown opponents, we unbalance towards value-shoving
            if is_strong:
                if RaiseAction in legal:
                    min_r, max_r = round_state.raise_bounds()
                    self.did_shove_this_round = True
                    return RaiseAction(max_r)
                if CallAction in legal: return CallAction()
                if CheckAction in legal: return CheckAction()
            else:
                # Standard play with weak hands: Check or Fold
                if CheckAction in legal: return CheckAction()
                # Limp if cheap
                if CallAction in legal and round_state.pips[1-active] <= BIG_BLIND:
                    return CallAction()
                return FoldAction()

        # ---------------------------------------------------------
        # 4. POST-FLOP STRATEGY
        # ---------------------------------------------------------
        # If we have a STRONG hand post-flop (we hit the board or had pocket pair)
        # We should bet for value.
        # Simplified: If we have "Good Preflop" cards, we assume we are still decent.
        if self.is_good_preflop(my_cards):
             if RaiseAction in legal:
                 # Let's Jam to deny equity if we are strong
                 min_r, max_r = round_state.raise_bounds()
                 return RaiseAction(max_r)
        
        # If we are weak post-flop:
        if CheckAction in legal: return CheckAction()
        return FoldAction()

    def get_discard_action(self, my_cards, board):
        rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
        board_suits = [c[1] for c in board]
        flush_suit = None
        if len(board_suits) == 2 and board_suits[0] == board_suits[1]:
            flush_suit = board_suits[0]
        
        best_discard_index = 0
        lowest_danger_score = float('inf')
        
        for i, card in enumerate(my_cards):
            card_rank = rank_map[card[0]]
            card_suit = card[1]
            
            # Value = Rank + FlushBonus
            value = card_rank
            if flush_suit and card_suit == flush_suit:
                value += 20 
            
            if value < lowest_danger_score:
                lowest_danger_score = value
                best_discard_index = i
        
        return DiscardAction(best_discard_index)

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
        
        if is_trips: return True
        if is_pair:
            # Fold small pairs 22, 33, 44
            if pair_rank <= 2: return False
            return True

        if len(set(suits)) == 1:
            total_val = sum(r + 2 for r in ranks)
            if total_val >= 25: return True

        return False

if __name__ == '__main__':
    run_bot(Player(), parse_args())