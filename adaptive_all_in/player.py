'''
Final Adaptive Pokerbot
- Safety Lockdown (Buffer included)
- Adaptive Pre-Flop Aggression (Tight vs Bully)
- Smart Discarding
- Post-Flop "Monster" Detection (Quads & Full Houses ONLY)
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState, NUM_ROUNDS
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    def __init__(self):
        # Stats tracking
        self.total_shoves = 0
        self.total_calls_faced = 0
        
        # Strategy constants
        self.SHOVE_SAMPLE_MIN = 10     
        self.LOOSE_CALL_THRESHOLD = 0.15  # If they fold > 85%, we go Bully Mode
        
        self.lockdown_mode = False
        self.did_shove_this_round = False

    def handle_new_round(self, game_state, round_state, active):
        self.did_shove_this_round = False
        
        # ---------------------------------------------------------
        # SAFETY LOCKDOWN CALCULATION
        # ---------------------------------------------------------
        rounds_remaining = NUM_ROUNDS - game_state.round_num + 1
        
        # MATH: Max cost of a single round is 2 (Big Blind). Average is 1.5.
        # We need a buffer of at least 0.5 to handle ending on a BB.
        # We use +5.0 just to be safe and clean.
        secure_win_threshold = (rounds_remaining * 1.5) + 5.0
        
        if game_state.bankroll > secure_win_threshold:
            self.lockdown_mode = True

    def handle_round_over(self, game_state, terminal_state, active):
        # Update Data for Adaptive Strategy
        if self.did_shove_this_round:
            self.total_shoves += 1
            # If delta is large (>50), it means they called our shove
            if abs(terminal_state.deltas[active]) > 50:
                self.total_calls_faced += 1

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board

        # ---------------------------------------------------------
        # 0. SAFETY LOCKDOWN (Priority #1)
        # ---------------------------------------------------------
        if self.lockdown_mode:
            if CheckAction in legal:
                return CheckAction()
            return FoldAction()

        # ---------------------------------------------------------
        # 1. SMART DISCARD LOGIC
        # ---------------------------------------------------------
        if DiscardAction in legal:
            return self.get_smart_discard(my_cards, board)

        # ---------------------------------------------------------
        # 2. PRE-FLOP STRATEGY (Street 0)
        # ---------------------------------------------------------
        if street == 0:
            # Determine Mode (Tight vs Bully)
            use_loose_range = False
            if self.total_shoves >= self.SHOVE_SAMPLE_MIN:
                call_freq = self.total_calls_faced / self.total_shoves
                if call_freq < self.LOOSE_CALL_THRESHOLD:
                    use_loose_range = True

            # Evaluate Hand
            if self.is_good_preflop(my_cards, use_loose_range):
                if RaiseAction in legal:
                    min_raise, max_raise = round_state.raise_bounds()
                    self.did_shove_this_round = True
                    return RaiseAction(max_raise) # All-In
                
                # If facing a raise, only Call with TIGHT range (don't bluff-call)
                if CallAction in legal:
                    if self.is_good_preflop(my_cards, use_loose_range=False):
                        return CallAction()
                    return FoldAction()
                    
                if CheckAction in legal:
                    return CheckAction()
            else:
                # Trash Hand
                if CheckAction in legal:
                    return CheckAction()
                return FoldAction()

        # ---------------------------------------------------------
        # 3. POST-FLOP STRATEGY (Street > 0)
        # ---------------------------------------------------------
        # Check if we hit a TRUE MONSTER (Full House or Quads only)
        # We intentionally exclude Flushes as they are vulnerable.
        if self.is_monster_postflop(my_cards, board):
            if RaiseAction in legal:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(max_raise) # Trap sprung! All-In.
            if CallAction in legal:
                return CallAction()

        # Otherwise, play passive
        if CheckAction in legal:
            return CheckAction()
        return FoldAction()

    # =================================================================
    # HELPER FUNCTIONS
    # =================================================================

    def is_monster_postflop(self, my_cards, board):
        '''
        Returns True ONLY if we have Quads or Full House.
        Flushes are excluded because they are often beaten by higher flushes.
        '''
        all_cards = my_cards + board
        rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
        
        ranks = [rank_map[c[0]] for c in all_cards]
        
        # Check Quads and Full House
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
            
        has_trips = False
        has_pair = False
        
        for r, count in rank_counts.items():
            if count == 4:
                return True # Quads (Invincible)
            if count == 3:
                has_trips = True
            elif count >= 2:
                has_pair = True
        
        # Check for Full House
        # We need to be careful: 3, 3, 3, 2, 2 is FH.
        vals = sorted(rank_counts.values(), reverse=True)
        if len(vals) >= 2:
            if vals[0] >= 3 and vals[1] >= 2:
                return True # Full House (Very Strong)

        return False

    def is_good_preflop(self, cards, use_loose_range):
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
            # Loose: All pairs. Tight: 55+
            if use_loose_range: return True
            return pair_rank > 2

        if len(set(suits)) == 1:
            # Loose: All suited. Tight: Sum >= 25
            if use_loose_range: return True
            total_val = sum(r + 2 for r in ranks)
            return total_val >= 25

        return False

    def get_smart_discard(self, my_cards, board):
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
            
            danger_score = card_rank
            if flush_suit and card_suit == flush_suit:
                danger_score += 100
            
            if danger_score < lowest_danger_score:
                lowest_danger_score = danger_score
                best_discard_index = i
                
        return DiscardAction(best_discard_index)

if __name__ == '__main__':
    run_bot(Player(), parse_args())