'''
Adaptive ABC Pokerbot
Adapts to opponent aggression while playing solid fundamental poker.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from collections import Counter

class Player(Bot):
    '''
    A bot that tracks opponent aggression to switch between 
    Exploitative (vs Passive) and Solid (vs Aggressive) styles.
    '''

    def __init__(self):
        # Card Mapping
        self.rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
        
        # Game Constants
        self.STARTING_STACK = 400
        self.BIG_BLIND = 2
        
        # Opponent Tracking
        self.rounds_played = 0
        self.opp_raises_preflop = 0
        self.opp_pfr = 0.0  # Pre-Flop Raise Frequency
        
        # Strategy Toggles
        self.IS_AGGRESSIVE_OPP = False # Updates dynamically

    def handle_new_round(self, game_state, round_state, active):
        self.rounds_played += 1
        # Update PFR stats periodically
        if self.rounds_played > 0:
            self.opp_pfr = self.opp_raises_preflop / self.rounds_played
            # If opponent raises > 15% of the time, consider them Aggressive
            self.IS_AGGRESSIVE_OPP = self.opp_pfr > 0.15

    def handle_round_over(self, game_state, terminal_state, active):
        # In a real heavy implementation, we would parse terminal_state history
        # to find exactly if opponent raised. 
        # For this lightweight bot, we track raises in real-time inside get_action.
        pass

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        
        # Track Opponent Aggression (Simple Heuristic)
        # If we are facing a raise pre-flop that is > BB, opponent raised.
        if street == 0 and opp_pip > self.BIG_BLIND and my_pip <= self.BIG_BLIND:
            self.opp_raises_preflop += 0.5 # Increment slightly (approximate)

        # ---------------------------------------------------------
        # 1. DISCARD PHASE
        # ---------------------------------------------------------
        if DiscardAction in legal:
            return self.get_discard_action(my_cards)

        # ---------------------------------------------------------
        # 2. PRE-FLOP STRATEGY
        # ---------------------------------------------------------
        if street == 0:
            return self.play_preflop(round_state, active, my_cards, legal)

        # ---------------------------------------------------------
        # 3. POST-FLOP STRATEGY
        # ---------------------------------------------------------
        return self.play_postflop(round_state, active, my_cards, board, legal)

    # =========================================================================
    # STRATEGY ENGINES
    # =========================================================================

    def play_preflop(self, round_state, active, cards, legal):
        '''
        Dynamic Pre-flop logic based on Hand Tier and Opponent Type.
        '''
        tier = self.classify_preflop_hand(cards)
        opp_pip = round_state.pips[1-active]
        call_cost = opp_pip - round_state.pips[active]

        # --- TIER 1: PREMIUM (AA, KK, AKs, etc.) ---
        if tier == 1:
            # Vs Aggressive: Trap (Call/Check) to let them bluff
            if self.IS_AGGRESSIVE_OPP:
                if CallAction in legal: return CallAction()
                if CheckAction in legal: return CheckAction()
            # Vs Passive: Build the pot immediately
            return self.raise_value(round_state, 5 * self.BIG_BLIND)

        # --- TIER 2: STRONG (Pairs, AQ, KQs) ---
        if tier == 2:
            # Always decent to open-raise
            if call_cost == 0: 
                return self.raise_value(round_state, 3 * self.BIG_BLIND)
            
            # If facing a Raise (3-bet)
            if call_cost > 0:
                # If it's a huge shove (> 50 chips), maybe fold unless we have pair
                if call_cost > 50 and not self.has_pair(cards):
                    return FoldAction()
                return CallAction() if CallAction in legal else FoldAction()

        # --- TIER 3: SPECULATIVE (Suited Connectors, Small Pairs) ---
        if tier == 3:
            # Vs Passive: Steal blinds (Raise small)
            if not self.IS_AGGRESSIVE_OPP and call_cost == 0:
                return self.raise_value(round_state, 3 * self.BIG_BLIND)
            
            # Vs Aggressive or facing raise: Only Call if cheap (Implied Odds)
            # "Cheap" defined as < 5% of stack
            if call_cost <= 20: 
                if CallAction in legal: return CallAction()
                if CheckAction in legal: return CheckAction()
            
            return FoldAction()

        # --- TIER 4: TRASH ---
        if CheckAction in legal: return CheckAction()
        return FoldAction()

    def play_postflop(self, round_state, active, cards, board, legal):
        '''
        Evaluates hand strength vs board and bets accordingly.
        '''
        strength = self.evaluate_strength(cards, board)
        
        # Strength 0: Air
        # Strength 1: Bottom/Middle Pair
        # Strength 2: Top Pair
        # Strength 3: Two Pair / Trips
        # Strength 4+: Straight/Flush/Boat etc.

        opp_pip = round_state.pips[1-active]
        current_pot = (self.STARTING_STACK * 2) - sum(round_state.stacks)

        # --- MONSTER (Two Pair +) ---
        if strength >= 3:
            # Bet Big / Raise for Value
            return self.raise_pot(round_state, current_pot, opp_pip)
        
        # --- STRONG (Top Pair) ---
        if strength == 2:
            # If opponent is Aggro and betting into us, Call (Bluff Catch)
            if self.IS_AGGRESSIVE_OPP and opp_pip > round_state.pips[active]:
                if CallAction in legal: return CallAction()
            
            # If opponent is Passive or checked, Bet for Value
            return self.raise_value(round_state, round_state.pips[1-active] + (current_pot // 2))

        # --- MARGINAL (Middle/Bottom Pair) ---
        if strength == 1:
            # Check/Call small bets, Fold to big aggression
            if CheckAction in legal: return CheckAction()
            if CallAction in legal:
                cost = opp_pip - round_state.pips[active]
                # Call if cost is < 1/3 pot
                if cost < (current_pot / 3): return CallAction()
            return FoldAction()

        # --- AIR (Nothing) ---
        # Check / Fold
        if CheckAction in legal: return CheckAction()
        return FoldAction()


    # =========================================================================
    # HELPERS & EVALUATORS
    # =========================================================================

    def raise_value(self, round_state, amount):
        '''Safely returns a RaiseAction within bounds, or Call/Check.'''
        legal = round_state.legal_actions()
        if RaiseAction in legal:
            min_r, max_r = round_state.raise_bounds()
            # If target amount is too small, check min
            if amount < min_r: amount = min_r
            # If target is too big (or all in), cap it
            if amount > max_r: amount = max_r
            return RaiseAction(int(amount))
        
        if CallAction in legal: return CallAction()
        if CheckAction in legal: return CheckAction()
        return FoldAction()

    def raise_pot(self, round_state, current_pot, opp_pip):
        '''Calculates a pot-sized raise.'''
        raise_target = opp_pip + current_pot
        return self.raise_value(round_state, raise_target)

    def get_discard_action(self, cards):
        '''Brute force best 2-card combo.'''
        best_score = -1
        discard_idx = 0
        for i in range(len(cards)):
            kept = cards[:i] + cards[i+1:]
            score = self.rate_2card_hand(kept)
            if score > best_score:
                best_score = score
                discard_idx = i
        return DiscardAction(discard_idx)

    def rate_2card_hand(self, cards):
        '''Score a 2-card hand for discard decisions.'''
        r1, r2 = self.rank_map[cards[0][0]], self.rank_map[cards[1][0]]
        s1, s2 = cards[0][1], cards[1][1]
        score = max(r1, r2) + min(r1, r2)/10.0
        if r1 == r2: score += 20 # Pair
        if s1 == s2: score += 5  # Suited
        if abs(r1 - r2) == 1: score += 3 # Connector
        if abs(r1 - r2) == 2: score += 1 # Gapper
        return score

    def classify_preflop_hand(self, cards):
        '''Returns Tier: 1 (Premium), 2 (Strong), 3 (Spec), 4 (Trash)'''
        ranks = sorted([self.rank_map[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        counts = Counter(ranks)
        
        # Check Pairs/Trips
        is_pair = any(c >= 2 for c in counts.values())
        is_trips = any(c >= 3 for c in counts.values())
        top_pair_val = max((r for r, c in counts.items() if c >= 2), default=-1)

        # TIER 1: Premium
        # Trips, High Pairs (TT+), AK/AQ suited
        if is_trips: return 1
        if is_pair and top_pair_val >= 8: return 1 # TT+
        # AK/AQ check
        if ranks[0] == 12 and ranks[1] >= 10: return 1 # AK, AQ, AJ

        # TIER 2: Strong
        # Mid Pairs (77-99), High Broadways
        if is_pair and top_pair_val >= 5: return 2 # 77+
        if ranks[0] >= 10 and ranks[1] >= 10: return 2 # Two broadways
        if ranks[0] == 12: return 2 # Any Ace

        # TIER 3: Speculative
        # Small Pairs, Suited Connectors
        if is_pair: return 3 # 22-66
        # Suited
        suit_counts = Counter(suits)
        if any(c >= 2 for c in suit_counts.values()): return 3
        # Connectors (gap <= 2)
        if (ranks[0] - ranks[1] <= 2) and (ranks[1] - ranks[2] <= 2): return 3
        
        return 4

    def has_pair(self, cards):
        ranks = [self.rank_map[c[0]] for c in cards]
        return len(set(ranks)) < len(ranks)

    def evaluate_strength(self, my_cards, board):
        '''
        Returns rough integer strength:
        0=Air, 1=WeakPair, 2=TopPair, 3=TwoPair/Trips, 4=Str/Flush+
        '''
        my_ranks = [self.rank_map[c[0]] for c in my_cards]
        board_ranks = [self.rank_map[c[0]] for c in board]
        my_suits = [c[1] for c in my_cards]
        board_suits = [c[1] for c in board]
        
        all_ranks = my_ranks + board_ranks
        all_suits = my_suits + board_suits
        
        rank_counts = Counter(all_ranks)
        suit_counts = Counter(all_suits)
        
        # 4. Check Flush (>= 5 same suit)
        if any(c >= 5 for c in suit_counts.values()):
            return 4
            
        # 4. Check Straight (5 consecutive)
        uniq_ranks = sorted(list(set(all_ranks)))
        consecutive = 0
        for i in range(len(uniq_ranks)-1):
            if uniq_ranks[i+1] == uniq_ranks[i] + 1:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 4: return 4
            
        # Check Pairs/Trips logic
        my_pair_ranks = [r for r, c in rank_counts.items() if c >= 2]
        my_trip_ranks = [r for r, c in rank_counts.items() if c >= 3]
        
        # 3. Trips or Two Pair
        if my_trip_ranks: return 3
        if len(my_pair_ranks) >= 2: return 3
        
        # Check if we have a pair using a hole card
        # (Exclude pairs that are purely on the board)
        matches = []
        for r in my_ranks:
            if r in board_ranks:
                matches.append(r)
        
        # Pocket pair
        if my_ranks[0] == my_ranks[1]:
            matches.append(my_ranks[0])
            
        if not matches: return 0
        
        # 2. Top Pair (Our pair matches highest board card or is overpair)
        max_match = max(matches)
        max_board = max(board_ranks) if board_ranks else 0
        
        if max_match >= max_board: return 2
        
        # 1. Weak Pair
        return 1

if __name__ == '__main__':
    run_bot(Player(), parse_args())