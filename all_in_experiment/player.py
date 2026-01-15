'''
The Solver Bot (Fixed & Optimized)
- Starts Conservative (Assumes opponent calls 50% of time)
- Fixes "Coinflip Fallacy" by defaulting to low equity on unknown ranges
- Increased simulation precision
'''
import random
import math
from collections import defaultdict, deque
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState, NUM_ROUNDS, STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

class Player(Bot):
    def __init__(self):
        self.lockdown_mode = False
        self.rng = random.Random()
        
        # --- STATISTICS TRACKING ---
        self.shove_history = deque(maxlen=20) 
        
        # FIXED: Start assuming opponent calls 50% of the time.
        # This makes US conservative (we won't bluff if we think they call).
        self.default_call_freq = 0.50 
        
        self.did_shove_this_round = False

    def handle_new_round(self, game_state, round_state, active):
        self.did_shove_this_round = False
        
        # SAFETY LOCKDOWN
        rounds_remaining = NUM_ROUNDS - game_state.round_num + 1
        secure_threshold = (rounds_remaining * 1.5) + 10.0
        if game_state.bankroll > secure_threshold:
            self.lockdown_mode = True
        else:
            self.lockdown_mode = False

    def handle_round_over(self, game_state, terminal_state, active):
        if self.did_shove_this_round:
            my_delta = terminal_state.deltas[active]
            if abs(my_delta) > 100:
                self.shove_history.append(1) # Called
            else:
                self.shove_history.append(0) # Folded

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board
        
        if self.lockdown_mode:
            return CheckAction() if CheckAction in legal else FoldAction()

        if DiscardAction in legal:
            return self.get_best_discard(my_cards, board)
        if street in (2, 3) and CheckAction in legal:
            return CheckAction()

        # -----------------------------------------------------------------
        # PRE-FLOP SOLVER
        # -----------------------------------------------------------------
        if street == 0:
            if len(self.shove_history) > 0:
                call_freq = sum(self.shove_history) / len(self.shove_history)
            else:
                call_freq = self.default_call_freq
            
            # Clamp: Assume at least 5% calls, max 95%
            call_freq = max(0.05, min(0.95, call_freq))
            
            # Solve EV
            ev_shove = self.solve_shove_ev(my_cards, call_freq)
            
            if RaiseAction in legal:
                min_r, max_r = round_state.raise_bounds()
                
                # REQUIREMENT: EV must be significant to shove (> 5 chips)
                # This prevents jamming for +0.1 EV edge
                if ev_shove > 5.0:
                    self.did_shove_this_round = True
                    return RaiseAction(max_r)
            
            # Passive Play Logic
            points = self.evaluate_preflop_strength(my_cards)
            
            # Call if decent hand (26+ points)
            if points >= 26: 
                if CallAction in legal: return CallAction()
                if CheckAction in legal: return CheckAction()
            
            if CheckAction in legal: return CheckAction()
            return FoldAction()

        # -----------------------------------------------------------------
        # POST-FLOP
        # -----------------------------------------------------------------
        equity = self.calculate_equity_postflop(my_cards, board)
        
        if equity > 0.60:
            if RaiseAction in legal:
                min_r, max_r = round_state.raise_bounds()
                pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])
                target = pot 
                return RaiseAction(max(min_r, min(max_r, round_state.pips[active] + target)))
            if CallAction in legal: return CallAction()
        
        elif equity > 0.40:
            if CheckAction in legal: return CheckAction()
            if CallAction in legal: return CallAction()
            
        if CheckAction in legal: return CheckAction()
        return FoldAction()

    # =================================================================
    # THE MATH ENGINE
    # =================================================================

    def solve_shove_ev(self, my_cards, call_freq):
        iterations = 400  # Increased for accuracy
        wins = 0
        samples_count = 0
        
        pot_dead = 3
        stack_risk = 400
        total_pot_if_called = 800
        
        full_deck = [r+s for r in "23456789TJQKA" for s in "cdhs"]
        used = set(my_cards)
        deck = [c for c in full_deck if c not in used]
        
        for _ in range(iterations):
            self.rng.shuffle(deck)
            opp_3 = [deck[0], deck[1], deck[2]]
            
            # Rate Opponent Hand Strength
            opp_strength = self.evaluate_preflop_strength(opp_3)
            
            # Linear map for threshold
            # High Call Freq (Loose) -> Low Threshold
            # Low Call Freq (Tight) -> High Threshold
            threshold = 70 * (1.0 - call_freq)
            
            if opp_strength < threshold:
                continue # They fold
            
            samples_count += 1
            
            # Runout
            sim_board = deck[3:9]
            my_2 = self.fast_keep_best(my_cards)
            opp_2 = self.fast_keep_best(opp_3)
            
            s1 = self.fast_eval(my_2 + sim_board)
            s2 = self.fast_eval(opp_2 + sim_board)
            if s1 > s2: wins += 1
            elif s1 == s2: wins += 0.5
            
        # FIXED: If we have 0 samples (opponent range is too tight for our sample size),
        # assume we have terrible equity (20%), NOT a coinflip (50%).
        if samples_count == 0:
            equity_when_called = 0.20 
        else:
            equity_when_called = wins / samples_count
            
        p_fold = 1.0 - call_freq
        p_call = call_freq
        
        ev_fold_part = p_fold * pot_dead
        ev_call_part = p_call * ((equity_when_called * total_pot_if_called) - stack_risk)
        
        return ev_fold_part + ev_call_part

    # =================================================================
    # UTILITIES
    # =================================================================

    def evaluate_preflop_strength(self, cards):
        rank_map = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
        ranks = sorted([rank_map[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        points = sum(ranks)
        if ranks[0] == ranks[1] or ranks[1] == ranks[2] or ranks[0] == ranks[2]:
            points += 20
            if ranks[0] == ranks[2]: points += 30
        if suits[0] == suits[1] == suits[2]: points += 15 
        elif suits[0] == suits[1] or suits[1] == suits[2] or suits[0] == suits[2]: points += 5
        gap1 = ranks[0] - ranks[1]
        gap2 = ranks[1] - ranks[2]
        if gap1 == 1 and gap2 == 1: points += 15
        elif (gap1 == 1 or gap2 == 1): points += 5
        return points

    def fast_keep_best(self, cards3):
        best_score = -1
        best_pair = cards3[:2]
        rank_map = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
        for i in range(3):
            keep = [cards3[j] for j in range(3) if j != i]
            r1 = rank_map[keep[0][0]]
            r2 = rank_map[keep[1][0]]
            score = r1 + r2
            if r1 == r2: score += 100
            if keep[0][1] == keep[1][1]: score += 20
            if score > best_score:
                best_score = score
                best_pair = keep
        return best_pair

    def get_best_discard(self, my_cards, board):
        rank_map = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
        scores = []
        for i in range(3):
            keep = [my_cards[j] for j in range(3) if j != i]
            r1 = rank_map[keep[0][0]]
            r2 = rank_map[keep[1][0]]
            s = r1 + r2
            if r1 == r2: s += 50
            if keep[0][1] == keep[1][1]: s += 10
            scores.append(s)
        best_idx = scores.index(max(scores))
        return DiscardAction(best_idx)

    def calculate_equity_postflop(self, my_cards, board):
        wins = 0
        iters = 50
        full_deck = [r+s for r in "23456789TJQKA" for s in "cdhs"]
        used = set(my_cards + board)
        deck = [c for c in full_deck if c not in used]
        for _ in range(iters):
            self.rng.shuffle(deck)
            opp = [deck[0], deck[1]]
            sim_board = list(board)
            idx = 2
            while len(sim_board) < 6:
                sim_board.append(deck[idx])
                idx += 1
            s1 = self.fast_eval(my_cards + sim_board)
            s2 = self.fast_eval(opp + sim_board)
            if s1 > s2: wins += 1
            elif s1 == s2: wins += 0.5
        return wins/iters

    def fast_eval(self, cards):
        rank_map = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
        ranks = sorted([rank_map[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        is_flush = False
        sc = defaultdict(int)
        for s in suits: sc[s]+=1
        if any(v>=5 for v in sc.values()): is_flush = True
        uniq = sorted(list(set(ranks)), reverse=True)
        is_str = False
        for i in range(len(uniq)-4):
            if uniq[i]-uniq[i+4]==4: is_str=True; break
        if is_flush and is_str: return 800
        rc = defaultdict(int)
        for r in ranks: rc[r]+=1
        counts = sorted(rc.values(), reverse=True)
        if counts[0]==4: return 700
        if counts[0]==3 and counts[1]>=2: return 600
        if is_flush: return 500
        if is_str: return 400
        if counts[0]==3: return 300
        if counts[0]==2 and counts[1]>=2: return 200
        if counts[0]==2: return 100
        return sum(ranks[:5])

if __name__ == '__main__':
    run_bot(Player(), parse_args())