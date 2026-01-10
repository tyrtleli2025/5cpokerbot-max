'''
The Professional v2 - Aggressive & Sticky
Fixes: Illegal Calls, Limping, Over-Folding Overpairs
'''
import random
from collections import namedtuple, Counter
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

# ==============================================================================
# 1. POKER ENGINE & CONSTANTS
# ==============================================================================

RANKS = '23456789TJQKA'
SUITS = 'shdc'
RANK_MAP = {r: i for i, r in enumerate(RANKS)}
SUIT_MAP = {s: i for i, s in enumerate(SUITS)}

# Hand Strength Constants
HAND_HIGH_CARD = 0
HAND_PAIR = 1
HAND_TWO_PAIR = 2
HAND_TRIPS = 3
HAND_STRAIGHT = 4
HAND_FLUSH = 5
HAND_FULL_HOUSE = 6
HAND_QUADS = 7
HAND_STRAIGHT_FLUSH = 8

class Card:
    def __init__(self, rank_char, suit_char):
        self.rank_char = rank_char
        self.suit_char = suit_char
        self.rank = RANK_MAP[rank_char]
        self.suit = SUIT_MAP[suit_char]

    def __repr__(self):
        return f"{self.rank_char}{self.suit_char}"

def parse_cards(card_strs):
    return [Card(s[0], s[1]) for s in card_strs]

class HandEvaluator:
    @staticmethod
    def evaluate(cards):
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        # Flush
        is_flush = False
        flush_ranks = []
        for s, count in suit_counts.items():
            if count >= 5:
                is_flush = True
                flush_ranks = sorted([c.rank for c in cards if c.suit == s], reverse=True)[:5]
                break

        # Straight
        unique_ranks = sorted(list(set(ranks)), reverse=True)
        straight_high_rank = -1
        is_straight = False
        for i in range(len(unique_ranks) - 4):
            window = unique_ranks[i:i+5]
            if window[0] - window[4] == 4:
                is_straight = True
                straight_high_rank = window[0]
                break
        
        # Wheel Straight (A-5)
        if not is_straight and 12 in unique_ranks:
            wheel = [3, 2, 1, 0]
            if all(r in unique_ranks for r in wheel):
                is_straight = True
                straight_high_rank = 3

        if is_flush: return (HAND_FLUSH, flush_ranks)
        if is_straight: return (HAND_STRAIGHT, [straight_high_rank])

        counts = rank_counts.most_common() 
        
        if counts[0][1] == 4:
            return (HAND_QUADS, [counts[0][0], counts[1][0]])
        if counts[0][1] == 3 and counts[1][1] >= 2:
            return (HAND_FULL_HOUSE, [counts[0][0], counts[1][0]])
        if counts[0][1] == 3:
            kickers = [r for r in ranks if r != counts[0][0]][:2]
            return (HAND_TRIPS, [counts[0][0]] + kickers)
        if counts[0][1] == 2 and counts[1][1] == 2:
            kickers = [r for r in ranks if r != counts[0][0] and r != counts[1][0]][:1]
            return (HAND_TWO_PAIR, [counts[0][0], counts[1][0]] + kickers)
        if counts[0][1] == 2:
            kickers = [r for r in ranks if r != counts[0][0]][:3]
            return (HAND_PAIR, [counts[0][0]] + kickers)
            
        return (HAND_HIGH_CARD, ranks[:5])

# ==============================================================================
# 2. BOT LOGIC
# ==============================================================================

class Player(Bot):
    def __init__(self):
        self.STACK_SIZE = 400
        self.BB = 2

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street 
        my_cards = parse_cards(round_state.hands[active])
        board_cards = parse_cards(round_state.board)
        
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        call_cost = opp_pip - my_pip
        pot = (self.STACK_SIZE*2) - sum(round_state.stacks)
        
        # 1. DISCARD
        if DiscardAction in legal:
            return self.get_discard_strategy(my_cards)

        # 2. PRE-FLOP
        if street == 0:
            return self.preflop_strategy(my_cards, legal, call_cost, round_state, opp_pip, active)

        # 3. POST-FLOP
        return self.postflop_strategy(my_cards, board_cards, legal, call_cost, pot, round_state, opp_pip)

    # --- ACTION HELPER (Fixes Illegal Call Bug) ---
    def safe_call_or_check(self, legal):
        if CallAction in legal:
            return CallAction()
        return CheckAction()

    # --- PRE-FLOP STRATEGY ---
    def preflop_strategy(self, cards, legal, call_cost, round_state, opp_pip, active):
        score = self.chen_score_3card(cards)
        is_sb = (round_state.button % 2 == active) # Am I the dealer/SB?
        
        # STRATEGY 1: NO LIMPING (SB)
        # If we are SB and haven't put money in yet (call_cost == 1 to call BB), 
        # we Raise or Fold.
        if is_sb and call_cost <= 1:
            if score >= 7: # Decent hand
                # Raise to 3BB (6)
                return self.make_raise(round_state, 6)
            # Fold trash logic (unless score is barely playable, but "No Limp" means Fold)
            return FoldAction()

        # STRATEGY 2: DEFEND BLINDS (BB vs Raise)
        # If opponent raised, we widen our calling range.
        if not is_sb and call_cost > 0:
            # Always call with any Pair
            if self.has_pair(cards): return self.safe_call_or_check(legal)
            # Always call with any Ace
            if self.has_high_card(cards, 12): return self.safe_call_or_check(legal)
            # Always call with Suited cards
            if self.is_suited(cards): return self.safe_call_or_check(legal)
            
            # Fallback to Chen score for unsuited connectors etc.
            if score >= 6: return self.safe_call_or_check(legal)
            
            return FoldAction()

        # GENERAL PLAY (Standard Raises/Calls)
        if score >= 9:
            target = opp_pip + call_cost + self.BB*2 # Raise logic
            return self.make_raise(round_state, target)
        
        if score >= 6:
            return self.safe_call_or_check(legal)

        if CheckAction in legal: return CheckAction()
        return FoldAction()

    # --- DISCARD STRATEGY ---
    def get_discard_strategy(self, my_cards):
        # Keep best 2-card Chen score
        best_score = -1
        best_keep_indices = [0, 1]
        
        indices = [(0, 1), (0, 2), (1, 2)]
        for idxs in indices:
            keep = [my_cards[idxs[0]], my_cards[idxs[1]]]
            score = self.chen_score_2card(keep)
            if score > best_score:
                best_score = score
                best_keep_indices = idxs
        
        for i in range(3):
            if i not in best_keep_indices:
                return DiscardAction(i)
        return DiscardAction(0)

    # --- POST-FLOP STRATEGY ---
    def postflop_strategy(self, my_cards, board, legal, call_cost, pot, round_state, opp_pip):
        full_hand = my_cards + board
        rank_class, tie_breakers = HandEvaluator.evaluate(full_hand)
        
        # Detect Overpair
        # Pair rank > Max board rank
        is_overpair = False
        if rank_class == HAND_PAIR:
            pair_rank = tie_breakers[0]
            board_ranks = [c.rank for c in board]
            max_board = max(board_ranks) if board_ranks else -1
            # Ensure the pair is in our hole cards (at least one)
            # Actually, standard Overpair definition: Pocket pair > Board.
            # Here we just check if our best pair is better than board high.
            if pair_rank > max_board:
                is_overpair = True

        # 1. MONSTER (Two Pair+)
        if rank_class >= HAND_TWO_PAIR:
            # Bet Pot
            return self.make_raise(round_state, opp_pip + pot)

        # 2. STRONG (Top Pair or Overpair)
        # STRATEGY 3: VALUE OVERPAIRS (Don't fold them!)
        if rank_class == HAND_PAIR:
            pair_rank = tie_breakers[0]
            board_ranks = [c.rank for c in board]
            max_board = max(board_ranks) if board_ranks else -1
            is_top_pair = (pair_rank >= max_board)

            if is_top_pair or is_overpair:
                if call_cost > 0:
                    # CALL bets (sticky). Don't fold Top/Overpair.
                    return self.safe_call_or_check(legal)
                else:
                    # Bet for value (1/2 pot)
                    return self.make_raise(round_state, opp_pip + (pot // 2))

        # 3. WEAK (Middle/Bottom Pair, Air)
        if CheckAction in legal: return CheckAction()
        return FoldAction()

    # --- HELPERS ---
    def make_raise(self, round_state, amount):
        legal = round_state.legal_actions()
        if RaiseAction in legal:
            min_r, max_r = round_state.raise_bounds()
            actual = min(max(min_r, amount), max_r)
            return RaiseAction(actual)
        return self.safe_call_or_check(legal)

    def has_pair(self, cards):
        ranks = [c.rank for c in cards]
        return len(set(ranks)) < len(ranks)

    def has_high_card(self, cards, threshold=12): # 12=Ace
        return any(c.rank >= threshold for c in cards)

    def is_suited(self, cards):
        suits = [c.suit for c in cards]
        return any(suits.count(s) >= 2 for s in suits)

    def chen_score_2card(self, cards):
        ranks = sorted([c.rank for c in cards], reverse=True)
        high_rank = ranks[0]
        score = 0
        if high_rank == 12: score = 10 
        elif high_rank == 11: score = 8 
        elif high_rank == 10: score = 7 
        elif high_rank == 9: score = 6 
        else: score = (high_rank + 2) / 2.0
        
        if ranks[0] == ranks[1]:
            score *= 2
            if score < 5: score = 5
        if cards[0].suit == cards[1].suit: score += 2
        gap = ranks[0] - ranks[1]
        if gap == 1: score += 1 
        elif gap == 2: score -= 1
        elif gap == 3: score -= 2
        elif gap >= 4: score -= 4
        return score

    def chen_score_3card(self, cards):
        best_2 = 0
        pairs = [(cards[0], cards[1]), (cards[0], cards[2]), (cards[1], cards[2])]
        for p in pairs:
            s = self.chen_score_2card(p)
            if s > best_2: best_2 = s
        return best_2

if __name__ == '__main__':
    run_bot(Player(), parse_args())