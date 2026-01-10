"""
player.py — Titan v3.1 (Fixed + Stronger)

Fixes:
- NEVER misformats discard: if DiscardAction is legal, we ALWAYS return DiscardAction(...) first.
- Robust legal-action checks (works whether legal_actions returns classes or instances).
- Postflop raise sizing fixed to be consistently "raise-to".
- Equity sim improved: opponent keeps best 2 *board-aware* (by evaluating keep2 + board).
- Equity caching actually used.
- Hand evaluator upgraded (accurate 7-card best-5 via 21 combos) — slower than the old hack,
  but much less wrong. Iterations are adjusted + cached to keep runtime reasonable.
"""

import random
from collections import defaultdict, Counter
from itertools import combinations

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState, NUM_ROUNDS, STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

# Try to import pkrbot for C++ speed. If not available, use Python evaluator.
try:
    import pkrbot  # noqa: F401
    PKRBOT_AVAILABLE = True
except ImportError:
    PKRBOT_AVAILABLE = False


RANK_TO_INT = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
               'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


class Player(Bot):
    def __init__(self):
        self.lockdown_mode = False
        self.rng = random.Random()
        self.eq_cache = {}

    # ----------------------------
    # Engine hooks
    # ----------------------------
    def handle_new_round(self, game_state, round_state, active):
        # Periodically clear cache to avoid uncontrolled growth
        if game_state.round_num % 10 == 0:
            self.eq_cache = {}

        # SAFETY LOCKDOWN: if we're far enough ahead, we play check/fold (but ONLY when not discarding)
        rounds_remaining = NUM_ROUNDS - game_state.round_num + 1
        secure_threshold = (rounds_remaining * 1.5) + 10.0
        self.lockdown_mode = (game_state.bankroll > secure_threshold)

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    # ----------------------------
    # Core decision
    # ----------------------------
    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board

        # 0) DISCARD OVERRIDES EVERYTHING (THIS FIXES YOUR DISCONNECT / MISFORMAT)
        if self.has_action(legal, DiscardAction):
            return self.get_best_discard(my_cards, board)

        # 1) LOCKDOWN (only applies when NOT discarding)
        if self.lockdown_mode:
            if self.has_action(legal, CheckAction):
                return CheckAction()
            if self.has_action(legal, CallAction):
                return CallAction()
            return FoldAction()

        # 2) PRE-FLOP
        if street == 0:
            points = self.evaluate_preflop_points(my_cards)

            if points >= 35:  # Strong
                if self.has_action(legal, RaiseAction):
                    # Random trap (10%)
                    if self.rng.random() < 0.10 and self.has_action(legal, CallAction):
                        return CallAction()

                    min_r, max_r = round_state.raise_bounds()

                    # Monster -> jam
                    if points > 45:
                        return RaiseAction(max_r)

                    # Pot-sized-ish raise-to
                    pot = self.compute_pot(round_state)
                    opp_pip = round_state.pips[1 - active]
                    target_raise_to = opp_pip + pot  # "raise-to"
                    amt = max(min_r, min(max_r, target_raise_to))
                    return RaiseAction(amt)

                if self.has_action(legal, CallAction):
                    return CallAction()
                if self.has_action(legal, CheckAction):
                    return CheckAction()
                return FoldAction()

            elif points >= 26:  # Playable
                if self.has_action(legal, CallAction):
                    return CallAction()
                if self.has_action(legal, CheckAction):
                    return CheckAction()
                return FoldAction()

            else:  # Trash
                if self.has_action(legal, CheckAction):
                    return CheckAction()
                return FoldAction()

        # 3) POST-FLOP+
        pot = self.compute_pot(round_state)

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        cost_to_call = max(0, opp_pip - my_pip)

        is_wet = self.is_board_wet(board)

        # Dynamic iterations (cached, so this is safe)
        # - deeper board = more valuable accuracy
        # - larger pot = more valuable accuracy
        base_iters = 50
        if pot >= 80:
            base_iters = 70
        if pot >= 200:
            base_iters = 90
        if len(board) >= 4:
            base_iters = max(base_iters, 80)
        if len(board) >= 5:
            base_iters = max(base_iters, 110)

        equity = self.calculate_equity(my_cards, board, street, iterations=base_iters)

        # A) VALUE / PROTECTION when strong
        if equity > 0.60 and self.has_action(legal, RaiseAction):
            min_r, max_r = round_state.raise_bounds()

            # Desired bet size (as "additional chips" to put in)
            if is_wet:
                bet_size = pot  # pot-ish
            else:
                bet_size = int(pot * 0.55)  # smaller on dry boards

            # Monster + wet -> jam
            if equity > 0.80 and is_wet:
                return RaiseAction(max_r)

            # Consistent "raise-to" sizing:
            # call_to is the amount we must match right now (highest pip)
            call_to = max(my_pip, opp_pip)
            raise_to = call_to + max(1, bet_size)

            amt = max(min_r, min(max_r, raise_to))
            return RaiseAction(amt)

        # If we can't raise but we’re strong and can call/check, do it
        if equity > 0.60:
            if cost_to_call > 0 and self.has_action(legal, CallAction):
                return CallAction()
            if self.has_action(legal, CheckAction):
                return CheckAction()

        # B) POT-ODDS CALLING
        if cost_to_call > 0:
            # pot already includes opponent's current bet (stack deltas include pips),
            # total after we call adds cost_to_call
            pot_total_after_call = pot + cost_to_call
            required_equity = cost_to_call / max(1, pot_total_after_call)

            # "Titan margin" to avoid razor-thin spots
            if equity >= required_equity + 0.05:
                if self.has_action(legal, CallAction):
                    return CallAction()

        # C) FREE CHECK
        if self.has_action(legal, CheckAction):
            return CheckAction()

        # D) FOLD
        return FoldAction()

    # ============================================================
    # Helper: legal action checks (robust)
    # ============================================================
    def has_action(self, legal_actions, action_cls):
        # Works whether legal_actions contains classes, instances, or mixed
        for a in legal_actions:
            if a == action_cls:
                return True
            try:
                if isinstance(a, action_cls):
                    return True
            except TypeError:
                # If a isn't a type / isn't suitable for isinstance
                pass
        return False

    # ============================================================
    # Game logic helpers
    # ============================================================
    def compute_pot(self, round_state):
        # Stack delta pot: total chips committed this hand by both players
        return (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])

    def is_board_wet(self, board):
        """
        "Wet" = draw-heavy.
        Criteria:
        - 3+ of same suit on board OR
        - 3 ranks within a span of 4 (rough connectedness)
        """
        if len(board) < 3:
            return False

        suits = [c[1] for c in board]
        s_counts = defaultdict(int)
        for s in suits:
            s_counts[s] += 1
        if any(v >= 3 for v in s_counts.values()):
            return True

        ranks = sorted([RANK_TO_INT[c[0]] for c in board])
        for i in range(len(ranks) - 2):
            if ranks[i + 2] - ranks[i] <= 4:
                return True

        return False

    def evaluate_preflop_points(self, cards):
        ranks = sorted([RANK_TO_INT[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        points = sum(ranks)

        # Pairs / trips
        if ranks[0] == ranks[1] or ranks[1] == ranks[2] or ranks[0] == ranks[2]:
            points += 20
            if ranks[0] == ranks[2]:
                points += 30

        # Suitedness
        if suits[0] == suits[1] == suits[2]:
            points += 12
        elif suits[0] == suits[1] or suits[1] == suits[2] or suits[0] == suits[2]:
            points += 3

        # Connectivity (in rank-sorted order)
        gap1 = ranks[0] - ranks[1]
        gap2 = ranks[1] - ranks[2]
        if gap1 == 1 and gap2 == 1:
            points += 12
        elif gap1 == 1 or gap2 == 1:
            points += 4

        return points

    # ============================================================
    # Discard logic (discarded card becomes a public board card in this variant)
    # ============================================================
    def get_best_discard(self, my_cards, board):
        # MUST return DiscardAction(idx) 0..2
        best_idx = 0
        best_eq = -1.0

        # Faster discard sim; cache will help a lot
        iters = 60 if len(board) >= 2 else 50

        for i in range(3):
            kept = [my_cards[j] for j in range(3) if j != i]
            board_after_discard = list(board) + [my_cards[i]]  # discarded card becomes board
            eq = self.calculate_equity(kept, board_after_discard, street=1, iterations=iters)
            if eq > best_eq:
                best_eq = eq
                best_idx = i

        return DiscardAction(best_idx)

    # ============================================================
    # Equity simulation (Monte Carlo)
    # ============================================================
    def calculate_equity(self, my_cards, board, street, iterations=60):
        # Cache by (hand, board, street, iters_bucket) — bucket iters so cache hit rate stays high
        it_bucket = 40 if iterations <= 45 else 60 if iterations <= 70 else 90 if iterations <= 100 else 120
        key = (tuple(sorted(my_cards)), tuple(board), street, it_bucket)

        cached = self.eq_cache.get(key)
        if cached is not None:
            return cached

        eq = self.calculate_equity_python(my_cards, board, iterations=it_bucket)
        self.eq_cache[key] = eq
        return eq

    def calculate_equity_python(self, my_cards, board, iterations):
        """
        Sim rules approximation:
        - Opponent is dealt 3; they keep best 2 in a board-aware way (maximize eval(keep2+board)).
        - If our my_cards has length 3 (rare here), we also keep best 2 in the same way.
        - Board runs out to 6 public cards total.
        """
        wins = 0.0
        full_deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
        used = set(my_cards + list(board))
        deck = [c for c in full_deck if c not in used]

        def best_two_from_three(cards3, current_board):
            best_score = None
            best_keep = None
            for drop_i in range(3):
                keep = [cards3[j] for j in range(3) if j != drop_i]
                score = self.eval7(keep + current_board)
                if best_score is None or score > best_score:
                    best_score = score
                    best_keep = keep
            return best_keep

        for _ in range(iterations):
            self.rng.shuffle(deck)
            d = 0

            # Opponent 3 cards
            opp3 = [deck[d], deck[d + 1], deck[d + 2]]
            d += 3

            sim_board = list(board)

            # Choose our 2 (if we still have 3 in some call path)
            if len(my_cards) == 3:
                my2 = best_two_from_three(my_cards, sim_board)
            else:
                my2 = list(my_cards)

            # Opp chooses 2 board-aware
            opp2 = best_two_from_three(opp3, sim_board)

            # Runout to 6 board cards
            while len(sim_board) < 6:
                sim_board.append(deck[d])
                d += 1

            s1 = self.eval7(my2 + sim_board)
            s2 = self.eval7(opp2 + sim_board)

            if s1 > s2:
                wins += 1.0
            elif s1 == s2:
                wins += 0.5

        return wins / float(iterations)

    # ============================================================
    # Hand evaluation (accurate, Python)
    # ============================================================
    def eval7(self, cards):
        """
        Returns a comparable tuple representing best 5-card hand from up to 7 cards.
        Higher tuple => stronger hand.
        """
        # If pkrbot exists and you know its API, wire it here safely.
        # Right now we keep pure Python to avoid API mismatch crashes.

        # Enumerate all 5-card combos (21 combos for 7 cards; fewer if 6/5 cards)
        best = None
        n = len(cards)
        if n <= 5:
            return self.eval5(cards)

        for idxs in combinations(range(n), 5):
            hand = [cards[i] for i in idxs]
            score = self.eval5(hand)
            if best is None or score > best:
                best = score
        return best

    def eval5(self, cards5):
        """
        5-card evaluator with proper kickers.
        Categories:
        8: straight flush
        7: four of a kind
        6: full house
        5: flush
        4: straight
        3: three of a kind
        2: two pair
        1: one pair
        0: high card
        """
        ranks = sorted([RANK_TO_INT[c[0]] for c in cards5], reverse=True)
        suits = [c[1] for c in cards5]

        rank_counts = Counter(ranks)
        counts_sorted = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

        is_flush = (len(set(suits)) == 1)

        # Straight detection (handle wheel A-5)
        uniq = sorted(set(ranks), reverse=True)
        is_straight = False
        straight_high = 0
        if len(uniq) == 5:
            if uniq[0] - uniq[4] == 4:
                is_straight = True
                straight_high = uniq[0]
            elif uniq == [14, 5, 4, 3, 2]:
                is_straight = True
                straight_high = 5

        if is_flush and is_straight:
            return (8, straight_high)

        # Four / Full house / Trips / Pairs
        c1_rank, c1_cnt = counts_sorted[0]
        c2_rank, c2_cnt = counts_sorted[1] if len(counts_sorted) > 1 else (0, 0)

        if c1_cnt == 4:
            kicker = max(r for r in ranks if r != c1_rank)
            return (7, c1_rank, kicker)

        if c1_cnt == 3 and c2_cnt == 2:
            return (6, c1_rank, c2_rank)

        if is_flush:
            # Flush breaks ties by all five ranks
            return (5, tuple(sorted(ranks, reverse=True)))

        if is_straight:
            return (4, straight_high)

        if c1_cnt == 3:
            kickers = sorted([r for r in ranks if r != c1_rank], reverse=True)
            return (3, c1_rank, tuple(kickers))

        if c1_cnt == 2 and c2_cnt == 2:
            high_pair = max(c1_rank, c2_rank)
            low_pair = min(c1_rank, c2_rank)
            kicker = max(r for r in ranks if r != high_pair and r != low_pair)
            return (2, high_pair, low_pair, kicker)

        if c1_cnt == 2:
            pair_rank = c1_rank
            kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
            return (1, pair_rank, tuple(kickers))

        return (0, tuple(sorted(ranks, reverse=True)))


if __name__ == '__main__':
    run_bot(Player(), parse_args())
