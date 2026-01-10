'''
Online Regret-Matching+ with Monte Carlo rollouts (discard-aware).

This is "CFR-ish" rather than true MCCFR over a full game tree, because the engine
doesn't give us an easy way to compute exact counterfactual values at every node.
Instead, we estimate action values using:
- current pot / cost
- showdown equity from Monte Carlo rollouts that simulate the discard mechanic

Then we update regrets toward higher-EV actions (Regret Matching+).
We also accumulate average strategy (strategy_sum) for stability.

Action abstraction:
0 = Fold (or Check if free)
1 = Check/Call
2 = Pot raise
3 = All-in
'''
from __future__ import annotations

import random
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState, STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

# ----------------------------
# Constants / Card utilities
# ----------------------------
RANKS = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS, start=2)}
SUITS = "cdhs"

ACT_FOLD = 0
ACT_CHECK_CALL = 1
ACT_RAISE_POT = 2
ACT_ALL_IN = 3
NUM_ACTS = 4

def parse_card(card: str) -> Tuple[int, str]:
    return (RANK_TO_INT[card[0]], card[1])

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def is_straight(ranks: List[int]) -> Tuple[bool, int]:
    uniq = sorted(set(ranks))
    if len(uniq) != 5:
        return (False, 0)
    # wheel A2345
    if uniq == [2, 3, 4, 5, 14]:
        return (True, 5)
    if max(uniq) - min(uniq) == 4:
        return (True, max(uniq))
    return (False, 0)

def eval_5(cards5: List[str]) -> Tuple[int, List[int]]:
    ranks = [parse_card(c)[0] for c in cards5]
    suits = [parse_card(c)[1] for c in cards5]

    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    by_count = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    flush = len(set(suits)) == 1
    straight, straight_high = is_straight(ranks)

    if flush and straight:
        return (8, [straight_high])
    if by_count[0][1] == 4:
        quad = by_count[0][0]
        kicker = max(r for r in ranks if r != quad)
        return (7, [quad, kicker])
    if by_count[0][1] == 3 and by_count[1][1] == 2:
        return (6, [by_count[0][0], by_count[1][0]])
    if flush:
        return (5, sorted(ranks, reverse=True))
    if straight:
        return (4, [straight_high])
    if by_count[0][1] == 3:
        trips = by_count[0][0]
        kickers = sorted([r for r in ranks if r != trips], reverse=True)
        return (3, [trips] + kickers)
    if by_count[0][1] == 2 and by_count[1][1] == 2:
        hp = max(by_count[0][0], by_count[1][0])
        lp = min(by_count[0][0], by_count[1][0])
        kicker = max(r for r in ranks if r != hp and r != lp)
        return (2, [hp, lp, kicker])
    if by_count[0][1] == 2:
        pair = by_count[0][0]
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        return (1, [pair] + kickers)
    return (0, sorted(ranks, reverse=True))

def best_hand(pool: List[str]) -> Tuple[int, List[int]]:
    best = (-1, [])
    for combo in itertools.combinations(pool, 5):
        val = eval_5(list(combo))
        if val > best:
            best = val
    return best

def full_deck() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]


# ----------------------------
# Discard heuristic
# ----------------------------
def keep2_score(ca: str, cb: str, board: List[str]) -> int:
    ra, sa = parse_card(ca)
    rb, sb = parse_card(cb)
    rhi, rlo = max(ra, rb), min(ra, rb)
    sc = 0

    # pair
    if ra == rb:
        sc += 12 + (2 if ra >= 11 else 0)

    # suited
    if sa == sb:
        sc += 4

    # connectivity
    gap = rhi - rlo
    if gap == 1:
        sc += 4
    elif gap == 2:
        sc += 3
    elif gap == 3:
        sc += 1

    # high cards
    sc += (2 if rhi >= 13 else 0)
    sc += (1 if rlo >= 11 else 0)

    # tiny board awareness (matching ranks)
    if board:
        br = {parse_card(b)[0] for b in board}
        if ra in br:
            sc += 2
        if rb in br:
            sc += 2

    return sc

def choose_discard_index(cards3: List[str], board: List[str]) -> int:
    # discard the card that leaves best 2-card core
    best_idx = 0
    best_sc = -10**9
    for i in range(3):
        keep = [c for j, c in enumerate(cards3) if j != i]
        sc = keep2_score(keep[0], keep[1], board)
        if sc > best_sc:
            best_sc = sc
            best_idx = i
    return best_idx


# ----------------------------
# State abstraction
# ----------------------------
def strength_bucket_from_equity(eq: float) -> int:
    # 5 buckets similar to your idea but grounded
    if eq < 0.30:
        return 0
    if eq < 0.42:
        return 1
    if eq < 0.52:
        return 2
    if eq < 0.65:
        return 3
    return 4

def street_bucket(street: int) -> int:
    # 0 preflop, 1 discard streets, 2 early post-discard, 3 late (turn/river)
    if street == 0:
        return 0
    if street in (2, 3):
        return 1
    if street == 4:
        return 2
    return 3

def cost_bucket(cost: int) -> int:
    if cost <= 0:
        return 0
    if cost <= 2:
        return 1
    if cost <= 10:
        return 2
    if cost <= 50:
        return 3
    return 4


# ----------------------------
# Rollout equity (discard-aware)
# ----------------------------
def rollout_equity(
    my_hole: List[str],
    board: List[str],
    street: int,
    iters: int,
    rng: random.Random,
) -> float:
    """
    Returns P(win) + 0.5*P(tie) versus a random opponent, using discard heuristics.

    Important: We DO model the toss mechanic:
    - Preflop: both have 3 hole. We simulate flop(2), then our discard + opp discard,
      then turn+river to reach 6 board.
    - During discard streets: simulate the remaining discard(s) with heuristics.
    - After discard: simulate remaining board cards to reach 6.
    """
    known = set(my_hole + board)
    deck = [c for c in full_deck() if c not in known]
    wins = 0.0

    # figure how many hole cards we currently have (engine gives 3 preflop and during discard, 2 later)
    # We'll trust my_hole length from round_state.
    for _ in range(iters):
        d = deck[:]  # copy
        rng.shuffle(d)
        idx = 0

        # sample opponent private cards (same count as ours *at this point*)
        opp_hole = [d[idx], d[idx + 1]]
        idx += 2
        if len(my_hole) == 3:
            opp_hole.append(d[idx])
            idx += 1

        sim_board = board[:]
        sim_my = my_hole[:]
        sim_opp = opp_hole[:]

        # If preflop, need to deal flop(2) before discards
        if street == 0:
            # flop is 2 cards
            sim_board.extend([d[idx], d[idx + 1]])
            idx += 2

        # If we are before both discards are done, perform discards to get to 4-card board
        # Streets: 2 = first discard pending, 3 = second discard pending (framework detail)
        # We'll just ensure we end with 2 hole each and board length 4 before dealing turn/river.
        if len(sim_my) == 3 and len(sim_board) == 2:
            # no discards yet (we're right after flop in preflop simulation path)
            # both discard once
            my_di = choose_discard_index(sim_my, sim_board)
            sim_board.append(sim_my.pop(my_di))
            opp_di = choose_discard_index(sim_opp, sim_board)
            sim_board.append(sim_opp.pop(opp_di))

        elif len(sim_my) == 3 and len(sim_board) == 3:
            # one discard already happened (board has 3)
            # whoever hasn't discarded in simulation: both should end with 2
            if len(sim_my) == 3:
                my_di = choose_discard_index(sim_my, sim_board)
                sim_board.append(sim_my.pop(my_di))
            if len(sim_opp) == 3:
                opp_di = choose_discard_index(sim_opp, sim_board)
                sim_board.append(sim_opp.pop(opp_di))

        elif len(sim_my) == 3 and len(sim_board) == 4:
            # both discards already done on board, we should drop ours to 2 to match rules
            my_di = choose_discard_index(sim_my, sim_board)
            sim_my.pop(my_di)
            if len(sim_opp) == 3:
                opp_di = choose_discard_index(sim_opp, sim_board)
                sim_opp.pop(opp_di)

        # If we already have 2 hole, just proceed.
        if len(sim_my) == 3:
            # safety
            my_di = choose_discard_index(sim_my, sim_board)
            sim_my.pop(my_di)
        if len(sim_opp) == 3:
            opp_di = choose_discard_index(sim_opp, sim_board)
            sim_opp.pop(opp_di)

        # Now deal remaining board cards to reach 6
        while len(sim_board) < 6:
            sim_board.append(d[idx])
            idx += 1

        # Evaluate
        my_best = best_hand(sim_my + sim_board)
        opp_best = best_hand(sim_opp + sim_board)
        if my_best > opp_best:
            wins += 1.0
        elif my_best == opp_best:
            wins += 0.5

    return wins / float(iters)


# ----------------------------
# Main bot
# ----------------------------
class Player(Bot):
    def __init__(self):
        # regret matching tables
        self.regret_sum = defaultdict(lambda: [0.0] * NUM_ACTS)
        self.strategy_sum = defaultdict(lambda: [0.0] * NUM_ACTS)

        # per-round decision log for regret update
        self.history = []  # list of dicts describing each decision

        # exploration
        self.eps = 0.05
        self.rng = random.Random(7)

        # cache equities for speed
        self.eq_cache = {}  # key -> equity

    def handle_new_round(self, game_state: GameState, round_state: RoundState, active: int):
        self.history = []

    def get_action(self, game_state: GameState, round_state: RoundState, active: int):
        legal = round_state.legal_actions()
        street = round_state.street
        my_hole = list(round_state.hands[active])
        board = list(round_state.board) if round_state.board is not None else []

        # Discard decision: just do heuristic (learning discard online is expensive)
        if DiscardAction in legal:
            return DiscardAction(choose_discard_index(my_hole, board))

        # pot + cost info
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        cost = opp_pip - my_pip
        pot = 2 * STARTING_STACK - round_state.stacks[0] - round_state.stacks[1]

        # equity estimate (cached)
        eq_key = (street, tuple(sorted(my_hole)), tuple(board))
        if eq_key in self.eq_cache:
            eq = self.eq_cache[eq_key]
        else:
            # tune iterations: more preflop since discard is big variance
            iters = 28 if street == 0 else 16
            eq = rollout_equity(my_hole, board, street, iters, self.rng)
            self.eq_cache[eq_key] = eq

        # build abstract state key (infoset)
        sb = street_bucket(street)
        strb = strength_bucket_from_equity(eq)
        cb = cost_bucket(cost)
        pos = 1 if active == 0 else 0  # active==0 is SB per framework convention
        state = (sb, strb, cb, pos)

        # valid actions mask
        valid = [False] * NUM_ACTS

        # ACT_FOLD is only meaningful if we can't check for free
        can_check = (CheckAction in legal)
        can_call = (CallAction in legal)
        can_raise = (RaiseAction in legal)

        valid[ACT_FOLD] = True
        valid[ACT_CHECK_CALL] = True
        valid[ACT_RAISE_POT] = can_raise
        valid[ACT_ALL_IN] = can_raise

        # regret matching+
        regrets = self.regret_sum[state]
        pos_regrets = [max(regrets[i], 0.0) if valid[i] else 0.0 for i in range(NUM_ACTS)]
        s = sum(pos_regrets)
        if s > 1e-12:
            probs = [pos_regrets[i] / s for i in range(NUM_ACTS)]
        else:
            # uniform among valid
            k = sum(1 for v in valid if v)
            probs = [(1.0 / k if valid[i] else 0.0) for i in range(NUM_ACTS)]

        # epsilon exploration
        k = sum(1 for v in valid if v)
        if k > 0 and self.eps > 0:
            for i in range(NUM_ACTS):
                if valid[i]:
                    probs[i] = (1 - self.eps) * probs[i] + self.eps * (1.0 / k)
                else:
                    probs[i] = 0.0

        # accumulate average strategy
        for i in range(NUM_ACTS):
            self.strategy_sum[state][i] += probs[i]

        # sample action
        r = self.rng.random()
        cum = 0.0
        action_idx = ACT_FOLD
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                action_idx = i
                break

        # map action to engine action + compute "raise_to" for regret estimation later
        action_obj, raise_to = self.map_action(action_idx, round_state, active)

        # log this decision so we can update regrets when the hand ends
        self.history.append({
            "state": state,
            "probs": probs,
            "valid": valid,
            "street": street,
            "pot": pot,
            "cost": cost,
            "my_pip": my_pip,
            "opp_pip": opp_pip,
            "eq": eq,
            "action": action_idx,
            "raise_to": raise_to,  # None if not raising
        })

        return action_obj

    def handle_round_over(self, game_state: GameState, terminal_state: TerminalState, active: int):
        """
        Regret update using estimated EV per action at each logged decision.
        This is the key improvement: regrets are updated toward higher rollout-EV actions,
        not based on arbitrary my_delta multipliers.
        """
        if not self.history:
            return

        # We *could* use terminal_state.deltas[active] as a weak baseline, but the point
        # is to update regrets from the decision states using rollout equity + pot geometry.
        for step in self.history:
            state = step["state"]
            probs = step["probs"]
            valid = step["valid"]
            pot = step["pot"]
            cost = step["cost"]
            eq = step["eq"]
            my_pip = step["my_pip"]
            opp_pip = step["opp_pip"]
            played = step["action"]
            raise_to = step["raise_to"]

            # EV estimates for each abstract action from THIS decision point.
            # Key approximations:
            # - If you win at showdown with no more betting: profit = pot.
            # - If you call and lose: loss = cost.
            # - For raises: assume opponent continues (calls) and no further betting (crude but consistent).
            utilities = [0.0] * NUM_ACTS

            # Fold/check
            if cost <= 0:
                utilities[ACT_FOLD] = 0.0
            else:
                utilities[ACT_FOLD] = -float(cost)

            # Check/Call
            if cost <= 0:
                # check: realize equity on current pot at showdown (very approximate)
                utilities[ACT_CHECK_CALL] = eq * pot
            else:
                # if call, win profit = pot, lose = cost
                utilities[ACT_CHECK_CALL] = eq * pot - (1.0 - eq) * cost

            # Raises
            if raise_to is not None:
                # how much more we invest, and how much opponent would need to call
                raise_cost = raise_to - my_pip
                opp_call = max(0, raise_to - opp_pip)

                # If raise gets called and then goes to showdown:
                # win profit = pot + opp_call (because pot already exists; you win their extra)
                # lose = raise_cost
                called_ev = eq * (pot + opp_call) - (1.0 - eq) * raise_cost

                # You *could* model fold equity here if you track opponent fold-to-raise,
                # but even without it, this is more grounded than the old code.
                utilities[ACT_RAISE_POT] = called_ev
                utilities[ACT_ALL_IN] = called_ev  # will differ because raise_to differs (see map_action)
            else:
                utilities[ACT_RAISE_POT] = utilities[ACT_CHECK_CALL]
                utilities[ACT_ALL_IN] = utilities[ACT_CHECK_CALL]

            # Convert to regret update. Use expected utility under current strategy as baseline.
            strat_ev = 0.0
            for a in range(NUM_ACTS):
                if valid[a]:
                    strat_ev += probs[a] * utilities[a]

            # Regret Matching+ style: accumulate regrets (we'll clip during action selection)
            for a in range(NUM_ACTS):
                if valid[a]:
                    self.regret_sum[state][a] += (utilities[a] - strat_ev)

    # ----------------------------
    # Action mapping
    # ----------------------------
    def map_action(self, action_idx: int, round_state: RoundState, active: int):
        legal = round_state.legal_actions()
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        pot = 2 * STARTING_STACK - round_state.stacks[0] - round_state.stacks[1]

        can_check = (CheckAction in legal)
        can_call = (CallAction in legal)
        can_raise = (RaiseAction in legal)

        # fold (or check if free)
        if action_idx == ACT_FOLD:
            if can_check:
                return CheckAction(), None
            return FoldAction(), None

        # check/call
        if action_idx == ACT_CHECK_CALL:
            if can_call:
                return CallAction(), None
            return CheckAction(), None

        # raises
        if not can_raise:
            # fallback
            if can_call:
                return CallAction(), None
            return CheckAction(), None

        min_r, max_r = round_state.raise_bounds()

        if action_idx == ACT_RAISE_POT:
            # "pot raise" target: make it roughly pot-sized over current opponent contribution
            # target total contribution to pot from us: opp_pip + pot
            target = opp_pip + pot
            amt = clamp(target, min_r, max_r)
            return RaiseAction(amt), amt

        # all-in
        if action_idx == ACT_ALL_IN:
            return RaiseAction(max_r), max_r

        return CheckAction(), None


if __name__ == "__main__":
    run_bot(Player(), parse_args())
