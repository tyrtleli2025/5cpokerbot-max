'''
Simple example pokerbot, written in Python.
'''
import traceback
import random
import itertools
import json
import os

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from treys import Card, Evaluator

class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.
        '''
        self.evaluator = Evaluator()
        
        # --- LOAD THE PREFLOP CHEAT SHEET ---
        self.preflop_equity = {}
        try:
            # 1. Get the absolute path to the folder containing this file (player.py)
            my_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. Join that directory with the filename
            file_path = os.path.join(my_dir, "preflop_equity.json")
            
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    self.preflop_equity = json.load(f)
                
            else:
                print(f"WARNING: preflop_equity.json not found at: {file_path}")
                
        except Exception as e:
            print(f"ERROR loading equity file: {e}")
    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_deck_stub(self, known_cards):
        """
        Returns a list of all cards in the deck EXCLUDING the known_cards.
        """
        full_deck = [
            Card.new(rank + suit) 
            for rank in '23456789TJQKA' 
            for suit in 'shdc'
        ]
        known_set = set(known_cards)
        stub = [card for card in full_deck if card not in known_set]
        return stub

    def get_hand_strength(self, hole_cards, board_cards):
        """
        Calculates the strongest 5-card hand from any number of hole/board cards.
        """
        all_cards = hole_cards + board_cards
        if len(all_cards) <= 7:
            return self.evaluator.evaluate(board_cards, hole_cards)

        best_score = float('inf')
        for combo in itertools.combinations(all_cards, 5):
            combo_list = list(combo)
            score = self.evaluator.evaluate([], combo_list)
            if score < best_score:
                best_score = score
        return best_score
    
    def choose_best_discard(self, my_3_cards, current_board):
        """
        Monte Carlo Simulation to decide which card to discard.
        Returns the index (0, 1, or 2) of the card to discard.
        """
        best_card_index = 0
        max_win_rate = -1.0
        NUM_SIMULATIONS = 200 
        
        for i, discard_candidate in enumerate(my_3_cards):
            my_keep_hand = [c for idx, c in enumerate(my_3_cards) if idx != i]
            sim_board_start = current_board + [discard_candidate]
            
            known_cards = my_3_cards + current_board
            stub = self.get_deck_stub(known_cards)
            wins = 0
            
            for _ in range(NUM_SIMULATIONS):
                random.shuffle(stub)
                cards_needed_for_board = 6 - len(sim_board_start)
                future_board = sim_board_start + stub[:cards_needed_for_board]
                opponent_hand = stub[cards_needed_for_board : cards_needed_for_board + 2]
                
                my_score = self.get_hand_strength(my_keep_hand, future_board)
                opp_score = self.get_hand_strength(opponent_hand, future_board)
                
                if my_score < opp_score:
                    wins += 1
                elif my_score == opp_score:
                    wins += 0.5 
            
            win_rate = wins / NUM_SIMULATIONS
            if win_rate > max_win_rate:
                max_win_rate = win_rate
                best_card_index = i
                
        return best_card_index
            
    def calculate_equity(self, hole_cards, board_cards):
        """
        Runs a Monte Carlo sim.
        Simulates opponent having 3 cards and discarding 1 if we are early in the hand.
        """
        wins = 0
        NUM_SIMULATIONS = 100 
        
        opponent_has_3_cards = (len(board_cards) <= 2)
        
        known_cards = hole_cards + board_cards
        stub = self.get_deck_stub(known_cards)
        
        for _ in range(NUM_SIMULATIONS):
            random.shuffle(stub)
            current_stub = list(stub) # Make a copy to pop from
            
            # 1. Deal Opponent Hand
            if opponent_has_3_cards:
                villain_raw = current_stub[:3]
                current_stub = current_stub[3:]
            else:
                villain_raw = current_stub[:2]
                current_stub = current_stub[2:]
            
            # 2. Deal Board Runout
            cards_needed = max(0, 6 - len(board_cards))
            future_board = board_cards + current_stub[:cards_needed]
            
            # 3. Simulate Opponent Discard (if they had 3 cards)
            if len(villain_raw) == 3:
                best_v_score = float('inf')
                villain_final = villain_raw[:2] 
                
                for i in range(3):
                    keep = villain_raw[:i] + villain_raw[i+1:]
                    score = self.get_hand_strength(keep, future_board)
                    if score < best_v_score:
                        best_v_score = score
                        villain_final = keep
            else:
                villain_final = villain_raw
            
            # 4. Compare vs Hero
            my_score = self.get_hand_strength(hole_cards, future_board)
            opp_score = self.get_hand_strength(villain_final, future_board)
            
            if my_score < opp_score:
                wins += 1
            elif my_score == opp_score:
                wins += 0.5
                
        return wins / NUM_SIMULATIONS

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens. Called every time it's our turn.
        '''
        try:
            legal_actions = round_state.legal_actions() 
            
            my_cards_ints = [Card.new(c) for c in round_state.hands[active]]
            board_cards_ints = [Card.new(c) for c in round_state.board]
            
            # --- DECISION 1: DISCARD ---
            if DiscardAction in legal_actions:
                best_card_index = self.choose_best_discard(my_cards_ints, board_cards_ints)
                return DiscardAction(best_card_index)

            # --- DECISION 2: BETTING ---
            
            # 2a. Calculate Equity (Use Lookup Table if Preflop!)
            equity = 0.5
            if round_state.street == 0 and self.preflop_equity:
                hand_strs = [Card.int_to_str(c) for c in my_cards_ints]
                key = "".join(sorted(hand_strs))
                equity = self.preflop_equity.get(key, 0.5)
            else:
                equity = self.calculate_equity(my_cards_ints, board_cards_ints)
            
            # 2b. Pot Odds
            my_pip = round_state.pips[active]
            opp_pip = round_state.pips[1-active]
            continue_cost = opp_pip - my_pip
            
            my_stack = round_state.stacks[active]
            opp_stack = round_state.stacks[1-active]
            current_pot = (STARTING_STACK - my_stack) + (STARTING_STACK - opp_stack)
            pot_total = current_pot + continue_cost
            
            if pot_total == 0:
                pot_odds = 0
            else:
                pot_odds = continue_cost / (pot_total + continue_cost)
            
            can_check = CheckAction in legal_actions
            can_raise = RaiseAction in legal_actions

            # --- STRATEGY: EQUITY vs POT ODDS with EXTREME RISK AVERSION ---
            
            stack_commitment = 0
            if (my_stack + continue_cost) > 0:
                stack_commitment = continue_cost / (my_stack + continue_cost)
            
            # RISK CONTROL: 
            # If the call is large (>= 30% of stack), we go into "Nit Mode".
            if stack_commitment >= 0.30:
                # Top 10% hands generally have ~65% equity or better.
                # If we are facing a shove, we FOLD unless we have 65%+ equity.
                required_equity = 0.65
            else:
                # Standard pot odds for small bets
                required_equity = pot_odds

            if equity > required_equity: 
                if can_raise:
                    try:
                        min_raise, max_raise = round_state.raise_bounds()
                        amount = min_raise
                    except:
                        amount = 2
                    return RaiseAction(amount)
                
                if can_check: return CheckAction()
                return CallAction()
                
            elif equity > pot_odds:
                # Positive EV but doesn't meet our "Nit Mode" threshold -> FOLD
                if can_check: return CheckAction()
                return FoldAction()
                
            else:
                if can_check: return CheckAction()
                return FoldAction()

        except Exception as e:
            # Don't write to a file, just print it!
            print("CRASH IN GET_ACTION:", e)
            traceback.print_exc()
            return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())