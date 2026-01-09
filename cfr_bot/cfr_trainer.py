import random
import pickle
from hand_utils import evaluate_strength, get_abstraction

class Node:
    def __init__(self, num_actions):
        self.regret_sum = [0.0] * num_actions
        self.strategy_sum = [0.0] * num_actions
        
    def get_strategy(self, realization_weight):
        normalizing_sum = 0
        strategy = [0.0] * len(self.regret_sum)
        for i, val in enumerate(self.regret_sum):
            strategy[i] = val if val > 0 else 0
            normalizing_sum += strategy[i]

        for i in range(len(strategy)):
            if normalizing_sum > 0:
                strategy[i] /= normalizing_sum
            else:
                strategy[i] = 1.0 / len(strategy)
            self.strategy_sum[i] += realization_weight * strategy[i]
        return strategy

    def get_average_strategy(self):
        normalizing_sum = sum(self.strategy_sum)
        if normalizing_sum > 0:
            return [s / normalizing_sum for s in self.strategy_sum]
        return [1.0 / len(self.strategy_sum)] * len(self.strategy_sum)

class MCCFRTrainer:
    def __init__(self):
        self.nodes = {}
        # Actions: 0=Fold, 1=Check/Call, 2=Raise, 3,4,5=Discard(0/1/2)
        self.action_chars = {0:'F', 1:'C', 2:'R', 3:'d0', 4:'d1', 5:'d2'}

    def train(self, iterations):
        print(f"Starting Advanced MCCFR for {iterations} iterations...")
        deck = [r+s for r in "23456789TJQKA" for s in "shdc"]
        
        for i in range(iterations):
            random.shuffle(deck)
            cards = [deck[0:3], deck[3:6]]
            board = deck[6:11] 
            
            self.cfr(cards, board, history="", active_player=0)
            
            if i % 1000 == 0 and i > 0:
                print(f"Iteration {i}")

        self.save_strategy()

    def get_legal_actions(self, history):
        if "F" in history: return [] 

        phases = history.split('/')
        current_phase_actions = phases[-1]
        
        # --- PHASE 0 (Pre-flop) & PHASE 3 (Post-flop) ---
        if len(phases) == 1 or len(phases) == 4:
            if is_betting_finished(current_phase_actions): 
                return [] # Should have transitioned
            
            # STOP INFINITE RECURSION: Cap raises at 3 per street
            raise_count = current_phase_actions.count('R')
            can_raise = (raise_count < 3)

            if can_raise:
                return [0, 1, 2] # Fold, Call, Raise
            else:
                return [0, 1] # Fold, Call (No Raise allowed)

        # --- PHASE 1 & 2 (Discard) ---
        if len(phases) == 2 or len(phases) == 3:
             return [3, 4, 5] # Discard 0, 1, or 2

        return []

    def cfr(self, cards, board, history, active_player):
        # 1. CHECK TERMINAL
        if "F" in history:
            return self.get_payoff(history, cards, board, active_player)
        
        phases = history.split('/')
        # Check if post-flop betting ended
        if len(phases) >= 4 and is_betting_finished(phases[-1]):
             return self.get_payoff(history, cards, board, active_player)

        # 2. HANDLE PHASE TRANSITIONS
        # Use recursion to move to next phase if current phase is done
        current_phase_log = phases[-1]
        
        # Transition from Pre-flop -> Discard 1
        if len(phases) == 1 and is_betting_finished(current_phase_log):
            return self.cfr(cards, board, history + "/", 1) # P1 discards first (Rule check)
        
        # Transition from Discard 1 -> Discard 2
        if len(phases) == 2: # P1 just discarded
             return self.cfr(cards, board, history + "/", 0) # P0 discards next

        # Transition from Discard 2 -> Post-flop
        if len(phases) == 3: # Both discarded
             return self.cfr(cards, board, history + "/", 1) # P1 starts post-flop

        # 3. GET INFOSET & NODE
        my_cards = cards[active_player]
        current_board = []
        if len(phases) >= 4: current_board = board[0:3]

        abstraction = get_abstraction(my_cards, current_board, history)
        legal_moves = self.get_legal_actions(history)
        
        if not legal_moves: 
            return 0 # Should have been terminal

        info_set = abstraction
        if info_set not in self.nodes:
            self.nodes[info_set] = Node(len(legal_moves))
        node = self.nodes[info_set]

        # 4. MCCFR RECURSION
        strategy = node.get_strategy(1.0)
        util = [0.0] * len(legal_moves)
        node_util = 0

        for i, action_idx in enumerate(legal_moves):
            action_char = self.action_chars[action_idx]
            
            # Prepare next state
            next_cards = [list(c) for c in cards]
            next_board = list(board)
            next_p = 1 - active_player
            
            # Handle Discard Execution
            if action_idx >= 3: 
                card_to_drop = action_idx - 3
                dropped = next_cards[active_player].pop(card_to_drop)
                next_board.append(dropped)
                # If discarding, we recurse with SAME player active (logic handled by phase transition at top)
                next_p = active_player 

            # Recurse
            if action_idx >= 3:
                 # Discarding: Don't flip player here, top logic handles it
                 util[i] = self.cfr(next_cards, next_board, history + action_char, next_p)
            else:
                 # Betting: Flip player
                 util[i] = -self.cfr(next_cards, next_board, history + action_char, next_p)
            
            node_util += strategy[i] * util[i]

        # Update Regrets
        for i in range(len(legal_moves)):
            regret = util[i] - node_util
            node.regret_sum[i] += regret

        return node_util

    def get_payoff(self, history, cards, board, player):
        # Simple Pot Calculation: Blinds(3) + Bets
        pot = 3 + history.count('R') * 2 + history.count('C')
        
        if history.endswith('F'): return pot
        
        p0_score = evaluate_strength(cards[0], board)
        p1_score = evaluate_strength(cards[1], board)
        
        if player == 0:
            if p0_score > p1_score: return pot
            if p0_score < p1_score: return -pot
            return 0
        else:
            if p1_score > p0_score: return pot
            if p1_score < p0_score: return -pot
            return 0

    def save_strategy(self):
        final_strategy = {}
        for info_set, node in self.nodes.items():
            final_strategy[info_set] = node.get_average_strategy()
        with open("strategy.pkl", "wb") as f:
            pickle.dump(final_strategy, f)
        print("Strategy saved.")

def is_betting_finished(segment):
    if segment == "": return False
    if segment == "CC": return True
    if segment.endswith("RC") and len(segment) > 1: return True
    if segment == "RC": return True
    return False

if __name__ == "__main__":
    trainer = MCCFRTrainer()
    trainer.train(50000)