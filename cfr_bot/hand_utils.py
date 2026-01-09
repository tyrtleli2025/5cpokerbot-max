# hand_utils.py
# A lightweight, pure-Python hand evaluator for the 3-card game

RANKS = "23456789TJQKA"
SUITS = "shdc"

def parse_card(card_str):
    # Converts 'Ah' to (14, 'h')
    rank_char = card_str[0]
    suit = card_str[1]
    rank = RANKS.index(rank_char) + 2
    return rank, suit

def evaluate_strength(cards, board=[]):
    """
    Returns a score (0-100) representing hand strength.
    Higher is better.
    """
    all_cards = cards + board
    if not all_cards:
        return 0
    
    parsed = [parse_card(c) for c in all_cards]
    ranks = sorted([p[0] for p in parsed], reverse=True)
    suits = [p[1] for p in parsed]
    
    # Check Flush (Need 3 of same suit in this variant?) 
    # The README says 3 hole cards. Standard poker needs 5 for flush. 
    # Let's assume standard 3-card poker rules for the abstraction buckets:
    # 3-of-a-kind > Straight > Flush > Pair > High Card
    
    # Count frequencies
    rank_counts = {r: ranks.count(r) for r in ranks}
    suit_counts = {s: suits.count(s) for s in suits}
    
    max_suit = max(suit_counts.values()) if suit_counts else 0
    max_rank_count = max(rank_counts.values()) if rank_counts else 0
    
    # 1. 3-of-a-kind (Trips)
    if max_rank_count >= 3:
        return 90 + ranks[0] # Score ~100
    
    # 2. Straight
    unique_ranks = sorted(list(set(ranks)))
    if len(unique_ranks) >= 3:
        for i in range(len(unique_ranks)-2):
            if unique_ranks[i+2] - unique_ranks[i] == 2:
                return 80 + unique_ranks[i+2]
                
    # 3. Flush (3 cards same suit)
    if max_suit >= 3:
        return 70 + ranks[0]
        
    # 4. Pair
    if max_rank_count >= 2:
        # Find which rank is the pair
        pair_rank = [r for r, c in rank_counts.items() if c >= 2][0]
        return 50 + pair_rank
        
    # 5. High Card
    return ranks[0] # Score 2-14

def get_abstraction(cards, board, history):
    """
    Groups millions of hands into a few 'buckets' for training.
    """
    score = evaluate_strength(cards, board)
    
    # Bucket by strength
    if score >= 90: bucket = "Monster"
    elif score >= 80: bucket = "Straight"
    elif score >= 70: bucket = "Flush"
    elif score >= 50: bucket = "Pair"
    elif score >= 13: bucket = "HighAceKing"
    elif score >= 10: bucket = "MidCard"
    else: bucket = "Low"
    
    # Special case: Pre-flop Discard Phase
    # If we are deciding what to discard, we need to know what we are KEEPING.
    # But for general betting, 'bucket' + 'history' is the key.
    
    return f"{bucket}|{history}"