"""
Poker AI — Fixed, robust version with:
 - correct 5-card evaluator (works on any 5-card combo)
 - Monte Carlo equity vs N opponents (uses random.sample to avoid slicing errors)
 - preflop categorization + postflop evaluation
 - EV-aware recommendation (FOLD / CALL / RAISE / ALL-IN)
 - Raise suggestions as a range (respects effective stack)
 - Simple CLI demo
"""

import random
import itertools
import math
from collections import Counter
import matplotlib.pyplot as plt

# ----------------------------
# Card utilities
# ----------------------------
RANKS = "23456789TJQKA"
RANK_TO_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
VALUE_TO_RANK = {v: r for r, v in RANK_TO_VALUE.items()}
SUITS = "cdhs"  # clubs, diamonds, hearts, spades

def parse_card(s):
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"Invalid card format: {s}. Use 'As', 'Td', etc.")
    r, su = s[0].upper(), s[1].lower()
    if r not in RANK_TO_VALUE or su not in SUITS:
        raise ValueError(f"Invalid card: {s}")
    return r + su

def full_deck():
    return [r + s for r in RANKS for s in SUITS]

# ----------------------------
# Robust 5-card evaluator (evaluate exactly 5 cards)
# returns a tuple where larger is stronger
# categories: 8 SF, 7 Quads, 6 FH, 5 Flush, 4 Straight, 3 Trips, 2 TwoPair, 1 Pair, 0 HighCard
# ----------------------------
def _rank_counts(cards):
    return Counter(RANK_TO_VALUE[c[0]] for c in cards)

def _suit_counts(cards):
    return Counter(c[1] for c in cards)

def _is_flush(cards):
    suits = _suit_counts(cards)
    for s, cnt in suits.items():
        if cnt >= 5:
            return True, s
    return False, None

def _is_straight(rvals):
    """Given a list of rank values (may have duplicates), detect straight and return (True, top_value)."""
    vals = sorted(set(rvals), reverse=True)
    # Ace-low handling: append 1 if Ace present
    if 14 in vals:
        vals = vals + [1]
    # find any run of length >=5
    for i in range(len(vals) - 4):
        window = vals[i:i+5]
        if max(window) - min(window) == 4 and len(window) == 5:
            return True, max(window)
    return False, None

def evaluate_5(cards):
    """Evaluate exactly 5 cards and return comparable tuple."""
    vals = [RANK_TO_VALUE[c[0]] for c in cards]
    suits = [c[1] for c in cards]
    counts = _rank_counts(cards)
    counts_items = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    count_vals = sorted(counts.values(), reverse=True)
    is_flush, flush_suit = _is_flush(cards)
    is_straight, top_straight = _is_straight(vals)

    # Straight flush
    if is_flush:
        flush_cards = [c for c in cards if c[1] == flush_suit]
        if len(flush_cards) >= 5:
            f_vals = [RANK_TO_VALUE[c[0]] for c in flush_cards]
            sf, top_sf = _is_straight(f_vals)
            if sf:
                return (8, top_sf)

    # Four of a kind
    if 4 in count_vals:
        four_rank = max(r for r, cnt in counts.items() if cnt == 4)
        kickers = sorted((v for v in vals if v != four_rank), reverse=True)
        return (7, four_rank, kickers[0])

    # Full house
    if 3 in count_vals and 2 in count_vals:
        trips_rank = max(r for r, cnt in counts.items() if cnt == 3)
        pair_rank = max(r for r, cnt in counts.items() if cnt == 2)
        return (6, trips_rank, pair_rank)
    if count_vals.count(3) == 2:
        trips = sorted([r for r, cnt in counts.items() if cnt == 3], reverse=True)
        return (6, trips[0], trips[1])

    # Flush
    if is_flush:
        flush_vals = sorted([RANK_TO_VALUE[c[0]] for c in cards if c[1] == flush_suit], reverse=True)
        top5 = tuple(flush_vals[:5])
        return (5,) + top5

    # Straight
    if is_straight:
        return (4, top_straight)

    # Trips
    if 3 in count_vals:
        trip_rank = max(r for r, cnt in counts.items() if cnt == 3)
        kickers = sorted((v for v in vals if v != trip_rank), reverse=True)[:2]
        return (3, trip_rank, kickers[0], kickers[1])

    # Two pair
    pairs = sorted([r for r, cnt in counts.items() if cnt == 2], reverse=True)
    if len(pairs) >= 2:
        kicker = max(v for v in vals if v not in pairs)
        return (2, pairs[0], pairs[1], kicker)

    # One pair
    if 2 in count_vals:
        pair_rank = max(r for r, cnt in counts.items() if cnt == 2)
        kickers = sorted((v for v in vals if v != pair_rank), reverse=True)[:3]
        return (1, pair_rank, kickers[0], kickers[1], kickers[2])

    # High card
    top5 = tuple(sorted(vals, reverse=True)[:5])
    return (0,) + top5

def best_hand(cards):
    """Return best 5-card evaluation for cards (len(cards) must be >=5)."""
    if len(cards) < 5:
        return None
    best = None
    for combo in itertools.combinations(cards, 5):
        score = evaluate_5(combo)
        if best is None or score > best:
            best = score
    return best

# ----------------------------
# Preflop categorization
# ----------------------------
def preflop_category(hole):
    v1, v2 = RANK_TO_VALUE[hole[0][0]], RANK_TO_VALUE[hole[1][0]]
    suited = (hole[0][1] == hole[1][1])
    pair = v1 == v2
    gap = abs(v1 - v2)
    high = max(v1, v2)
    # return (label, score 0..1)
    if pair:
        if v1 >= 13:  # AA,KK
            return "Premium Pair", 0.96
        elif v1 >= 10:  # TT-QQ
            return "Strong Pair", 0.86
        elif v1 >= 7:
            return "Medium Pair", 0.76
        else:
            return "Small Pair", 0.60
    if suited and gap == 1 and high >= 8:
        return "Suited Connector", 0.68
    if suited and gap <= 3 and high >= 10:
        return "Suited Broadway-ish", 0.72
    if high >= 11 and gap <= 3:
        return "Broadway-ish", 0.68
    if suited and gap <= 4:
        return "Suited Gapper", 0.55
    if high >= 13:
        return "High Card", 0.5
    return "Trash", 0.30

# ----------------------------
# Outs/draw heuristics (lightweight)
# ----------------------------
def detect_flush_draw(hole, board):
    all_cards = hole + board
    suits = _suit_counts(all_cards)
    suit, cnt = suits.most_common(1)[0]
    # If we have 4 to a suit total (holes + board), it's a flush draw
    if cnt == 4:
        return True, 9
    return False, 0

def detect_straight_draw(hole, board):
    all_vals = sorted(set(RANK_TO_VALUE[c[0]] for c in hole + board))
    # naive check for open-ended or gutshot: check any 4-of-5-run missing one
    # We'll approximate: OE ~8 outs, gutshot ~4
    for low in range(1, 11):  # start of 5-run
        needed = set(range(low, low+5))
        present = needed & set(all_vals)
        if len(present) == 4:
            # decide if open-ended or gutshot by positions
            # approximate: if present contains endpoints -> gutshot, else OE
            mn, mx = min(present), max(present)
            if (mn == low) or (mx == low+4):
                return True, 4  # gutshot approx
            else:
                return True, 8  # open ended approx
    return False, 0

def estimate_outs(hole, board):
    fd, f_outs = detect_flush_draw(hole, board)
    sd, s_outs = detect_straight_draw(hole, board)
    # pair-up outs: if a hole card is single, 3 outs maybe
    rank_counts = _rank_counts(hole + board)
    pair_outs = 0
    for h in hole:
        r = RANK_TO_VALUE[h[0]]
        present = rank_counts.get(r, 0)
        if present < 2:
            pair_outs = max(pair_outs, 4 - present)
    outs = max(f_outs, s_outs, pair_outs)
    return {"flush_draw": fd, "flush_outs": f_outs, "straight_draw": sd, "straight_outs": s_outs, "pair_outs": pair_outs, "outs_est": outs}

# ----------------------------
# Monte Carlo equity vs N opponents
# ----------------------------
def monte_carlo_equity(hole, board, n_opponents=1, iterations=2000, seed=None):
    """Return dict with equity/win/tie/lose fractions. Uses random.sample each iter to avoid slicing issues."""
    if seed is not None:
        random.seed(seed)
    deck = full_deck()
    used = set([c.upper() for c in hole + board])
    deck = [c for c in deck if c.upper() not in used]
    wins = ties = losses = 0
    to_deal = 5 - len(board)
    if to_deal < 0:
        raise ValueError("Board has too many cards.")
    # ensure we have enough cards to deal; if too many opponents, still works with sampling
    for _ in range(iterations):
        draw = random.sample(deck, to_deal + 2 * n_opponents)
        new_board = board + draw[:to_deal]
        opp_holes = []
        idx = to_deal
        for _ in range(n_opponents):
            opp_holes.append([draw[idx], draw[idx+1]])
            idx += 2
        my_best = best_hand([c.upper() for c in hole + new_board])
        opp_bests = [best_hand([c.upper() for c in opp + new_board]) for opp in opp_holes]
        # Compare
        # If my_best > all opps -> win. If any opp > my_best -> loss. If all <= and at least one == -> tie.
        if all(opp < my_best for opp in opp_bests):
            wins += 1
        elif any(opp > my_best for opp in opp_bests):
            losses += 1
        else:
            ties += 1
    total = iterations
    win_pct = wins / total
    tie_pct = ties / total
    lose_pct = losses / total
    equity = win_pct + tie_pct * 0.5
    return {"wins": wins, "ties": ties, "losses": losses, "win_pct": win_pct, "tie_pct": tie_pct, "lose_pct": lose_pct, "equity": equity, "iterations": iterations}

# ----------------------------
# Feature builder & aggregated strength
# ----------------------------
def compute_features(hole, board, stage, n_opponents=1, effective_stack=1000, pot=100, to_call=0, iterations=1500, aggression=0.6):
    hole = [c.upper() for c in hole]
    board = [c.upper() for c in board]

    mc = monte_carlo_equity(hole, board, n_opponents=n_opponents, iterations=iterations)
    equity = mc["equity"]

    if stage.lower() == "preflop":
        cat_label, cat_score = preflop_category(hole)
        best = None
    else:
        best = best_hand(hole + board)
        cat_label = f"post_{best[0]}" if best is not None else "unknown"
        cat_score = (best[0] / 8.0) if best is not None else 0.0

    suited = 1.0 if hole[0][1] == hole[1][1] else 0.0
    rv = sorted([RANK_TO_VALUE[c[0]] for c in hole], reverse=True)
    gap = abs(rv[0] - rv[1])
    connectedness = max(0.0, 1.0 - (gap - 1) / 12.0)

    outs_info = estimate_outs(hole, board)
    outs_norm = outs_info["outs_est"] / 20.0

    # SPR and stack context
    denom = (pot + to_call) if (pot + to_call) > 0 else 1.0
    s2p = effective_stack / denom
    s2p_norm = min(1.0, math.log1p(s2p) / 5.0)

    # stage weight (late streets give more weight to made hands)
    stage_weight = {"preflop": 0.6, "flop": 0.8, "turn": 0.95, "river": 1.0}
    sw = stage_weight.get(stage.lower(), 0.8)

    features = {
        "equity": equity,
        "category_score": cat_score,
        "suited": suited,
        "connectedness": connectedness,
        "board_conn": 0.0,  # could compute more advanced metric
        "outs_norm": outs_norm,
        "s2p_norm": s2p_norm,
        "aggression": aggression,
        "n_opponents": n_opponents,
        "stage_weight": sw,
        "mc": mc,
        "outs_info": outs_info,
        "best_hand_score": best,
        "preflop_label": cat_label if stage.lower() == "preflop" else None
    }

    # Weighted aggregator
    weights = {
        "equity": 0.62,
        "category_score": 0.22 * sw,
        "suited": 0.05,
        "connectedness": 0.03,
        "board_conn": 0.02,
        "outs_norm": 0.06,
        "s2p_norm": 0.03,
        "aggression": 0.03
    }
    raw_strength = sum(features[k] * weights.get(k, 0) for k in features if k in weights)
    normalized_strength = min(1.0, raw_strength / 1.05)

    features["raw_strength"] = raw_strength
    features["strength"] = normalized_strength

    return features

# ----------------------------
# Decision engine (EV-aware, gives raise ranges)
# ----------------------------
def recommend_action(features, pot, to_call, effective_stack, min_raise_mult=1.5, max_raise_mult=3.5):
    equity = features["equity"] = features["mc"]["equity"]
    strength = features["strength"]
    n_opps = features["n_opponents"]
    aggression = features["aggression"]
    s2p_norm = features["s2p_norm"]

    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
    implied_factor = 1 + s2p_norm * 0.9
    looseness_bonus = 0.04 * aggression
    required_equity = max(0.0, pot_odds / implied_factor - looseness_bonus)

    call_ev = equity * (pot + to_call) - (1 - equity) * to_call if to_call > 0 else 0.0

    # Monster
    if equity >= 0.85 or strength >= 0.95:
        # big raise (but clipped by effective_stack)
        min_bet = min(effective_stack, int(max(1, min_raise_mult * pot)))
        max_bet = min(effective_stack, int(max(1, max_raise_mult * pot)))
        return {"action": "RAISE (VALUE)", "range": (min_bet, max_bet), "reason": "Very strong hand (monster).", "call_ev": call_ev, "required_equity": required_equity}

    # Strong value region
    if equity >= 0.58 or strength >= 0.75:
        min_bet = min(effective_stack, int(max(1, 1.2 * pot)))  # around 1.2x pot to 2.5x
        max_bet = min(effective_stack, int(max(1, 2.5 * pot * (1 + 0.3*aggression))))
        return {"action": "RAISE", "range": (min_bet, max_bet), "reason": "Strong hand/value build.", "call_ev": call_ev, "required_equity": required_equity}

    # Marginal / draw spots: call if meets required equity or EV positive
    if equity >= required_equity or call_ev >= 0 or (features["outs_info"]["outs_est"] >= 4 and aggression > 0.5):
        # semi-bluff raise sometimes on draws
        if features["outs_info"]["outs_est"] >= 4 and aggression > 0.5:
            min_bet = min(effective_stack, int(max(1, 0.35 * pot)))
            max_bet = min(effective_stack, int(max(1, 0.9 * pot)))
            return {"action": "RAISE (SEMI-BLUFF)", "range": (min_bet, max_bet), "reason": "Decent draw + aggression allows semi-bluff.", "call_ev": call_ev, "required_equity": required_equity}
        else:
            return {"action": "CALL", "amount": to_call, "reason": "Equity >= required or call EV positive.", "call_ev": call_ev, "required_equity": required_equity}

    # Fold, except small float if very aggressive and to_call is tiny
    if aggression >= 0.85 and to_call <= max(1, int(0.03 * pot)):
        return {"action": "CALL (FLOAT)", "amount": to_call, "reason": "Aggressive float on tiny bet.", "call_ev": call_ev, "required_equity": required_equity}

    # Shove if short stack and marginally ahead
    spr = effective_stack / max(pot, 1)
    if spr <= 3 and equity > 0.5:
        return {"action": "ALL-IN", "amount": effective_stack, "reason": "Short stack shove", "call_ev": call_ev, "required_equity": required_equity}

    return {"action": "FOLD", "reason": f"Equity {equity:.2f} below required {required_equity:.2f} and negative EV.", "call_ev": call_ev, "required_equity": required_equity}

# ----------------------------
# Visualization
# ----------------------------
def plot_analysis(features, hole, board, stage):
    mc = features["mc"]
    labels = ['Win', 'Tie', 'Loss']
    sizes = [mc['win_pct'], mc['tie_pct'], mc['lose_pct']]
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.pie(sizes, labels=[f"{l} ({v*100:.1f}%)" for l, v in zip(labels, sizes)])
    plt.title(f"{stage.upper()} — Equity {features['mc']['equity']:.2f}")

    plt.subplot(1,2,2)
    keys = ['equity', 'category_score', 'suited', 'connectedness', 'outs_norm', 's2p_norm']
    vals = [features.get(k, 0) for k in keys]
    bars = plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.title(f"Feature snapshot | Hand: {' '.join(hole)} | Board: {' '.join(board) if board else '---'}")
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()

# ----------------------------
# CLI Demo
# ----------------------------
def input_cards(prompt, expected=2):
    s = input(prompt + " (e.g. 'As Ks'): ").strip().split()
    if len(s) != expected:
        raise ValueError(f"Expected {expected} cards, got {len(s)}.")
    return [parse_card(x) for x in s]

def quick_run_demo():
    wsop = input("WSOP settings? (y or n): ")
    print("=== Poker AI demo (All Stages) ===")
    
    while True:  # Loop for multiple hands
        # Reset everything for a new hand
        hand = input_cards("Enter your hole cards", 2)
        stages = ["preflop", "flop", "turn", "river"]
        board = []

        stage_index = 0
        while stage_index < len(stages):
            if not hand:
                hand = input_cards("Enter your hole cards", 2)
            stage = stages[stage_index]
            repeat_stage = True
            while repeat_stage:
                print(f"\n--- Stage: {stage.capitalize()} ---")
                
                # Enter board cards progressively
                if stage == "flop" and len(board) == 0:
                    board += input_cards("Enter flop cards", 3)
                elif stage == "turn" and len(board) == 3:
                    board += input_cards("Enter turn card", 1)
                elif stage == "river" and len(board) == 4:
                    board += input_cards("Enter river card", 1)

                pot = int(input("Enter current pot size: "))
                to_call = int(input("Amount to call: "))
                eff_stack = int(input("Effective stack size (your stack or smallest in pot): "))
                n_opps = int(input("Number of opponents (1..8): "))
                if wsop == "y":
                    iterations = 2000
                    agression = 0.6
                else:
                    iterations = int(input("Monte Carlo iterations (e.g. 2000): ") or 2000)
                    aggression = float(input("Aggression (0.0 conservative - 1.0 loose-aggressive) [0.6]: ") or 0.6)

                features = compute_features(hand, board, stage, n_opponents=n_opps, effective_stack=eff_stack, pot=pot, to_call=to_call, iterations=iterations, aggression=aggression)
                plot_analysis(features, hand, board, stage)
                rec = recommend_action(features, pot, to_call, eff_stack)

                print("\n=== Result ===")
                print(f"Hole: {' '.join(hand)} | Board: {' '.join(board) if board else '---'}")
                if stage.lower() == "preflop":
                    print(f"Preflop category: {features.get('preflop_label')} (score {features['category_score'] if 'category_score' in features else 'N/A'})")
                else:
                    print(f"Best hand score (postflop): {features['best_hand_score']}")
                print(f"Monte Carlo equity: {features['mc']['equity']:.3f} (win {features['mc']['win_pct']:.3f}, tie {features['mc']['tie_pct']:.3f})")
                print(f"Estimated outs (heuristic): {features['outs_info']['outs_est']}")
                print("\nRecommendation:")
                if rec["action"].startswith("RAISE"):
                    low, high = rec["range"]
                    print(f"  {rec['action']}: suggested raise between {low} and {high} chips. Reason: {rec['reason']}")
                elif rec["action"] == "ALL-IN":
                    print(f"  SHOVE ALL-IN for {rec['amount']} chips. Reason: {rec['reason']}")
                elif rec["action"].startswith("CALL"):
                    print(f"  CALL {rec.get('amount', to_call)}. Reason: {rec['reason']}")
                else:
                    print(f"  {rec['action']}. Reason: {rec['reason']}")
                print(f"Call EV: {rec.get('call_ev', 0):.2f}, Required equity: {rec.get('required_equity', 0):.3f}")

                # Reraise / restart logic
                reraise_input = input(
                    "Was there a reraise by an opponent? (y = repeat stage, n = next stage, r = restart hand): "
                ).strip().lower()

                if reraise_input == 'y':
                    repeat_stage = True
                elif reraise_input == 'r':
                    print("\nRestarting entire hand...\n")
                    repeat_stage = False
                    stage_index = -1  # Reset stage loop
                    board = []
                    hand = []  # Clear hole cards so new ones are entered
                    break
                else:  # 'n' or other input
                    repeat_stage = False

            stage_index += 1
            
if __name__ == "__main__":
    quick_run_demo()
