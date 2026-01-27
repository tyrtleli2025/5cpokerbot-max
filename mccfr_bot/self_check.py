import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from mccfr_bot.player import Player
from python_skeleton.skeleton.states import RoundState, STARTING_STACK


def make_round_state(street, pips, stacks, hands, board, button=1):
    return RoundState(button, street, pips, stacks, hands, board, None)


def run_checks():
    player = Player()

    hands = [
        (["As", "Ad", "7c"], ["Ah", "2d", "9s"]),
        (["Ks", "Qs", "9h"], ["2s", "7s", "Jd"]),
        (["8c", "8d", "Kh"], ["4s", "8h", "Ts"]),
    ]
    for hand, board in hands:
        discard_idx = player.choose_discard(hand, board, street=2)
        print(f"hand={hand} board={board} discard={hand[discard_idx]}")

    my_hand = ["Ah", "Qd", "7s"]
    board = ["2c", "9d", "Jh", "4s"]
    pips = [0, 4]
    stacks = [STARTING_STACK, STARTING_STACK - 4]
    round_state = make_round_state(4, pips, stacks, [my_hand, []], board)
    should_call = player.should_call_postflop(my_hand, board, round_state, active=0)
    print(f"postflop_call_vs_4={should_call}")


if __name__ == "__main__":
    run_checks()
