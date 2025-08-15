"""
human_player.py
Let a human play against the trained Q-learning agent in the console.
"""

from __future__ import annotations

from typing import Dict, Tuple

import models.model_manager as mng
from agents.q_agent import QLearningAgent
from game.game_logic import Board


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _choose_sides() -> Tuple[str, str, bool]:
    """
    Prompt user for X or O.
    Returns (human_char, agent_char, human_goes_first).
    """
    while True:
        choice = input("\nChoose your side:\n1. You play X (go first)\n2. You play O (go second)\nEnter 1 or 2: ").strip()
        if choice == "1":
            return "X", "O", True
        if choice == "2":
            return "O", "X", False
        print("Please enter 1 or 2.")


def _prompt_move(board: Board, human_char: str) -> int:
    """Ask human for a 1-9 move until valid."""
    while True:
        try:
            move = int(input(f"Your move ({human_char}): "))
            board.place_char(human_char, move)
            return move
        except (ValueError, IndexError) as e:
            print(f"Invalid input: {e}")


def _play_single_game(
    agent: QLearningAgent,
    human_char: str,
    human_first: bool,
) -> str:
    """
    Run one game.
    Returns 'human', 'agent', or 'draw'.
    """
    board = Board()
    current = "human" if human_first else "agent"

    board.print_board()
    while True:
        if current == "human":
            _prompt_move(board, human_char)
            current = "agent"
        else:
            print(f"Agent ({agent.player_char}) is thinking...")
            agent.make_move(board, training=False)  # deterministic
            current = "human"

        board.print_board()

        result = board.check_win()
        if result is not None:
            if result == human_char:
                return "human"
            if result == agent.player_char:
                return "agent"
            return "draw"


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def play_vs_agent() -> None:
    """Main CLI loop."""
    agent = QLearningAgent()
    if not mng.load_model_for_agent(agent):
        return

    print(
        f"\nLoaded agent with {agent.games_played} games of experience "
        f"(win-rate: {agent.get_win_rate():.3f})\n"
    )
    print("Positions 1-9:")
    print("1 | 2 | 3\n4 | 5 | 6\n7 | 8 | 9\n")

    human_char, agent_char, human_first = _choose_sides()
    agent.set_player_char(agent_char)

    stats: Dict[str, int] = {"human": 0, "agent": 0, "draw": 0}

    while True:
        winner = _play_single_game(agent, human_char, human_first)
        stats[winner] += 1

        print(f"\nScore → You: {stats['human']}  Agent: {stats['agent']}  Draws: {stats['draw']}")

        again = input("\nPlay again? [y/n]: ").strip().lower()
        if again != "y":
            break

        # Swap sides for variety
        human_char, agent_char = agent_char, human_char
        agent.set_player_char(agent_char)
        human_first = not human_first

    print("\nThanks for playing!")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random, os, sys

    random.seed(42)  # deterministic
    os.environ["PYTHONHASHSEED"] = "42"

    play_vs_agent()
