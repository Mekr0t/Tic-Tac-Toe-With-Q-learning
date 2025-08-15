"""
random_player.py
A trivial agent that picks moves uniformly at random.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.game_logic import Board


class RandomPlayer:
    """Agent that plays uniformly at random on any empty cell."""

    def __init__(self, player_char: str) -> None:
        self.player_char: str = player_char.upper()

    def make_move(self, board: Board) -> int | None:
        """
        Choose an empty cell at random and play there.

        Args:
            board: live Board instance (mutated in-place).

        Returns:
            0-based index of the move actually played, or None if no move left.
        """
        available = [idx for idx, cell in enumerate(board.get_board()) if cell == " "]
        if not available:
            return None

        move = random.choice(available)
        board.place_char(self.player_char, move + 1)  # convert to 1-based
        return move

    def set_player_char(self, new_char: str):
        self.player_char: str = new_char.upper()
