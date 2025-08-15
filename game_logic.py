"""
game_logic.py
Core board state and move-handling for a 3×3 Tic-Tac-Toe game.
Compatible drop-in replacement for earlier game_logic.py
"""

from __future__ import annotations

from typing import Literal, Tuple, List

Player = Literal["X", "O"]
CellState = Literal["X", "O", " "]
BoardState = List[CellState]

# --------------------------------------------------------------------------- #
# Public constants                                                            #
# --------------------------------------------------------------------------- #
WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)

NUMERIC_ENCODE = {"X": 1, "O": -1, " ": 0}
NUMERIC_DECODE = {1: "X", -1: "O", 0: " "}  # reverse map, handy for ML


class InvalidMoveError(ValueError):
    """Raised when a move violates game rules."""


class Board:
    """
    Immutable public interface:
        place_char() mutates internal state (kept for compat),
        otherwise prefer get_numeric_board() for ML / search.
    """

    def __init__(self) -> None:
        self._board: BoardState = [" "] * 9

    # ------------------------------------------------------------------ #
    # Static helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def empty() -> BoardState:
        """Return a fresh empty board list (static helper)."""
        return [" "] * 9

    # ------------------------------------------------------------------ #
    # I/O helpers                                                        #
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        return "".join(self._board)

    def pretty(self) -> str:
        """
        Human-readable ASCII board, suitable for console or logs.
        """
        b = self._board
        return (
            "+---+---+---+\n"
            f"| {b[0]} | {b[1]} | {b[2]} |\n"
            "+---+---+---+\n"
            f"| {b[3]} | {b[4]} | {b[5]} |\n"
            "+---+---+---+\n"
            f"| {b[6]} | {b[7]} | {b[8]} |\n"
            "+---+---+---+"
        )

    def print_board(self) -> None:
        """Legacy wrapper; delegates to pretty()."""
        print(self.pretty())

    # ------------------------------------------------------------------ #
    # Core game mechanics                                                #
    # ------------------------------------------------------------------ #
    def place_char(self, char: str, place: int) -> None:
        """
        Place a mark on the board (1-based index).

        Args:
            char: "X" or "O" (case-insensitive)
            place: 1–9

        Raises:
            ValueError:  invalid char or out-of-range place
            InvalidMoveError: cell already occupied
        """
        char = char.upper()
        if char not in {"X", "O"}:
            raise ValueError("Character must be 'X' or 'O'")
        if not 1 <= place <= 9:
            raise ValueError("Place must be 1-9")

        idx = place - 1
        if self._board[idx] != " ":
            raise InvalidMoveError("Cell already occupied")

        self._board[idx] = char

    def check_win(self) -> Player | int | None:
        """
        Return:
            "X" or "O" if that player has won
            0          if board is full (draw)
            None       otherwise
        """
        for a, b, c in WIN_LINES:
            if self._board[a] == self._board[b] == self._board[c] != " ":
                return self._board[a]  # type: ignore[return-value]

        return 0 if self.is_full() else None

    # ------------------------------------------------------------------ #
    # Utility / ML helpers                                               #
    # ------------------------------------------------------------------ #
    def get_board(self) -> BoardState:
        """Return a *copy* of the raw board list."""
        return self._board.copy()

    def get_numeric_board(self) -> tuple[int, ...]:
        """
                Encode the board as ints: 1 = X, -1 = O, 0 = empty.
        """
        return tuple(NUMERIC_ENCODE[cell] for cell in self._board)

    def undo_move(self, move: int) -> None:
        """
        Revert a 0-based index move (used by search algorithms).
        """
        self._board[move] = " "

    def is_full(self) -> bool:
        """True if no empty cell left."""
        return " " not in self._board

    # ------------------------------------------------------------------ #
    # Convenience read-only properties                                   #
    # ------------------------------------------------------------------ #
    @property
    def available_moves(self) -> List[int]:
        """0-based indices of empty cells."""
        return [i for i, v in enumerate(self._board) if v == " "]