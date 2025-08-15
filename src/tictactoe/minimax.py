"""
minimax.py
Minimax + α-β pruning with configurable depth ("difficulty").
Small clean-ups: constants, type hints, faster loops, DRY helpers.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tictactoe.game_logic import Board  # forward ref to avoid circular imports

# --------------------------------------------------------------------------- #
# Configuration constants                                                     #
# --------------------------------------------------------------------------- #
SCORE_WIN: int = 10
SCORE_LOSS: int = -10
SCORE_DRAW: int = 0

DEPTH_MAP: Dict[str, int] = {
    "random": 0,
    "easy": 2,
    "medium": 5,
    "hard": 7,
    "perfect": 9,
}


# --------------------------------------------------------------------------- #
# Minimax player                                                              #
# --------------------------------------------------------------------------- #
class MinimaxPlayer:
    """
    Classic minimax with α-β pruning.
    Difficulty is mapped to a max search depth for quick tuning.
    """

    def __init__(
        self,
        player_char: str,
        difficulty: str = "perfect",
    ) -> None:
        self.player_char: str = player_char.upper()
        self.opponent_char: str = "O" if self.player_char == "X" else "X"
        self.max_depth: int = DEPTH_MAP[difficulty]

        # stats
        self.games_played: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #
    def _evaluate(self, board: Board) -> int:
        """Return static score for terminal or truncated positions."""
        result = board.check_win()
        if result == self.player_char:
            return SCORE_WIN
        if result == self.opponent_char:
            return SCORE_LOSS
        if result == 0:
            return SCORE_DRAW
        return 0  # non-terminal

    # ------------------------------------------------------------------ #
    # Core search                                                        #
    # ------------------------------------------------------------------ #
    def _minimax(
        self,
        board: Board,
        depth: int,
        maximizing: bool,
        alpha: float = -math.inf,
        beta: float = math.inf,
    ) -> float:
        score = self._evaluate(board)

        # terminal or depth limit reached
        if score != 0 or depth >= self.max_depth or board.check_win() is not None:
            return score

        moves = self._available_moves(board)
        if not moves:
            return SCORE_DRAW

        if maximizing:
            value = -math.inf
            for m in moves:
                board.place_char(self.player_char, m + 1)
                value = max(value, self._minimax(board, depth + 1, False, alpha, beta))
                board.undo_move(m)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # β cut-off
            return value
        else:
            value = math.inf
            for m in moves:
                board.place_char(self.opponent_char, m + 1)
                value = min(value, self._minimax(board, depth + 1, True, alpha, beta))
                board.undo_move(m)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # α cut-off
            return value

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _available_moves(board: Board) -> List[int]:
        """Return 0-based indices of empty cells."""
        return [i for i, cell in enumerate(board.get_board()) if cell == " "]

    def get_best_move(self, board: Board) -> Optional[int]:
        """Return 0-based index of the best move (None if board full)."""
        moves = self._available_moves(board)
        if not moves:
            return None

        best_score, best_move = -math.inf, None
        for m in moves:
            board.place_char(self.player_char, m + 1)
            score = self._minimax(board, 0, False)
            board.undo_move(m)
            if score > best_score:
                best_score, best_move = score, m
        return best_move

    def make_move(self, board: Board) -> Optional[int]:
        """Perform the best move on the supplied board and return it."""
        move = self.get_best_move(board)
        if move is not None:
            board.place_char(self.player_char, move + 1)
        return move

    # ------------------------------------------------------------------ #
    # Stats                                                              #
    # ------------------------------------------------------------------ #
    def update_stats(self, result: str | int) -> None:
        self.games_played += 1
        if result == self.player_char:
            self.wins += 1
        elif result == self.opponent_char:
            self.losses += 1
        elif result == 0:
            self.draws += 1

    def get_win_rate(self) -> float:
        return 0.0 if self.games_played == 0 else self.wins / self.games_played

    def get_stats(self) -> dict:
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.get_win_rate(),
            "difficulty": next(k for k, v in DEPTH_MAP.items() if v == self.max_depth),
        }

    def set_player_char(self, char: str) -> None:
        """Change player symbol on the fly."""
        self.player_char = char.upper()
        self.opponent_char = "O" if self.player_char == "X" else "X"


# --------------------------------------------------------------------------- #
# Imperfect variant (occasional random blunder)                              #
# --------------------------------------------------------------------------- #
class ImperfectMinimaxPlayer(MinimaxPlayer):
    """
    Same as MinimaxPlayer, but occasionally makes a random mistake
    controlled by `mistake_probability`.
    """

    def __init__(self, player_char: str, mistake_probability: float = 0.1) -> None:
        super().__init__(player_char, difficulty="perfect")
        self.mistake_p: float = mistake_probability

    def make_move(self, board: Board) -> Optional[int]:
        if random.random() < self.mistake_p:
            moves = self._available_moves(board)
            if moves:
                blunder = random.choice(moves)
                board.place_char(self.player_char, blunder + 1)
                return blunder
        return super().make_move(board)


# --------------------------------------------------------------------------- #
# Simple self-test (run `python -m algorithms.minimax`)                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.tictactoe.game_logic import Board
    from random_player import RandomPlayer

    def _bench() -> None:
        for diff, depth in DEPTH_MAP.items():
            bot = MinimaxPlayer("X", difficulty=diff)
            rand = RandomPlayer("O")
            wins = draws = 0
            for _ in range(100):
                b = Board()
                turn = bot
                while b.check_win() is None:
                    turn.make_move(b)
                    turn = rand if turn is bot else bot
                res = b.check_win()
                if res == "X":
                    wins += 1
                elif res == 0:
                    draws += 1
            print(
                f"{diff:>7}: W/D {wins:>3}/{draws:<3} "
                f"(win-rate {wins/100:.2%})"
            )

    _bench()