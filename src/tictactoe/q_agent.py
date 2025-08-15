"""
q_agent.py
A Q-learning agent for 3×3 Tic-Tac-Toe.
"""

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional

from .game_logic import Board

import yaml
import pathlib

_CONFIG_PATH = pathlib.Path(__file__).parents[2] / "config.yaml"
_config = yaml.safe_load(_CONFIG_PATH.read_text())

__all__ = ["QLearningAgent"]

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
DEFAULT_ALPHA = _config["q_agent"]["learning_rate"]
DEFAULT_GAMMA = _config["q_agent"]["discount_factor"]
DEFAULT_EPS = _config["q_agent"]["epsilon_start"]
EPS_DECAY = _config["q_agent"]["epsilon_decay"]
EPS_MIN = _config["q_agent"]["epsilon_min"]


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
class QLearningAgent:
    """
    Tabular Q-learning agent for Tic-Tac-Toe.
    """

    def __init__(
        self,
        player_char: str = "X",
        learning_rate: float = DEFAULT_ALPHA,
        discount_factor: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPS,
    ) -> None:
        self.player_char: str = player_char.upper()
        self.opponent_char: str = "O" if self.player_char == "X" else "X"

        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon

        # Q-table:  state_key + "_" + action  →  float
        self.q_table: defaultdict[tuple[tuple[int, ...], int], float] = defaultdict(float)

        # Statistics
        self.games_played: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def make_move(self, board: Board, *, training: bool = True) -> Optional[int]:
        """Pick and execute a move (1-based) returning 0-based index."""
        action = self._choose_action(board, training)
        if action is not None:
            board.place_char(self.player_char, action + 1)
        return action

    def update_q_value(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: List[int],
    ) -> None:
        """Apply Bellman update to a single (s,a) pair."""
        key = self._make_key(state, action)
        current_q = self.q_table[key]

        max_next = 0.0
        next_board = self._board_from_numeric(next_state)
        if (avail := next_board.available_moves):
            max_next = max(self.q_table[self._make_key(next_state, a)] for a in avail)

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next - current_q
        )
        self.q_table[key] = new_q

    def update_stats(self, game_result: str | int) -> None:
        """Update win/loss/draw counters."""
        self.games_played += 1
        if game_result == self.player_char:
            self.wins += 1
        elif game_result == self.opponent_char:
            self.losses += 1
        elif game_result == 0:
            self.draws += 1

    def get_reward(self, game_result: str | int) -> float:
        """Return immediate reward for game outcome."""
        if game_result == self.player_char:
            return 1.0
        if game_result == self.opponent_char:
            return -1.0
        if game_result == 0:
            return 0.5
        return 0.0  # on-going

    def decay_epsilon(self) -> None:
        """Reduce exploration rate once per training cycle."""
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    def get_stats(self) -> Dict[str, int | float]:
        """Return a snapshot of current statistics."""
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self._win_rate(),
        }

    def get_win_rate(self):
        return self._win_rate()

    def save_model(self, path: str) -> None:
        """Persist Q-table and meta-data to disk."""
        payload = {
            "q_table": dict(self.q_table),
            "stats": self.get_stats(),
            "params": {
                "player_char": self.player_char,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> bool:
        """Load previously saved model."""
        if not os.path.exists(path):
            return False
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.q_table = defaultdict(float, data["q_table"])
        stats = data["stats"]
        self.games_played = stats["games_played"]
        self.wins, self.losses, self.draws = stats["wins"], stats["losses"], stats["draws"]
        print(f"Model loaded from {path}")
        return True

    def set_player_char(self, char: str) -> None:
        """Change symbol and update opponent accordingly."""
        self.player_char = char.upper()
        self.opponent_char = "O" if self.player_char == "X" else "X"

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _choose_action(self, board: Board, training: bool) -> Optional[int]:
        """Epsilon-greedy action selection."""
        avail = board.available_moves
        if not avail:
            return None

        # Explore
        if training and random.random() < self.epsilon:
            return random.choice(avail)

        # Exploit
        state = board.get_numeric_board()
        best_val = float("-inf")
        best_move = None
        for a in avail:
            val = self.q_table[self._make_key(state, a)]
            if val > best_val:
                best_val, best_move = val, a
        return best_move or random.choice(avail)

    @staticmethod
    def _make_key(state: tuple[int, ...], action: int) -> tuple[tuple[int, ...], int]:
        return (state, action)

    @staticmethod
    def _board_from_numeric(state: List[int]) -> Board:
        """Reconstruct Board from numeric encoding."""
        board = Board()
        board._board = [
            " " if v == 0 else ("X" if v == 1 else "O") for v in state
        ]
        return board

    def _win_rate(self) -> float:
        return 0.0 if self.games_played == 0 else self.wins / self.games_played