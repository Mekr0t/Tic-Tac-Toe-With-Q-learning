"""
trainer.py
Refactored training orchestrator for the Q-learning Tic-Tac-Toe agent.

Compatible with the original q_agent.py without any modifications.
"""

from __future__ import annotations

from typing import List, Tuple, Callable

import model_manager as mng
from q_agent import QLearningAgent
from minimax import MinimaxPlayer, ImperfectMinimaxPlayer
from random_player import RandomPlayer
from game_logic import Board

import yaml
import pathlib

_CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")
_config = yaml.safe_load(_CONFIG_PATH.read_text())

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
PlayerFactory = Callable[[str], object]  # opponent generator with a given char


def _run_games(
    agent: QLearningAgent,
    make_opponent: PlayerFactory,
    num_games: int,
    print_interval: int,
    decay_each: int = 100,
) -> None:
    """
    Generic training loop.
    `make_opponent` is a callable that returns an opponent when given its char.
    """
    for game in range(num_games):
        # Alternate starting order and symbols
        agent.set_player_char("X" if game % 2 == 0 else "O")
        opponent = make_opponent("O" if agent.player_char == "X" else "X")

        board = Board()
        states_buffer: List[Tuple[List[int], int]] = []

        current, other = (
            (agent, opponent) if agent.player_char == "X" else (opponent, agent)
        )

        # Play game
        while board.check_win() is None:
            if current is agent:
                state = board.get_numeric_board()
                action = agent.make_move(board, training=True)
                if action is not None:
                    states_buffer.append((state, action))
            else:
                current.make_move(board)

            current, other = other, current

        # Learn
        result = board.check_win()
        agent.update_stats(result)
        final_reward = agent.get_reward(result)

        for i, (s, a) in enumerate(reversed(states_buffer)):
            discounted = final_reward * (agent.discount_factor ** i)
            next_s = board.get_numeric_board() if i == 0 else states_buffer[-i][0]
            agent.update_q_value(s, a, discounted, next_s)

        # Exploration decay
        if game % decay_each == 0:
            agent.decay_epsilon()

        # Logging
        if (game + 1) % print_interval == 0:
            stats = agent.get_stats()
            print(
                f"Game {game + 1}: "
                f"Win-rate={stats['win_rate']:.3f}, "
                f"ε={agent.epsilon:.3f}, "
                f"|Q|={len(agent.q_table)}"
            )


def _train_against_random(agent: QLearningAgent, num_games: int,
                          print_interval: int = _config["training"]["against_random"]["print_interval"]) -> None:
    _run_games(agent, RandomPlayer, num_games, print_interval)


def _train_against_minimax(agent: QLearningAgent, difficulty: str, num_games: int,
                           print_interval: int = _config["training"]["against_minimax"]["print_interval"]) -> None:
    _run_games(agent, lambda c: MinimaxPlayer(c, difficulty=difficulty), num_games, print_interval)


def _train_self_play(agent1: QLearningAgent, agent2: QLearningAgent, num_games: int,
                     print_interval: int = _config["training"]["against_selfplay"]["print_interval"]) -> None:
    """Self-play between two *mutable* agents."""
    for game in range(num_games):
        agent1.set_player_char("X" if game % 2 == 0 else "O")
        agent2.set_player_char("O" if agent1.player_char == "X" else "X")

        board = Board()
        states1, states2 = [], []

        current, other = (agent1, agent2) if agent1.player_char == "X" else (agent2, agent1)

        while board.check_win() is None:
            state = board.get_numeric_board()
            action = current.make_move(board, training=True)
            if action is not None:
                (states1 if current is agent1 else states2).append((state, action))
            current, other = other, current

        result = board.check_win()
        agent1.update_stats(result)
        agent2.update_stats(result)

        for agent, states in ((agent1, states1), (agent2, states2)):
            reward = agent.get_reward(result)
            for i, (s, a) in enumerate(reversed(states)):
                discounted = reward * (agent.discount_factor ** i)
                next_s = board.get_numeric_board() if i == 0 else states[-i][0]
                agent.update_q_value(s, a, discounted, next_s)

            if game % 100 == 0:
                agent.decay_epsilon()

        if (game + 1) % print_interval == 0:
            print(
                f"Game {game + 1}: "
                f"A1-win={agent1.get_win_rate():.3f}, "
                f"A2-win={agent2.get_win_rate():.3f}"
            )


def _train_against_all_difficulties(agent: QLearningAgent, num_games: int,
                                    print_interval: int = _config["training"]["all_difficulties"]["print_interval"]
                                    ) -> None:
    """Randomly rotate through all minimax levels + random."""
    difficulties = ["random", "easy", "medium", "hard", "perfect"]

    def opponent_factory(char: str) -> object:
        diff = random.choice(difficulties)
        print(diff)
        return RandomPlayer(char) if diff == "random" else MinimaxPlayer(char, difficulty=diff)

    _run_games(agent, opponent_factory, num_games, print_interval)


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_agent_vs_all_opponents(agent: QLearningAgent, games_per_opponent: int = 1000):
    opponents = [
        ("Random", lambda c: RandomPlayer(c)),
        ("Easy Minimax", lambda c: MinimaxPlayer(c, "easy")),
        ("Medium Minimax", lambda c: MinimaxPlayer(c, "medium")),
        ("Hard Minimax", lambda c: MinimaxPlayer(c, "hard")),
        ("Perfect Minimax", lambda c: MinimaxPlayer(c, "perfect")),
        ("Imperfect Minimax", lambda c: ImperfectMinimaxPlayer(c, 0.1)),
    ]

    results = {}
    backup = agent.get_stats()  # save original stats

    for name, factory in opponents:
        agent.games_played = agent.wins = agent.losses = agent.draws = 0
        opp = factory("O" if agent.player_char == "X" else "X")

        for _ in range(games_per_opponent):
            board = Board()
            cur, other = (agent, opp) if agent.player_char == "X" else (opp, agent)

            while board.check_win() is None:
                (cur if cur is agent else other).make_move(board, training=False)
                cur, other = other, cur

            agent.update_stats(board.check_win())

        results[name] = agent.get_stats()
        print(
            f"{name:<20} | win-rate={results[name]['win_rate']:.3f} "
            f"({results[name]['wins']}W/{results[name]['losses']}L/{results[name]['draws']}D)"
        )

    # restore original counters
    for k, v in backup.items():
        setattr(agent, k, v)
    return results


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _choice(prompt: str, options: dict) -> str:
    """Small helper for numbered menus."""
    print(prompt)
    for k, v in options.items():
        print(f"{k}. {v}")
    return input("Choice: ").strip()


def main() -> None:
    print("Q-Learning Tic-Tac-Toe Trainer\n")

    menu = {
        "1": "Train against random player",
        "2": "Train against specific minimax difficulty",
        "3": "Train via self-play (two agents)",
        "4": "Train against all difficulties (random schedule)",
        "9": "Evaluate existing model",
        "0": "Exit",
    }

    choice = _choice("Select mode:", menu)

    agent = QLearningAgent(
        player_char="X",
        learning_rate=_config["q_agent"]["learning_rate"],
        discount_factor=_config["q_agent"]["discount_factor"],
        epsilon=_config["q_agent"]["epsilon_start"],
    )

    agent2 = QLearningAgent(
        player_char="O",
        learning_rate=_config["q_agent"]["learning_rate"],
        discount_factor=_config["q_agent"]["discount_factor"],
        epsilon=_config["q_agent"]["epsilon_start"],
    )

    if choice == "1":
        mng.load_model_for_agent(agent)
        _train_against_random(agent, num_games=_config["training"]["against_random"]["games"])
        mng.save_model_from_agent(agent)

    elif choice == "2":
        diff = _choice(
            "Difficulty:",
            {"1": "easy", "2": "medium", "3": "hard", "4": "perfect"},
        )
        mng.load_model_for_agent(agent)
        _train_against_minimax(agent, difficulty=diff, num_games=_config["training"]["against_minimax"]["games"])
        mng.save_model_from_agent(agent)

    elif choice == "3":
        print("\nLoading model for first agent...")
        mng.load_model_for_agent(agent)

        print("\nLoading model for second agent...")
        mng.load_model_for_agent(agent2)

        _train_self_play(agent, agent2, num_games=_config["training"]["against_selfplay"]["games"])

        print("\nSaving model for first agent...")
        mng.save_model_from_agent(agent)

        print("\nSaving model for second agent...")
        mng.save_model_from_agent(agent2)

    elif choice == "4":
        mng.load_model_for_agent(agent)
        _train_against_all_difficulties(agent, num_games=_config["training"]["all_difficulties"]["games"])
        mng.save_model_from_agent(agent)

    elif choice == "9":
        mng.load_model_for_agent(agent)
        evaluate_agent_vs_all_opponents(agent, games_per_opponent=500)

    elif choice == "0":
        print("Bye!")
    else:
        print("Invalid option.")

    stats = agent.get_stats()
    print("\nSummary:")
    print(
        f"Games={stats['games_played']}  "
        f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}  "
        f"Win-rate={stats['win_rate']:.3f}  "
        f"|Q|={len(agent.q_table)}"
    )


if __name__ == "__main__":
    import random, os

    random.seed(42)  # deterministic
    os.environ["PYTHONHASHSEED"] = "42"

    main()
