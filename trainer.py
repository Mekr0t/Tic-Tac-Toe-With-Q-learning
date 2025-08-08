import models.model_manager as mng
from game.game_logic import Board
from agents.q_agent import QLearningAgent
from algorithms.minimax import MinimaxPlayer, ImperfectMinimaxPlayer
from algorithms.random_player import RandomPlayer


def train_against_random(agent, num_games=10000, print_interval=1000):
    print(f"Training agent against random player for {num_games} games...")

    for game in range(num_games):
        if game % 2 == 0:
            agent.set_player_char('X')
        else:
            agent.set_player_char('O')

        opponent = RandomPlayer(agent.opponent_char)

        board = Board()
        game_states = []

        current_player = agent if agent.player_char == 'X' else opponent
        other_player = opponent if current_player == agent else agent

        while board.check_win() is None:
            if current_player == agent:
                state_before = board.get_numeric_board().copy()
                action = agent.make_move(board, training=True)
                if action is not None:
                    game_states.append((state_before, action))
            else:
                current_player.make_move(board)

            current_player, other_player = other_player, current_player

        game_result = board.check_win()
        agent.update_stats(game_result)

        final_reward = agent.get_reward(game_result)

        for i, (state, action) in enumerate(reversed(game_states)):
            reward = final_reward * (agent.discount_factor ** i)
            agent.update_q_value(state, action, reward, board.get_numeric_board())

        if game % 100 == 0:
            agent.decay_epsilon()

        if (game + 1) % print_interval == 0:
            stats = agent.get_stats()
            print(f"Game {game + 1}: Win Rate: {stats['win_rate']:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Q-table size: {len(agent.q_table)}")


def train_self_play(agent1, agent2, num_games=5000, print_interval=500):
    print(f"Training agents against each other for {num_games} games...")

    for game in range(num_games):
        board = Board()
        game_states_1 = []
        game_states_2 = []

        if game % 2 == 0:
            agent1.set_player_char('X')
            agent2.set_player_char('O')

        else:
            agent1.set_player_char('O')
            agent2.set_player_char('X')

        current_agent = agent1 if agent1.player_char == 'X' else agent2
        other_agent = agent2 if current_agent == agent1 else agent1

        while board.check_win() is None:
            state_before = board.get_numeric_board().copy()
            action = current_agent.make_move(board, training=True)

            if action is not None:
                if current_agent == agent1:
                    game_states_1.append((state_before, action))
                else:
                    game_states_2.append((state_before, action))

            current_agent, other_agent = other_agent, current_agent

        game_result = board.check_win()
        agent1.update_stats(game_result)
        agent2.update_stats(game_result)

        for agent, states in [(agent1, game_states_1), (agent2, game_states_2)]:
            final_reward = agent.get_reward(game_result)

            for i, (state, action) in enumerate(reversed(states)):
                reward = final_reward * (agent.discount_factor ** i)
                agent.update_q_value(state, action, reward, board.get_numeric_board())

        if game % 100 == 0:
            agent1.decay_epsilon()
            agent2.decay_epsilon()

        if (game + 1) % print_interval == 0:
            stats1 = agent1.get_stats()
            stats2 = agent2.get_stats()
            print(f"Game {game + 1}: First Agent Win Rate: {stats1['win_rate']:.3f}, "
                  f"Second Agent Win Rate Win Rate: {stats2['win_rate']:.3f}")


def train_against_minimax(agent, difficulty='medium', num_games=5000, print_interval=500):
    print(f"Training {agent.player_char} agent against {difficulty} minimax for {num_games} games...")

    opponent = MinimaxPlayer('O' if agent.player_char == 'X' else 'X', difficulty=difficulty)

    for game in range(num_games):
        board = Board()
        game_states = []

        if agent.player_char == 'X':
            current_player = agent
            other_player = opponent
        else:
            current_player = opponent
            other_player = agent

        while board.check_win() is None:
            if current_player == agent:
                state_before = board.get_numeric_board().copy()
                action = agent.make_move(board, training=True)
                if action is not None:
                    game_states.append((state_before, action))
            else:
                current_player.make_move(board)

            current_player, other_player = other_player, current_player

        game_result = board.check_win()
        agent.update_stats(game_result)
        opponent.update_stats(game_result)

        final_reward = agent.get_reward(game_result)

        for i, (state, action) in enumerate(reversed(game_states)):
            reward = final_reward * (agent.discount_factor ** i)

            if i == 0:
                next_state = board.get_numeric_board()
            else:
                next_state = game_states[len(game_states) - i][0]

            agent.update_q_value(state, action, reward, next_state)

        if game % 100 == 0:
            agent.decay_epsilon()

        if (game + 1) % print_interval == 0:
            agent_stats = agent.get_stats()
            opponent_stats = opponent.get_stats()
            print(f"Game {game + 1}: Agent Win Rate: {agent_stats['win_rate']:.3f}, "
                  f"Minimax Win Rate: {opponent_stats['win_rate']:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}")


def evaluate_agent_vs_all_opponents(agent, games_per_opponent=1000):
    print(f"\nEvaluating agent against different opponents ({games_per_opponent} games each):")

    opponents = [
        ('Random', RandomPlayer('O' if agent.player_char == 'X' else 'X')),
        ('Easy Minimax', MinimaxPlayer('O' if agent.player_char == 'X' else 'X', 'easy')),
        ('Medium Minimax', MinimaxPlayer('O' if agent.player_char == 'X' else 'X', 'medium')),
        ('Hard Minimax', MinimaxPlayer('O' if agent.player_char == 'X' else 'X', 'hard')),
        ('Perfect Minimax', MinimaxPlayer('O' if agent.player_char == 'X' else 'X', 'perfect')),
        ('Imperfect Minimax', ImperfectMinimaxPlayer('O' if agent.player_char == 'X' else 'X', 0.1))
    ]

    results = {}

    for opponent_name, opponent in opponents:
        print(f"\nTesting against {opponent_name}...")

        original_stats = agent.get_stats()
        agent.games_played = 0
        agent.wins = 0
        agent.losses = 0
        agent.draws = 0

        for _ in range(games_per_opponent):
            board = Board()

            if agent.player_char == 'X':
                current_player = agent
                other_player = opponent
            else:
                current_player = opponent
                other_player = agent

            while board.check_win() is None:
                if current_player == agent:
                    agent.make_move(board, training=False)
                else:
                    if hasattr(other_player, 'make_move'):
                        other_player.make_move(board)

                current_player, other_player = other_player, current_player

            result = board.check_win()
            agent.update_stats(result)

        stats = agent.get_stats()
        results[opponent_name] = stats
        print(f"Win rate: {stats['win_rate']:.3f} ({stats['wins']}W/{stats['losses']}L/{stats['draws']}D)")

        # Restore original stats
        agent.games_played = original_stats['games_played']
        agent.wins = original_stats['wins']
        agent.losses = original_stats['losses']
        agent.draws = original_stats['draws']

    return results


def main():
    print("Q-Learning Tic-Tac-Toe Training")
    print("1. Train against random player")
    print("2. Train against specific minimax difficulty")
    print("3. Train against self")
    print("4. Evaluate existing model")

    choice = input("Choose training option (1-4): ").strip()

    agent = QLearningAgent(player_char='X', learning_rate=0.1, discount_factor=0.9, epsilon=0.3)
    agent2 = QLearningAgent(player_char='O', learning_rate=0.1, discount_factor=0.9, epsilon=0.3)

    if choice == '1':
        mng.load_model_for_agent(agent)
        train_against_random(agent, num_games=20000)
        mng.save_model_from_agent(agent)

    elif choice == '2':
        print("Choose minimax difficulty:")
        print("1. Easy")
        print("2. Medium")
        print("3. Hard")
        print("4. Perfect")

        diff_choice = input("Enter choice (1-4): ").strip()
        difficulties = {'1': 'easy', '2': 'medium', '3': 'hard', '4': 'perfect'}
        difficulty = difficulties.get(diff_choice, 'medium')

        mng.load_model_for_agent(agent)
        train_against_minimax(agent, difficulty=difficulty, num_games=10000)
        mng.save_model_from_agent(agent)

    elif choice == '3':
        print("\nLoading model for first agent...")
        mng.load_model_for_agent(agent)

        print("\nLoading model for second agent...")
        mng.load_model_for_agent(agent2)

        train_self_play(agent, agent2, num_games=10000, print_interval=1000)

        print("\nSaving model for first agent...")
        mng.save_model_from_agent(agent)

        print("\nSaving model for second agent...")
        mng.save_model_from_agent(agent2)

    elif choice == '4':
        print("Evaluating loaded model...")
        mng.load_model_for_agent(agent)
        evaluate_agent_vs_all_opponents(agent, games_per_opponent=500)

    else:
        print("Invalid choice!")
        return

    stats = agent.get_stats()
    print(f"\nTraining complete!")
    print(f"Games played: {stats['games_played']}")
    print(f"Win rate: {stats['win_rate']:.3f}")
    print(f"Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")
    print(f"Q-table size: {len(agent.q_table)}")


if __name__ == '__main__':
    main()
