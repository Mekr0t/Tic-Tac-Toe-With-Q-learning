import random
from game.game_logic import Board
from agents.q_agent import QLearningAgent


class RandomPlayer:
    """Simple random player for training opponent"""

    def __init__(self, player_char):
        self.player_char = player_char

    def make_move(self, board):
        available_moves = [i for i, cell in enumerate(board.get_board()) if cell == ' ']
        if available_moves:
            move = random.choice(available_moves)
            board.place_char(self.player_char, move + 1)  # Convert to 1-9 indexing
            return move
        return None


def train_against_random(agent, num_games=10000, print_interval=1000):
    """Train the Q-learning agent against a random player"""
    print(f"Training {agent.player_char} agent against random player for {num_games} games...")

    opponent = RandomPlayer('O' if agent.player_char == 'X' else 'X')

    for game in range(num_games):
        board = Board()
        game_states = []  # Store (state, action) pairs for learning

        # Randomly decide who goes first
        if random.random() < 0.5:
            current_player = agent
            other_player = opponent
        else:
            current_player = opponent
            other_player = agent

        # Play the game
        while board.check_win() is None:
            if current_player == agent:
                # Store state before agent's move
                state_before = board.get_numeric_board().copy()
                action = agent.make_move(board, training=True)
                if action is not None:
                    game_states.append((state_before, action))
            else:
                other_player.make_move(board)

            # Switch players
            current_player, other_player = other_player, current_player

        # Game ended, update Q-values
        game_result = board.check_win()
        agent.update_stats(game_result)

        # Backpropagate rewards through all agent's moves
        final_reward = agent.get_reward(game_result)

        for i, (state, action) in enumerate(reversed(game_states)):
            # Give immediate reward for final move, discounted for earlier moves
            reward = final_reward * (agent.discount_factor ** i)

            # Get next state (current board state)
            if i == 0:  # Most recent move
                next_state = board.get_numeric_board()
            else:  # Earlier moves
                next_state = game_states[len(game_states) - i][0]

            agent.update_q_value(state, action, reward, next_state)

        # Decay exploration rate
        if game % 100 == 0:
            agent.decay_epsilon()

        # Print progress
        if (game + 1) % print_interval == 0:
            stats = agent.get_stats()
            print(f"Game {game + 1}: Win Rate: {stats['win_rate']:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Q-table size: {len(agent.q_table)}")


def train_self_play(agent1, agent2, num_games=5000, print_interval=500):
    """Train two agents against each other"""
    print(f"Training agents against each other for {num_games} games...")

    for game in range(num_games):
        board = Board()
        game_states_1 = []
        game_states_2 = []

        # X always goes first
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

        # Update both agents
        game_result = board.check_win()
        agent1.update_stats(game_result)
        agent2.update_stats(game_result)

        # Update Q-values for both agents
        for agent, states in [(agent1, game_states_1), (agent2, game_states_2)]:
            final_reward = agent.get_reward(game_result)

            for i, (state, action) in enumerate(reversed(states)):
                reward = final_reward * (agent.discount_factor ** i)
                agent.update_q_value(state, action, reward, board.get_numeric_board())

        # Decay exploration
        if game % 100 == 0:
            agent1.decay_epsilon()
            agent2.decay_epsilon()

        # Print progress
        if (game + 1) % print_interval == 0:
            stats1 = agent1.get_stats()
            stats2 = agent2.get_stats()
            print(f"Game {game + 1}: {agent1.player_char} Win Rate: {stats1['win_rate']:.3f}, "
                  f"{agent2.player_char} Win Rate: {stats2['win_rate']:.3f}")


def main():
    """Main training function"""
    # Create and train agent
    agent = QLearningAgent(player_char='X', learning_rate=0.1, discount_factor=0.9, epsilon=0.3)
    agent2 = QLearningAgent(player_char='O', learning_rate=0.1, discount_factor=0.9, epsilon=0.3)

    # Try to load existing model
    if agent.load_model('tic_tac_toe_model.pkl'):
        print("Loaded existing model. Continuing training...")
    else:
        print("Starting fresh training...")

    # Train against random player
    train_self_play(agent, agent2, num_games=15000)

    # Save the trained model
    agent.save_model('tic_tac_toe_model.pkl')

    # Print final stats
    stats = agent.get_stats()
    print(f"\nTraining complete!")
    print(f"Games played: {stats['games_played']}")
    print(f"Win rate: {stats['win_rate']:.3f}")
    print(f"Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")
    print(f"Q-table size: {len(agent.q_table)}")


if __name__ == '__main__':
    main()