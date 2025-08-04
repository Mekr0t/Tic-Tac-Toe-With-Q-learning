import random
import pickle
import os
from collections import defaultdict
from game.game_logic import Board


class QLearningAgent:
    def __init__(self, player_char='X', learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.player_char = player_char
        self.opponent_char = 'O' if player_char == 'X' else 'X'

        # Q-learning parameters
        self.learning_rate = learning_rate  # α - how much to update Q-values
        self.discount_factor = discount_factor  # γ - importance of future rewards
        self.epsilon = epsilon  # exploration rate

        # Q-table: defaultdict returns 0 for unseen state-action pairs
        self.q_table = defaultdict(float)

        # For tracking learning progress
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def get_state_key(self, board_state):
        """Convert board state to a string key for Q-table"""
        return ''.join(map(str, board_state))

    def get_available_actions(self, board):
        """Get list of valid moves (0-8 indexing)"""
        return [i for i, cell in enumerate(board.get_board()) if cell == ' ']

    def choose_action(self, board, training=True):
        """Choose action using epsilon-greedy strategy"""
        available_actions = self.get_available_actions(board)

        if not available_actions:
            return None

        # During training, use epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)

        # Otherwise, choose the best action based on Q-values
        state_key = self.get_state_key(board.get_numeric_board())
        best_action = None
        best_value = float('-inf')

        for action in available_actions:
            q_value = self.q_table[f"{state_key}_{action}"]
            if q_value > best_value:
                best_value = q_value
                best_action = action

        # If all Q-values are equal (or 0), choose randomly
        if best_action is None:
            best_action = random.choice(available_actions)

        return best_action

    def get_reward(self, game_result):
        """Calculate reward based on game outcome"""
        if game_result == self.player_char:
            return 1.0  # Win
        elif game_result == self.opponent_char:
            return -1.0  # Loss
        elif game_result == 0:
            return 0.5  # Draw (slightly positive to encourage not losing)
        else:
            return 0.0  # Game ongoing

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        current_q = self.q_table[f"{state_key}_{action}"]

        # Find maximum Q-value for next state
        next_board = Board()
        next_board._board = [' ' if x == 0 else ('X' if x == 1 else 'O') for x in next_state]
        next_actions = self.get_available_actions(next_board)

        if next_actions:
            max_next_q = max([self.q_table[f"{next_state_key}_{a}"] for a in next_actions])
        else:
            max_next_q = 0

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[f"{state_key}_{action}"] = new_q

    def make_move(self, board, training=True):
        """Make a move and return the action taken"""
        action = self.choose_action(board, training)
        if action is not None:
            board.place_char(self.player_char, action + 1)  # Convert to 1-9 indexing
        return action

    def update_stats(self, game_result):
        """Update win/loss/draw statistics"""
        self.games_played += 1
        if game_result == self.player_char:
            self.wins += 1
        elif game_result == self.opponent_char:
            self.losses += 1
        elif game_result == 0:
            self.draws += 1

    def get_win_rate(self):
        """Calculate current win rate"""
        if self.games_played == 0:
            return 0
        return self.wins / self.games_played

    def get_stats(self):
        """Get training statistics"""
        return {
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.get_win_rate()
        }

    def save_model(self, filename):
        """Save the Q-table and stats to file"""
        data = {
            'q_table': dict(self.q_table),
            'stats': self.get_stats(),
            'parameters': {
                'player_char': self.player_char,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Load Q-table and stats from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(float, data['q_table'])
                stats = data['stats']
                self.games_played = stats['games_played']
                self.wins = stats['wins']
                self.losses = stats['losses']
                self.draws = stats['draws']
            print(f"Model loaded from {filename}")
            return True
        return False

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Gradually reduce exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)