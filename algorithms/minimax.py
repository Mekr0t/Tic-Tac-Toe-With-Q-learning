from game.game_logic import Board
import math


class MinimaxPlayer:
    def __init__(self, player_char, difficulty='perfect'):
        self.player_char = player_char.upper()
        self.opponent_char = 'O' if player_char.upper() == 'X' else 'X'
        self.difficulty = difficulty

        self.max_depth = {
            'random': 0,
            'easy': 2,
            'medium': 5,
            'hard': 7,
            'perfect': 9
        }[difficulty]

        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def evaluate_board(self, board):
        result = board.check_win()

        if result == self.player_char:
            return 10
        elif result == self.opponent_char:
            return -10
        elif result == 0:
            return 0
        else:
            return 0

    def minimax(self, board, depth, is_maximizing, alpha=-math.inf, beta=math.inf):
        """
        Minimax algorithm with alpha-beta pruning

        Args:
            board: Current board state
            depth: Current search depth
            is_maximizing: True if maximizing player's turn
            alpha: Alpha value for pruning
            beta: Beta value for pruning

        Returns:
            Best score for current position
        """
        score = self.evaluate_board(board)

        if score != 0 or depth >= self.max_depth or board.check_win() is not None:
            return score

        available_moves = [i for i, cell in enumerate(board.get_board()) if cell == ' ']

        if not available_moves:
            return 0

        if is_maximizing:
            max_eval = -math.inf

            for move in available_moves:
                board.place_char(self.player_char, move + 1)

                eval_score = self.minimax(board, depth + 1, False, alpha, beta)

                board.undo_move(move)

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break

            return max_eval

        else:
            min_eval = math.inf

            for move in available_moves:
                board.place_char(self.opponent_char, move + 1)

                eval_score = self.minimax(board, depth + 1, True, alpha, beta)

                board.undo_move(move)

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break

            return min_eval

    def get_best_move(self, board):
        available_moves = [i for i, cell in enumerate(board.get_board()) if cell == ' ']

        if not available_moves:
            return None

        best_move = None
        best_score = -math.inf

        for move in available_moves:
            board.place_char(self.player_char, move + 1)

            score = self.minimax(board, 0, False)

            board.undo_move(move)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def make_move(self, board):
        move = self.get_best_move(board)
        if move is not None:
            board.place_char(self.player_char, move + 1)
        return move

    def update_stats(self, game_result):
        self.games_played += 1
        if game_result == self.player_char:
            self.wins += 1
        elif game_result == self.opponent_char:
            self.losses += 1
        elif game_result == 0:
            self.draws += 1

    def get_win_rate(self):
        if self.games_played == 0:
            return 0
        return self.wins / self.games_played

    def get_stats(self):
        return {
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.get_win_rate(),
            'difficulty': self.difficulty
        }

    def set_player_char(self, char):
        """Set the player's character and update opponent character"""
        self.player_char = char
        self.opponent_char = 'O' if char == 'X' else 'X'


class ImperfectMinimaxPlayer(MinimaxPlayer):
    def __init__(self, player_char, mistake_probability=0.1):
        super().__init__(player_char, difficulty='perfect')
        self.mistake_probability = mistake_probability

    def make_move(self, board):
        import random

        if random.random() < self.mistake_probability:
            available_moves = [i for i, cell in enumerate(board.get_board()) if cell == ' ']
            if available_moves:
                move = random.choice(available_moves)
                board.place_char(self.player_char, move + 1)
                return move

        return super().make_move(board)


def test_minimax_performance():
    from algorithms.random_player import RandomPlayer
    import time

    print("Testing Minimax Player Performance...")

    difficulties = ['easy', 'medium', 'hard', 'perfect']

    for difficulty in difficulties:
        print(f"\nTesting {difficulty} difficulty:")

        minimax_player = MinimaxPlayer('X', difficulty=difficulty)
        random_player = RandomPlayer('O')

        start_time = time.time()

        for _ in range(100):
            board = Board()
            current_player = minimax_player

            while board.check_win() is None:
                if current_player == minimax_player:
                    minimax_player.make_move(board)
                    current_player = random_player
                else:
                    random_player.make_move(board)
                    current_player = minimax_player

            result = board.check_win()
            minimax_player.update_stats(result)

        end_time = time.time()
        stats = minimax_player.get_stats()

        print(f"Win rate: {stats['win_rate']:.3f}")
        print(f"Games: {stats['wins']}W/{stats['losses']}L/{stats['draws']}D")
        print(f"Average time per game: {(end_time - start_time) / 100:.4f}s")


def test_minimax_vs_minimax():
    player1 = MinimaxPlayer('X', difficulty='hard')
    player2 = MinimaxPlayer('O', difficulty='hard')
    wins_x = 0
    draws = 0
    for _ in range(100):
        board = Board()
        current_player = player1
        while board.check_win() is None:
            current_player.make_move(board)
            current_player = player2 if current_player == player1 else player1
        result = board.check_win()
        if result == 'X':
            wins_x += 1
        elif result == 0:
            draws += 1
    print(f"Minimax vs Minimax: X Wins: {wins_x}, Draws: {draws}, O Wins: {100 - wins_x - draws}")


if __name__ == '__main__':
    test_minimax_performance()
    test_minimax_vs_minimax()

