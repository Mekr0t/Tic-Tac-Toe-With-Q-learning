import random


class RandomPlayer:
    def __init__(self, player_char: str):
        self.player_char = player_char

    def make_move(self, board):
        available_moves = [i for i, cell in enumerate(board.get_board()) if cell == ' ']
        if available_moves:
            move = random.choice(available_moves)
            board.place_char(self.player_char, move + 1)
            return move
        return None

