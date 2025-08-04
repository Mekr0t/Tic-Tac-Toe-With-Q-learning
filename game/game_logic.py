class Board:
    def __init__(self):
        self._board = self.new_board()

    @staticmethod
    def new_board():
        return [" " for _ in range(9)]

    def __str__(self):
        return str(self._board)

    def print_board(self):
        print("-------------")
        print(f"| {self._board[0]} | {self._board[1]} | {self._board[2]} |")
        print("-------------")
        print(f"| {self._board[3]} | {self._board[4]} | {self._board[5]} |")
        print("-------------")
        print(f"| {self._board[6]} | {self._board[7]} | {self._board[8]} |")
        print("-------------")

    def place_char(self, char: str, place: int):
        self._board[place-1] = char.upper()

    def check_win(self):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        for line in lines:
            if self._board[line[0]] == self._board[line[1]] == self._board[line[2]] != " ":
                return self._board[line[0]]

        if self.is_full():
            return 0

        return None

    def get_board(self):
        return self._board

    def get_numeric_board(self):
        numbers = {"X": 1, "O": -1, " ": 0}
        return [numbers[x] for x in self._board]

    def undo_move(self, move):
        self._board[move] = " "

    def is_full(self):
        return False if " " in self._board else True
