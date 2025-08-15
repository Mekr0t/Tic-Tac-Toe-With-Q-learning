import pytest
from src.tictactoe.game_logic import Board


def test_horizontal_win():
    b = Board()
    for i in [1, 2, 3]:
        b.place_char("X", i)
    assert b.check_win() == "X"


def test_draw():
    b = Board()
    # perfect draw pattern
    moves = [1,2,3,5,4,7,6,8,9]
    chars = ["X","O","X","O","X","O","O","X","O"]
    for c, m in zip(chars, moves):
        b.place_char(c, m)
    assert b.check_win() == 0


def test_illegal_move_raises():
    b = Board()
    b.place_char("X", 5)
    with pytest.raises(Exception):  # InvalidMoveError
        b.place_char("O", 5)
