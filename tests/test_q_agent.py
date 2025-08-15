import pytest
from game_logic import Board
from q_agent import QLearningAgent


def test_bellman_update():
    agent = QLearningAgent()
    state = (0,0,0,0,0,0,0,0,0)      # empty board
    action = 4                        # centre
    next_state = (0,0,0,0,1,0,0,0,0)  # after X plays centre
    agent.update_q_value(state, action, reward=1.0, next_state=next_state)
    key = (state, action)
    assert agent.q_table[key] == pytest.approx(0.1)   # 1st update
