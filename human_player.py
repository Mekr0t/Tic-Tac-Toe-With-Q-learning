import models.model_manager as mng
from game.game_logic import Board
from agents.q_agent import QLearningAgent


def play_vs_agent():
    """Play tic-tac-toe against the trained Q-learning agent"""

    # Load the trained agent
    agent = QLearningAgent(player_char='X')
    if mng.load_model_for_agent(agent) is False:
        return

    print(f"Loaded trained agent with {agent.games_played} games of experience!")
    print(f"Agent's training win rate: {agent.get_win_rate():.3f}")
    print()

    # Choose who plays X and O
    print("Choose your side:")
    print("1. You play X (go first)")
    print("2. You play O (go second)")

    choice = input("Enter choice (1 or 2): ").strip()
    if choice == '1':
        human_char = 'X'
        agent.player_char = 'O'
        agent.opponent_char = 'X'
        human_first = True
    else:
        human_char = 'O'
        agent.player_char = 'X'
        agent.opponent_char = 'O'
        human_first = False

    print(f"\nYou are {human_char}, Agent is {agent.player_char}")
    print("Enter positions 1-9 corresponding to:")
    print("1 | 2 | 3")
    print("4 | 5 | 6")
    print("7 | 8 | 9")
    print()

    # Play multiple games
    human_wins = 0
    agent_wins = 0
    draws = 0

    while True:
        board = Board()
        print("New game!")
        board.print_board()

        # Determine first player
        if human_first:
            current_player = 'human'
        else:
            current_player = 'agent'

        # Game loop
        while board.check_win() is None:
            if current_player == 'human':
                # Human move
                try:
                    move = int(input(f"Your move ({human_char}): "))
                    board.place_char(human_char, move)
                    board.print_board()
                    current_player = 'agent'
                except ValueError:
                    print("Invalid input! Enter a number 1-9.")
                    continue
                except IndexError as e:
                    print(f"Invalid move: {e}")
                    continue
            else:
                # Agent move
                print(f"Agent ({agent.player_char}) is thinking...")
                action = agent.make_move(board, training=False)  # No exploration during play
                if action is not None:
                    print(f"Agent chooses position {action + 1}")
                    board.print_board()
                current_player = 'human'

        # Game ended
        result = board.check_win()
        if result == human_char:
            print("You won! 🎉")
            human_wins += 1
        elif result == agent.player_char:
            print("Agent won! 🤖")
            agent_wins += 1
        else:
            print("It's a draw! 🤝")
            draws += 1

        print(f"\nScore - You: {human_wins}, Agent: {agent_wins}, Draws: {draws}")

        # Play again?
        play_again = input("\nPlay again? (y/n): ").strip().lower()
        if play_again != 'y':
            break

        human_char, agent.player_char, agent.opponent_char = agent.player_char, human_char, agent.player_char
        human_first = not human_first

    print(f"\nFinal Score:")
    print(f"You: {human_wins}")
    print(f"Agent: {agent_wins}")
    print(f"Draws: {draws}")
    print("Thanks for playing!")


def main():
    play_vs_agent()


if __name__ == '__main__':
    main()
