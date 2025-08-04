from agents.q_agent import QLearningAgent


def save(agent):
    print("Do you want to save this model")
    print("1. Yes")
    print("2. No")

    choice = input("Choose training option (1-2): ").strip()

    if choice == '1':
        path = input("Enter name of model: ").strip()
        path = "models\\" + path + '.pkl'
        agent.save_model(path)
    if choice == '2':
        print("Model was not saved!")


def load(agent):
    agent.load_model('path')
