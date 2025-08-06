import os
import datetime


MODEL_DIR = 'models/'


def save_model_from_agent(agent):
    choice = input("\nDo you want to save this model (y/n): ").strip()

    if choice.upper() == 'Y' or choice.upper() == 'YES':
        filename = input("Enter name of model: ").strip()

        if filename == '':
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"model_{timestamp}.pkl"
        else:
            filename = f"{filename}.pkl"

        full_path = os.path.join(MODEL_DIR, filename)
        os.makedirs(MODEL_DIR, exist_ok=True)

        agent.save_model(full_path)

    if choice.upper() == 'N' or choice.upper() == 'NO':
        print("\nModel was not saved!\n")


def load_model_for_agent(agent):
    if not os.path.exists(MODEL_DIR):
        print(f"\nModel directory '{MODEL_DIR}' does not exist.")
        return

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    if not model_files:
        print(f"\nNo model files found in '{MODEL_DIR}'.")
        return

    print("\nAvailable models:")
    for i, file in enumerate(model_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("\nEnter the number of the model to load (0 to skip): "))
            if choice == 0:
                print("\nNo model loaded. Agent will start fresh.")
                return False
            elif 1 <= choice <= len(model_files):
                selected_file = model_files[choice - 1]
                full_path = os.path.join(MODEL_DIR, selected_file)
                if agent.load_model(full_path):
                    print(f"\nSuccessfully loaded model: {selected_file}")
                    return True
                else:
                    print(f"\nFailed to load model: {selected_file}")
                return False
            else:
                print(f"\nPlease enter a number between 0 and {len(model_files)}.")
        except ValueError:
            print("\nPlease enter a valid number.")
