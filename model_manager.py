"""
model_manager.py
Utilities for saving / loading / deleting pickled agents.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from q_agent import QLearningAgent  # adjust if your agent lives elsewhere

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
MODEL_DIR = "models"
PKL_EXT = ".pkl"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _model_dir() -> str:
    """Ensure directory exists and return absolute path."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return MODEL_DIR


def _list_models() -> List[str]:
    """Return all *.pkl files in MODEL_DIR (without paths)."""
    return [f for f in os.listdir(_model_dir()) if f.endswith(PKL_EXT)]


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def save_model_from_agent(agent: QLearningAgent) -> None:
    """Prompt user to save the given agent."""
    choice = input("\nDo you want to save this model? [y/n]: ").strip().lower()
    if choice not in {"y", "yes"}:
        print("\nModel was not saved!\n")
        return

    filename = input("Enter model name (leave blank for auto): ").strip()
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"model_{timestamp}"
    filename += PKL_EXT

    full_path = os.path.join(_model_dir(), filename)
    agent.save_model(full_path)


def load_model_for_agent(agent: QLearningAgent) -> bool:
    """Prompt user to load a model into the given agent."""
    files = _list_models()
    if not files:
        print(f"\nNo model files found in '{MODEL_DIR}'.")
        return False

    print("\nAvailable models:")
    for idx, name in enumerate(files, 1):
        print(f"{idx}. {name}")

    try:
        choice = int(input("\nEnter number to load (0 to skip): "))
    except ValueError:
        print("Please enter a valid number.")
        return False

    if choice == 0:
        print("\nNo model loaded. Agent will start fresh.")
        return False

    if 1 <= choice <= len(files):
        full_path = os.path.join(_model_dir(), files[choice - 1])
        success = agent.load_model(full_path)
        if success:
            print(f"\nLoaded model: {files[choice - 1]}")
        return success

    print(f"\nPlease enter a number between 0 and {len(files)}.")
    return False


def delete_model(model_name: Optional[str] = None) -> None:
    """Delete a specific model or prompt the user to choose one."""
    files = _list_models()
    if not files:
        print(f"No model files found in '{MODEL_DIR}'.")
        return

    if model_name is None:
        print("Available models:")
        for idx, name in enumerate(files, 1):
            print(f"{idx}. {name}")

        try:
            choice = int(input("Enter number to delete (0 to cancel): "))
        except ValueError:
            print("Please enter a valid number.")
            return

        if choice == 0:
            print("Deletion canceled.")
            return

        if 1 <= choice <= len(files):
            model_name = files[choice - 1]
        else:
            print(f"Please enter a number between 0 and {len(files)}.")
            return

    full_path = os.path.join(_model_dir(), model_name)
    if not os.path.exists(full_path):
        print(f"Model '{model_name}' not found.")
        return

    confirm = input(f"Delete {model_name}? [y/n]: ").strip().lower()
    if confirm == "y":
        try:
            os.remove(full_path)
            print(f"Deleted model: {model_name}")
        except Exception as e:
            print(f"Error deleting model: {e}")
    else:
        print("Deletion canceled.")
