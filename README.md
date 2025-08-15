# T3-Qlearn 🎮

> A minimal yet production-ready **Q-learning Tic-Tac-Toe agent** in pure Python.

[![CI](https://github.com/Mekr0t/Tic-Tac-Toe-With-Q-learning/workflows/CI/badge.svg)](https://github.com/Mekr0t/Tic-Tac-Toe-With-Q-learning/actions)

---

## Features
- Tabular Q-learning with ε-greedy exploration  
- Self-play, random, and minimax (easy→perfect) training schedules  
- YAML configuration – no code edits for hyper-parameter sweeps  
- Disk-backed replay buffer – train millions of games on a laptop  
- 100 % type-checked, fully tested (`pytest`)  
- Install once, run anywhere: `t3-train` & `t3-play` CLI tools  

---

## Quick Start
```bash
git clone https://github.com/Mekr0t/Tic-Tac-Toe-With-Q-learning.git
cd Tic-Tac-Toe-With-Q-learning
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Train 20 000 games vs a random opponent:
```bash
t3-train
```

### Play against the trained agent in the console:

```bash
t3-play
```

---

## Configuration
Edit ```config.yaml``` to tweak learning rate, ε schedule, buffer size, etc.
No code changes needed.

---

## Development
Run the test-suite:
```bash
pytest -q
```
Run static type check:
```bash
mypy src --strict
```

---

## File Layout
```
Tic-Tac-Toe-With-Q-learning
├── src/tictactoe/      # all source code
├── tests/              # pytest suite
├── models/             # *.pkl files (git-ignored)
├── config.yaml         # hyper-parameters
└── pyproject.toml      # modern packaging
```
