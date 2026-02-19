# SOS Game AI

A Python implementation of the SOS game with multiple AI strategies including Monte Carlo Tree Search (MCTS) and Neural Network-based AI.

## 🎮 What is SOS Game?

SOS is a combinatorial game where players take turns placing either 'S' or 'O' on a grid. The goal is to create the sequence "SOS" (horizontally, vertically, or diagonally). Each time a player creates an SOS pattern, they score a point and get another turn. The player with the most SOS patterns when the board is full wins.

## 🚀 Features

- **Interactive GUI**: Play against AI or watch AI vs AI matches
- **Multiple Game Modes**:
  - Human vs Human
  - Human vs MCTS AI
  - Human vs Neural Network AI
  - MCTS vs Neural Network (watch them battle!)
- **Advanced AI Implementations**:
  - **MCTS with UCB1**: Strategic tree search with heuristic rollouts
  - **Deep Neural Network**: Policy and value network trained via self-play
  - **AlphaZero-style Training**: Combines MCTS with neural network guidance

## 📋 Requirements

```
torch>=2.0.0
numpy>=1.24.0
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/ellaro/SOS_Game.git
cd SOS_Game
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venvsos
# On Windows:
venvsos\Scripts\activate
# On Unix/macOS:
source venvsos/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Play the Game

Run the GUI to play against the AI:
```bash
python gui_game.py
```

Select your preferred game mode from the menu:
- **👤 vs 👤**: Play against another human
- **👤 vs AI (MCTS)**: Play against the MCTS AI (2000 simulations)
- **👤 vs AI (Network)**: Play against the neural network AI
- **🤖 MCTS vs Network**: Watch both AIs compete

### Train the Neural Network

Train a new neural network from scratch:
```bash
python run_training.py
```

Training parameters can be adjusted in [training.py](training.py):
- `num_iterations`: Number of training iterations (default: 100)
- `games_per_iteration`: Self-play games per iteration (default: 50)
- `mcts_simulations`: Simulations per move during training (default: 400)
- `epochs`: Training epochs per iteration (default: 10)

### Test Trained Network

Evaluate the trained network's performance:
```bash
python test_trained_network.py
```

## 🏗️ Project Structure

```
SOS_game/
├── gui_game.py              # Tkinter GUI for interactive play
├── play.py                  # Core game logic and state management
├── mcts.py                  # Monte Carlo Tree Search implementation
├── network.py               # Neural network architecture (policy + value)
├── training.py              # AlphaZero-style training loop
├── run_training.py          # Training script
├── test_trained_network.py # Evaluation script
├── quick_eval.py            # Quick performance testing
├── puct.py                  # PUCT algorithm for network-guided MCTS
├── network_mcts_*.pth       # Saved network weights
├── FIXES_APPLIED.md         # Documentation of critical bug fixes
└── requirements.txt         # Python dependencies
```

## 🧠 AI Architecture

### MCTS AI
The MCTS implementation uses:
- **UCB1 selection** with c=√2 for exploration
- **Heuristic rollout policy** that prefers SOS-creating moves (70% weighted, 30% random)
- **Score-based evaluation** (not just win/loss)
- **Perspective-corrected backpropagation** for proper minimax behavior
- **2000 simulations** per move for strong play

### Neural Network AI
- **Dual-head architecture**: Policy head (move probabilities) + Value head (position evaluation)
- **Residual blocks**: 5 residual layers with batch normalization
- **Input**: 8×8×3 tensor (player positions + current player indicator)
- **Training**: Self-play with MCTS-guided policy improvement
- **Loss**: Combined policy loss (cross-entropy) + value loss (MSE)

## 🐛 Critical Fixes Applied

The MCTS AI had several critical bugs that were fixed:

1. **Perspective Inversion**: Fixed UCB selection to properly convert opponent values to parent perspective
2. **Weak Rollout Policy**: Added heuristic rollout that understands SOS patterns
3. **Binary Evaluation**: Changed from win/loss to score-difference evaluation
4. **Insufficient Search**: Increased from 300 to 2000 simulations
5. **UCB Parameter**: Tuned exploration constant from 1.0 to √2

See [FIXES_APPLIED.md](FIXES_APPLIED.md) for detailed technical explanations.

## 📊 Performance

After fixes, the MCTS AI:
- ✅ Recognizes and prioritizes SOS pattern creation
- ✅ Plans multi-move strategies
- ✅ Adapts to score differences
- ✅ Provides challenging gameplay for human players

The neural network AI (when trained):
- Learns opening patterns and tactical motifs
- Evaluates positions without explicit search
- Can be combined with MCTS for even stronger play

## 🎓 Learning from this Project

This project demonstrates:
- **Game AI fundamentals**: Minimax thinking, tree search, evaluation functions
- **MCTS implementation**: Selection, expansion, simulation, backpropagation
- **Deep RL**: Self-play training, policy gradients, value function approximation
- **AlphaZero concepts**: Combining neural networks with tree search
- **Software engineering**: Clean code structure, bug fixing, documentation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- GUI enhancements (themes, animations, move history)
- Additional AI strategies (minimax, reinforcement learning)
- Network architecture experiments (attention, transformers)
- Opening book and endgame databases
- Online multiplayer support

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Ella**
- GitHub: [@ellaro](https://github.com/ellaro)

## 🙏 Acknowledgments

- Inspired by AlphaZero's approach to game AI
- MCTS algorithm based on classic UCT/UCB1 literature
- Neural network architecture influenced by modern deep RL research

---

**Enjoy playing SOS! 🎮**

If you find this project helpful, please give it a ⭐ on GitHub!
