# 📂 Files Overview

Quick reference for all files in the SOS Game project.

## 📚 Documentation Files (Start Here!)

| File | Purpose | Read This If... |
|------|---------|----------------|
| **README.md** | Main project documentation | You're new to the project |
| **MCTS_EXPLAINED.md** | Deep dive into MCTS algorithm | You want to understand how MCTS works |
| **ARCHITECTURE.md** | Code structure and design | You want to understand the codebase |
| **DOCUMENTATION_GUIDE.md** | Navigation guide | You're not sure where to start |
| **FILES_OVERVIEW.md** | This file! | You want a quick file reference |

## 🎮 Core Game Files

### `main.py` - Game Engine
**What it does:**
- Implements SOS game rules
- Manages board state (8×8 grid)
- Detects SOS patterns in all directions
- Handles move validation and execution

**Key Classes:**
- `SOSGame` - Main game state and logic

**When to use:**
```python
from main import SOSGame
game = SOSGame()
game.make_move((0, 0, 'S'))  # Place 'S' at (0,0)
```

## 🤖 AI Player Files

### `mcts.py` - Pure Monte Carlo Tree Search
**What it does:**
- Implements classic MCTS algorithm
- Uses random rollouts for position evaluation
- Balances exploration vs exploitation with UCB1

**Key Classes:**
- `MCTSNode` - Tree node with visit counts and values
- `MCTSPlayer` - AI that uses MCTS to choose moves

**When to use:**
```python
from mcts import MCTSPlayer
ai = MCTSPlayer(num_simulations=100)
move = ai.get_move(game)
```

**Strength:** Good without neural network, but slower

### `puct.py` - Neural Network-Guided MCTS
**What it does:**
- Enhanced MCTS using neural network predictions
- Faster than pure MCTS (no random rollouts)
- Uses network priors to guide search

**Key Classes:**
- `PUCTNode` - Tree node with network priors
- `PUCTPlayer` - AI that combines MCTS + neural network

**When to use:**
```python
from puct import PUCTPlayer
from network import GameNetwork
net = GameNetwork.load("trained_model.pth")
ai = PUCTPlayer(net, num_simulations=100)
move = ai.get_move(game)
```

**Strength:** Stronger and faster than pure MCTS

## 🧠 Neural Network Files

### `network.py` - Deep Learning Model
**What it does:**
- Implements dual-head neural network
- Policy head: predicts good moves
- Value head: evaluates positions
- PyTorch-based architecture

**Key Classes:**
- `GameNetwork` - Neural network model
- `NetworkTrainer` - Training helper

**Architecture:**
```
Input (131 values)
    ↓
Shared Layers (256 → 256 → 256)
    ↓
├─→ Policy Head (128 move probabilities)
└─→ Value Head (win probability)
```

**When to use:**
```python
from network import GameNetwork
net = GameNetwork()
policy, value = net.predict(game)
```

### `training.py` - Self-Play Training
**What it does:**
- Generates training data through self-play
- Trains neural network on game outcomes
- Implements AlphaZero-style training loop

**Key Classes:**
- `SelfPlayTrainer` - Manages training pipeline

**Training Process:**
```
1. Generate games using current network
2. Collect (state, move_probs, outcome) data
3. Train network on this data
4. Repeat with improved network
```

**When to use:**
```python
from training import SelfPlayTrainer
trainer = SelfPlayTrainer()
trainer.generate_puct_games(num_games=50)
trainer.train_network(epochs=10)
```

### `run_training.py` - Training Script
**What it does:**
- Complete training pipeline
- Bootstraps with MCTS games
- Continues with PUCT self-play
- Saves trained models

**When to use:**
```bash
python run_training.py
```

**What it does:**
1. Generate 100 MCTS games (no network needed)
2. Train initial network
3. Generate 50 PUCT games (using trained network)
4. Train network more
5. Save `network_mcts_YYYYMMDD_HHMMSS.pth`

## 🎨 User Interface Files

### `gui_game.py` - Graphical Interface
**What it does:**
- Beautiful Tkinter-based GUI
- Multiple game modes (Human vs AI, Human vs Human)
- Visual board with 8×8 button grid
- Real-time score updates
- Threaded AI (doesn't freeze UI)

**When to use:**
```bash
python gui_game.py
```

**Features:**
- Play against trained AI
- Play against MCTS AI
- Two-player mode
- Letter selection (S or O)
- Colorful interface

### `play_vs_network.py` - Terminal Play
**What it does:**
- Command-line interface
- Play against trained network
- Text-based board display

**When to use:**
```bash
python play_vs_network.py
```

**Good for:** Quick testing without GUI

### `play.py` - Basic Play Script
**What it does:**
- Simple terminal gameplay
- Human vs Human or Human vs AI

**When to use:**
```bash
python play.py
```

## 🧪 Test Files

### `test_trained_network.py` - Network Testing
**What it does:**
- Tests trained network quality
- Plays games to evaluate strength

**When to use:**
```bash
python test_trained_network.py
```

## 📊 Data Files

### `network_mcts_*.pth` - Trained Models
**What it is:**
- Saved neural network weights
- Created by training
- Can be loaded and used

**Usage:**
```python
net = GameNetwork.load("network_mcts_20260203_223556.pth")
```

### `training_data_mcts_*.pkl` - Training Data
**What it is:**
- Saved game data (state, policy, value) tuples
- Used for training
- Pickle format

## 🔧 Configuration Files

### `requirements/` - Dependencies
**What it contains:**
- Python package requirements
- PyTorch, NumPy, etc.

**Usage:**
```bash
pip install -r requirements/requirements.txt
```

## 📁 Directory Structure

```
SOS_Game/
├── 📚 Documentation
│   ├── README.md                    ← Start here!
│   ├── MCTS_EXPLAINED.md           ← How MCTS works
│   ├── ARCHITECTURE.md             ← Code design
│   ├── DOCUMENTATION_GUIDE.md      ← Where to read
│   └── FILES_OVERVIEW.md           ← This file
│
├── 🎮 Game Engine
│   └── main.py                      ← SOS game rules
│
├── 🤖 AI Players
│   ├── mcts.py                      ← Pure MCTS
│   └── puct.py                      ← MCTS + Neural Net
│
├── 🧠 Neural Network
│   ├── network.py                   ← Deep learning model
│   ├── training.py                  ← Self-play training
│   └── run_training.py              ← Training script
│
├── 🎨 User Interfaces
│   ├── gui_game.py                  ← Tkinter GUI
│   ├── play_vs_network.py          ← Terminal play
│   └── play.py                      ← Basic play
│
├── 🧪 Testing
│   └── test_trained_network.py     ← Test network
│
├── 💾 Saved Data
│   ├── network_mcts_*.pth          ← Trained models
│   └── training_data_mcts_*.pkl    ← Training data
│
└── 🔧 Configuration
    └── requirements/                ← Dependencies
```

## 🎯 Quick Start Guide

### Just Want to Play?
```bash
python gui_game.py
```

### Want to Understand MCTS?
1. Read `README.md` → "How Does MCTS Work?"
2. Read `MCTS_EXPLAINED.md`
3. Look at `mcts.py` with comments

### Want to Train an AI?
```bash
python run_training.py
```

### Want to Modify the Code?
1. Read `ARCHITECTURE.md`
2. Find relevant file in this overview
3. Read that file's comments

## 📈 Complexity Levels

### Beginner
- `README.md` - Overview
- `gui_game.py` - Play the game
- `main.py` - Understand game rules

### Intermediate  
- `MCTS_EXPLAINED.md` - Algorithm details
- `mcts.py` - MCTS implementation
- `play_vs_network.py` - Test AI

### Advanced
- `ARCHITECTURE.md` - Full design
- `puct.py` - Advanced search
- `network.py` - Deep learning
- `training.py` - Self-play training

## 💡 File Relationships

```
main.py (Game Engine)
    ↓ used by
├─→ mcts.py (Pure MCTS AI)
├─→ puct.py (MCTS + NN)
│       ↓ uses
│   network.py (Neural Net)
│       ↓ trained by
│   training.py (Self-Play)
│       ↓ executed by
│   run_training.py
│
└─→ gui_game.py (User Interface)
    play_vs_network.py (Terminal UI)
    play.py (Basic UI)
```

## 🔍 Finding What You Need

| I want to... | Use this file... |
|-------------|------------------|
| Play the game | `gui_game.py` |
| Understand MCTS | `MCTS_EXPLAINED.md` + `mcts.py` |
| Train an AI | `run_training.py` |
| Modify game rules | `main.py` |
| Change AI behavior | `mcts.py` or `puct.py` |
| Adjust network architecture | `network.py` |
| Change training process | `training.py` |
| Understand code structure | `ARCHITECTURE.md` |

---

**Now you know what every file does! 📂✨**

Start exploring from [README.md](README.md)!
