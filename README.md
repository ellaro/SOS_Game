# 🎮 SOS Game with AI

A Python implementation of the SOS game with multiple AI players using Monte Carlo Tree Search (MCTS) and neural network-guided PUCT (Predictor + Upper Confidence bounds applied to Trees) algorithms.

## 📖 Table of Contents
- [What is SOS Game?](#what-is-sos-game)
- [What Does This Project Do?](#what-does-this-project-do)
- [How Does MCTS Work?](#how-does-mcts-work)
- [How Does PUCT Work?](#how-does-puct-work)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training Your Own AI](#training-your-own-ai)

## 🎯 What is SOS Game?

SOS is a two-player paper-and-pencil game where players take turns writing either 'S' or 'O' in empty cells of a grid. The goal is to create the sequence "SOS" (horizontally, vertically, or diagonally). Each time a player creates an SOS pattern, they score a point and get another turn. The player with the most points when the board is full wins.

**Board**: 8x8 grid  
**Players**: 2 (Player 0 and Player 1)  
**Actions**: Place 'S' or 'O' in any empty cell  
**Scoring**: +1 point for each SOS pattern created  

## 🤖 What Does This Project Do?

This project provides:

1. **Core Game Engine** (`main.py`)
   - Complete SOS game implementation with all rules
   - Board state management
   - SOS pattern detection in all 8 directions
   - Legal move generation

2. **AI Players**
   - **MCTS Player** (`mcts.py`): Pure Monte Carlo Tree Search AI
   - **PUCT Player** (`puct.py`): Neural network-guided search AI
   - Both use advanced search algorithms to find optimal moves

3. **Neural Network** (`network.py`)
   - Deep learning model that learns to evaluate positions
   - Dual-head architecture: policy (what move to play) + value (who's winning)
   - Trained through self-play reinforcement learning

4. **Training Pipeline** (`training.py`, `run_training.py`)
   - Self-play game generation
   - Network training from game data
   - Continuous improvement through iteration

5. **Graphical User Interface** (`gui_game.py`)
   - Beautiful Tkinter-based GUI
   - Play against AI or another human
   - Visual board with real-time score updates

## 🎲 How Does MCTS Work?

Monte Carlo Tree Search (MCTS) is a powerful AI algorithm that explores possible game moves by simulating random games. It's the same algorithm used in AlphaGo!

### The Four Phases of MCTS

MCTS repeats these four steps for many simulations (iterations):

```
┌─────────────────────────────────────────────────┐
│  1. SELECTION                                   │
│  Start at root, use UCB1 to pick best child    │
│  Continue until you reach an unexplored node   │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  2. EXPANSION                                   │
│  Add a new child node for an untried move      │
│  (Prioritizes moves that create SOS patterns)  │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  3. SIMULATION (Rollout)                        │
│  From the new node, play random moves until    │
│  the game ends to see who wins                 │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  4. BACKPROPAGATION                             │
│  Update all nodes in the path with the result  │
│  Wins increase value, losses decrease it       │
└─────────────────────────────────────────────────┘
```

### Key Components

#### 1. **Selection (UCB1 Formula)**
```
UCB1 = (wins/visits) + c × sqrt(ln(parent_visits) / visits)
         ⬆ Exploitation  ⬆ Exploration
```

- **Exploitation**: Favor moves that have worked well (high win rate)
- **Exploration**: Try moves we haven't explored much
- **c parameter**: Controls the balance (default: 1.4)

#### 2. **Expansion**
When we reach a node that hasn't been fully explored:
- Pick an untried move
- Create a new child node for that move
- **Smart heuristic**: Prioritize moves that immediately create SOS patterns

#### 3. **Simulation (Rollout)**
From the new position:
- Play completely random moves
- Continue until the game ends
- Record the final result (who won and final scores)

#### 4. **Backpropagation**
Update statistics back up the tree:
```python
self.visits += 1
self.value_sum += result_value
```
Each node remembers:
- How many times it was visited
- Sum of all game outcomes from this position

### Example MCTS Tree Growth

```
Initial State (Root)
        │
        ├─── Move: (0,0,'S') - visits: 0
        ├─── Move: (0,0,'O') - visits: 0
        └─── Move: (0,1,'S') - visits: 0

After 100 Simulations:
        │
        ├─── Move: (0,0,'S') - visits: 45, value: 0.62
        ├─── Move: (0,0,'O') - visits: 30, value: 0.51
        └─── Move: (0,1,'S') - visits: 25, value: 0.48
              │
              ├─── Move: (0,2,'O') - visits: 12
              └─── Move: (1,0,'S') - visits: 13
```

**Final Decision**: Pick the move with the most visits (most robust choice)

### Why MCTS is Powerful

1. **No need for game evaluation**: Just simulates to the end
2. **Anytime algorithm**: Can be stopped early and still give a good answer
3. **Focuses on promising moves**: Doesn't waste time on bad moves
4. **Self-improving**: More simulations = better decisions

## 🧠 How Does PUCT Work?

PUCT (Predictor + Upper Confidence bounds applied to Trees) is an enhanced version of MCTS that uses a neural network to guide the search. This is what AlphaZero uses!

### Key Differences from MCTS

| Feature | MCTS | PUCT |
|---------|------|------|
| Move selection | UCB1 formula | PUCT formula with neural network priors |
| Position evaluation | Random rollout to end | Neural network value prediction |
| Prior knowledge | None | Network provides move probabilities |
| Speed | Slower (full rollouts) | Faster (network evaluation) |

### PUCT Formula

```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
            ⬆             ⬆
        Exploitation  Exploration (guided by neural net)
```

Where:
- **Q(s,a)**: Average value of taking action 'a' from state 's'
- **P(s,a)**: Neural network's predicted probability for this move
- **N(s)**: Number of times parent node was visited
- **N(s,a)**: Number of times this action was tried
- **c_puct**: Exploration constant (default: 1.0)

### The Neural Network

The network has two outputs:

1. **Policy Head** (128 outputs)
   - Predicts which moves are most promising
   - Uses these as "prior probabilities" P(s,a)
   - Guides PUCT to explore good moves first

2. **Value Head** (1 output)
   - Predicts who's winning from current position
   - Value in range [-1, 1]
   - Replaces random rollouts with instant evaluation

### Training Loop (AlphaZero-style)

```
1. Generate self-play games using current network
   └─> PUCT uses network to guide search
   └─> Record: (position, PUCT move probabilities, game outcome)

2. Train network on collected data
   └─> Policy learns from PUCT's move choices
   └─> Value learns from actual game outcomes

3. Repeat: Better network → Better self-play → Better training data
```

This creates a **positive feedback loop** where the AI continuously improves itself!

## 🏗️ Project Architecture

```
SOS_Game/
│
├── main.py                 # Core game logic
│   ├── SOSGame             # Game state and rules
│   ├── make_move()         # Execute moves
│   ├── legal_moves()       # Generate legal moves
│   └── status()            # Check game result
│
├── mcts.py                 # Pure MCTS AI
│   ├── MCTSNode            # Tree node (stores stats)
│   ├── MCTSPlayer          # AI that uses MCTS
│   ├── Selection (UCB1)    # Choose promising nodes
│   ├── Expansion           # Add new nodes
│   ├── Simulation          # Random rollout
│   └── Backpropagation     # Update statistics
│
├── puct.py                 # Neural network-guided MCTS
│   ├── PUCTNode            # Tree node with network priors
│   ├── PUCTPlayer          # AI that uses PUCT
│   └── PUCT formula        # Network-guided selection
│
├── network.py              # Neural network
│   ├── GameNetwork         # PyTorch model
│   ├── Policy Head         # Predicts move probabilities
│   ├── Value Head          # Predicts position value
│   └── NetworkTrainer      # Training utilities
│
├── training.py             # Self-play and training
│   ├── SelfPlayTrainer     # Manages training loop
│   ├── generate_mcts_games() # Bootstrap with MCTS
│   └── generate_puct_games() # Self-play with network
│
├── gui_game.py             # Graphical interface
│   ├── SOSGameGUI          # Tkinter GUI
│   ├── Human vs AI modes   # Different game modes
│   └── Visual board        # 8x8 grid with buttons
│
├── run_training.py         # Training script
│   └── Full training pipeline
│
└── play_vs_network.py      # Test trained network
    └── Command-line play against AI
```

### Data Flow

```
┌─────────────────┐
│   Self-Play     │  Games generated by current AI
│   (PUCT/MCTS)   │
└────────┬────────┘
         │ Training data: (state, policy, value)
         ▼
┌─────────────────┐
│ Neural Network  │  Learns from game data
│   Training      │
└────────┬────────┘
         │ Improved network
         ▼
┌─────────────────┐
│  Better PUCT    │  Uses better network for search
│    Player       │
└─────────────────┘
```

## 📦 Installation

### Requirements
- Python 3.7+
- PyTorch
- NumPy
- Tkinter (for GUI)

### Setup

```bash
# Clone the repository
git clone https://github.com/ellaro/SOS_Game.git
cd SOS_Game

# Install dependencies
pip install torch numpy

# Tkinter usually comes with Python, but if needed:
# Ubuntu/Debian: sudo apt-get install python3-tk
# macOS: brew install python-tk
```

## 🎮 Usage

### Play with GUI

```bash
python gui_game.py
```

Options:
- **Human vs AI (Trained)**: Play against the neural network
- **Human vs AI (MCTS)**: Play against pure MCTS
- **Human vs Human**: Two-player mode

### Play in Terminal

```bash
# Play against trained network
python play_vs_network.py

# Test MCTS player
python mcts.py

# Test PUCT player
python puct.py
```

### Run Tests

```bash
# Test core game logic
python main.py

# Test neural network
python network.py
```

## 🏋️ Training Your Own AI

### Quick Start

```bash
# Generate initial training data with MCTS, then train network
python run_training.py
```

This will:
1. Generate 100 self-play games using MCTS (slower but no network needed)
2. Train neural network on this data
3. Generate 50 self-play games using trained network (faster)
4. Continue training
5. Save the trained network as `network_mcts_YYYYMMDD_HHMMSS.pth`

### Custom Training

```python
from training import SelfPlayTrainer
from network import GameNetwork

# Create or load network
network = GameNetwork()
trainer = SelfPlayTrainer(network)

# Bootstrap with MCTS (no network needed)
mcts_data = trainer.generate_mcts_games(num_games=100, num_simulations=300)
trainer.train_network(mcts_data, epochs=10)

# Self-play with trained network
puct_data = trainer.generate_puct_games(num_games=50, num_simulations=200)
trainer.train_network(puct_data, epochs=10)

# Save trained network
network.save("my_trained_network.pth")
```

### Training Parameters

```python
# MCTS Parameters
num_simulations = 300      # More = stronger but slower
c_param = 1.4             # Exploration constant

# PUCT Parameters
num_simulations = 200      # Fewer needed than MCTS
c_puct = 1.0              # Exploration constant
temperature = 1.0          # Move randomness (0 = greedy)

# Network Training
learning_rate = 0.001      # Adam optimizer learning rate
batch_size = 32           # Training batch size
epochs = 10               # Passes through data
```

## 🎯 Performance Tips

### For Faster Training
- Reduce `num_simulations` (100-200 is often enough)
- Use smaller batches of training data
- Reduce network size (hidden_size=128 instead of 256)

### For Stronger AI
- Increase `num_simulations` (500-1000)
- Generate more self-play games
- Train for more epochs
- Use larger network (hidden_size=512)

### For Faster GUI Play
- AI simulation counts are already reduced in GUI:
  - Trained AI: 100 simulations (vs 200 in training)
  - MCTS AI: 50 simulations (vs 300 in training)
- Multi-threading prevents UI freezing during AI thinking

## 📊 What Makes This Implementation Special?

1. **Smart MCTS Expansion**: Prioritizes moves that create immediate SOS patterns
2. **Efficient Encoding**: Compact 131-value state representation
3. **Dual-Head Network**: Learns both policy and value simultaneously
4. **Self-Play Training**: Continuously improves through playing against itself
5. **User-Friendly GUI**: Beautiful interface with visual feedback
6. **Educational**: Well-commented code explaining each algorithm

## 📚 Further Reading

- [Monte Carlo Tree Search - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [AlphaGo Paper](https://www.nature.com/articles/nature16961) - Original MCTS+Neural Network approach
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Self-play training methodology
- [UCT Algorithm](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation) - UCB1 formula explained

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new AI algorithms
- Improve the neural network architecture
- Enhance the GUI
- Add more documentation

## 📄 License

Open source - feel free to use and modify for educational purposes.

## 👨‍💻 Author

Created as a demonstration of AI game-playing algorithms including MCTS and neural network-guided search.

---

**Enjoy playing SOS with AI! 🎮🤖**
