# 🏗️ Code Architecture Overview

This document explains the architecture and design decisions in the SOS Game project.

## 📐 System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  ┌──────────────┐                    ┌──────────────┐       │
│  │  gui_game.py │                    │   play.py    │       │
│  │   (Tkinter)  │                    │  (Terminal)  │       │
│  └──────┬───────┘                    └──────┬───────┘       │
└─────────┼──────────────────────────────────┼───────────────┘
          │                                   │
          │         Game Engine Layer         │
          │  ┌────────────────────────────┐   │
          └─▶│       main.py              │◀──┘
             │     (SOSGame class)        │
             │  - Board state             │
             │  - Move validation         │
             │  - SOS detection           │
             │  - Game rules              │
             └─────────┬──────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
┌─────────▼─────┐ ┌───▼──────┐ ┌──▼────────────┐
│   mcts.py     │ │ puct.py  │ │  network.py   │
│ (Pure MCTS)   │ │(NN-MCTS) │ │ (Neural Net)  │
│               │ │          │ │               │
│ - MCTSNode    │ │- PUCTNode│ │- GameNetwork  │
│ - MCTSPlayer  │ │- PUCTPlayer│- Policy Head │
│ - UCB1        │ │- PUCT    │ │- Value Head   │
│ - Rollout     │ │- NN eval │ │- Training     │
└───────────────┘ └──────────┘ └───────┬───────┘
                                       │
                                       │
                          ┌────────────▼────────────┐
                          │     training.py         │
                          │  (Self-Play Training)   │
                          │                         │
                          │  - SelfPlayTrainer      │
                          │  - Data generation      │
                          │  - Training loop        │
                          └─────────────────────────┘
```

## 📁 File-by-File Breakdown

### 1. `main.py` - Core Game Engine

**Purpose:** Implements the SOS game rules and state management.

**Key Classes:**
- `SOSGame`: Main game state class

**Responsibilities:**
```python
class SOSGame:
    # State Management
    - board: 8x8 grid (None, 'S', or 'O')
    - current_player: 0 or 1
    - scores: [player0_score, player1_score]
    - game_over: boolean flag
    
    # Core Methods
    - make_move(move)      # Execute a move, update state
    - unmake_move(move)    # Undo a move (for search)
    - legal_moves()        # Generate all valid moves
    - status()             # Check if game is over, who won
    - clone()              # Deep copy for search algorithms
    
    # Internal Methods
    - _check_sos(r, c)     # Detect SOS patterns (8 directions)
    - _is_board_full()     # Check termination condition
    
    # Utility Methods
    - encode()             # Convert to neural network input
    - decode()             # Convert action index to move
    - print_board()        # Display for debugging
```

**Design Decisions:**

1. **Immutable Move Representation**
   ```python
   move = (row, col, letter)  # Tuple (immutable, hashable)
   ```
   Why: Can use as dictionary keys, no accidental modification

2. **Deep Copy Support**
   ```python
   def clone(self):
       return copy.deepcopy(self)
   ```
   Why: Search algorithms need independent game states

3. **Separate Encode/Decode**
   - `encode()`: Game state → Neural network input (131 values)
   - `decode()`: Action index → Move tuple
   Why: Clean separation of concerns

4. **SOS Detection Algorithm**
   ```python
   def _check_sos(self, r, c):
       # Check all 8 directions
       # If 'S': check if start or end of S-O-S
       # If 'O': check if middle of S-O-S
   ```
   Why: Efficient - only check patterns involving the new move

### 2. `mcts.py` - Pure Monte Carlo Tree Search

**Purpose:** AI player using traditional MCTS algorithm.

**Key Classes:**
```python
class MCTSNode:
    """Represents a node in the search tree"""
    - game_state          # Position at this node
    - parent              # Parent node
    - move                # Move that led here
    - children            # Child nodes
    - value_sum           # Sum of simulation results
    - visits              # Number of times visited
    - untried_moves       # Moves not yet expanded
    - player_to_move      # Whose turn it is
    
class MCTSPlayer:
    """AI that uses MCTS to choose moves"""
    - num_simulations     # How many iterations to run
    - get_move()          # Main entry point
```

**Algorithm Flow:**
```python
def get_move(self, game_state):
    root = MCTSNode(game_state)
    
    for _ in range(num_simulations):
        node = root
        
        # 1. Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child()  # UCB1 formula
        
        # 2. Expansion
        if not node.is_terminal():
            node = node.expand()  # Add one child
        
        # 3. Simulation
        result, scores = node.rollout()  # Random playout
        
        # 4. Backpropagation
        node.backpropagate(result, scores)
    
    # Return most visited child
    return max(root.children, key=lambda c: c.visits).move
```

**Design Decisions:**

1. **UCB1 for Selection**
   ```python
   ucb_value = avg_value + c * sqrt(2 * ln(parent_visits) / child_visits)
   ```
   Why: Proven optimal exploration-exploitation balance

2. **SOS Move Prioritization in Expansion**
   ```python
   if sos_moves:
       move = sos_moves.pop()  # Try SOS-creating moves first
       child.value_sum = 0.8   # Initial optimism
   ```
   Why: Domain knowledge speeds up learning

3. **Random Rollout**
   ```python
   move = random.choice(possible_moves)
   ```
   Why: Fast, unbiased, works well with enough simulations

4. **Per-Node Perspective**
   ```python
   self.player_to_move = game_state.current_player
   ```
   Why: Correctly evaluate wins/losses during backpropagation

### 3. `puct.py` - Neural Network-Guided MCTS

**Purpose:** Enhanced MCTS using neural network predictions.

**Key Classes:**
```python
class PUCTNode:
    """Node with neural network guidance"""
    - (same as MCTSNode, plus:)
    - prior_prob          # P(s,a) from neural network
    - Q                   # Average value (W/N)
    - is_expanded         # Track expansion state
    
class PUCTPlayer:
    """AI that uses PUCT search"""
    - network             # Neural network for guidance
    - num_simulations     # Search iterations
    - c_puct              # Exploration constant
    - temperature         # Move selection randomness
```

**Algorithm Flow:**
```python
def get_move(self, game_state):
    root = PUCTNode(game_state)
    
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        
        # 1. Selection (PUCT formula)
        while node.is_expanded and not node.is_terminal():
            node = node.select_child(c_puct)  # PUCT score
            search_path.append(node)
        
        # 2. Expansion (all children at once, with priors)
        if not node.is_terminal():
            node.expand(network)  # Get policy from NN
        
        # 3. Evaluation (neural network, not rollout)
        if node.is_terminal():
            value = actual_result()
        else:
            _, value = network.predict(node.game_state)
        
        # 4. Backpropagation (with value flipping)
        for n in reversed(search_path):
            n.update(value)
            value = -value  # Flip for opponent
    
    # Select based on temperature
    return choose_move(root.children, temperature)
```

**Design Decisions:**

1. **PUCT Formula Instead of UCB1**
   ```python
   puct_score = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
   ```
   Why: Uses neural network priors to guide exploration

2. **Expand All Children at Once**
   ```python
   for move in legal_moves:
       prior = policy_probs[action_idx]
       child = PUCTNode(..., prior_prob=prior)
   ```
   Why: Need all priors to properly apply PUCT formula

3. **Neural Network Evaluation**
   ```python
   _, value = network.predict(game_state)  # Fast!
   ```
   Why: Faster than rollout, learns from experience

4. **Temperature-Based Move Selection**
   ```python
   if temperature == 0:
       pick most visited  # Greedy (for play)
   else:
       sample from distribution  # Stochastic (for training)
   ```
   Why: Exploration during training, exploitation during play

### 4. `network.py` - Neural Network

**Purpose:** Deep learning model for position evaluation and move prediction.

**Architecture:**
```python
class GameNetwork(nn.Module):
    Input: 131 values (board + player + scores)
    
    Shared Layers:
    fc1: 131 → 256  (ReLU)
    fc2: 256 → 256  (ReLU)
    fc3: 256 → 256  (ReLU)
    
    Policy Head (move prediction):
    policy_fc1: 256 → 128  (ReLU)
    policy_fc2: 128 → 128  (log_softmax)
    Output: 128 action probabilities
    
    Value Head (position evaluation):
    value_fc1: 256 → 64  (ReLU)
    value_fc2: 64 → 1   (tanh)
    Output: value in [-1, 1]
```

**Design Decisions:**

1. **Dual-Head Architecture**
   ```
   Shared layers → Policy head (what to play)
                 → Value head (who's winning)
   ```
   Why: Both tasks share features, more efficient learning

2. **Log Softmax for Policy**
   ```python
   policy = F.log_softmax(policy, dim=1)
   ```
   Why: Numerical stability, works with cross-entropy loss

3. **Tanh for Value**
   ```python
   value = torch.tanh(value)  # Range: [-1, 1]
   ```
   Why: -1=loss, 0=draw, +1=win (natural interpretation)

4. **State Encoding**
   ```python
   # Board: 2 bits per cell (S or O)
   # Player: 1 bit
   # Scores: 2 values
   # Total: 128 + 1 + 2 = 131 values
   ```
   Why: Compact, contains all relevant information

### 5. `training.py` - Self-Play Training

**Purpose:** Generate training data and improve the network.

**Key Class:**
```python
class SelfPlayTrainer:
    """Manages the training loop"""
    - network               # Neural network to train
    - trainer               # NetworkTrainer helper
    - training_data         # Accumulated (state, policy, value) tuples
    
    # Data Generation
    - generate_mcts_games() # Bootstrap with pure MCTS
    - generate_puct_games() # Self-play with network
    
    # Training
    - train_network()       # Train on collected data
```

**Training Pipeline:**
```python
# 1. Generate Games
for game in range(num_games):
    game_history = []
    
    while not game_over:
        # PUCT search with current network
        move, visit_probs = puct.get_move(game, return_probs=True)
        game_history.append((state, visit_probs, current_player))
        game.make_move(move)
    
    # 2. Assign Values (game outcome)
    winner = game.status()
    for (state, policy, player) in game_history:
        if winner == player:
            value = 1.0  # Win
        elif winner == "draw":
            value = 0.0  # Draw
        else:
            value = -1.0 # Loss
        
        training_data.append((state, policy, value))

# 3. Train Network
for epoch in range(epochs):
    shuffle(training_data)
    for batch in batches:
        loss = train_step(batch)
```

**Design Decisions:**

1. **MCTS Bootstrap**
   ```python
   generate_mcts_games()  # Initial data without network
   ```
   Why: Cold-start problem - need data before network is useful

2. **Visit Count as Policy Target**
   ```python
   target_policy = visit_counts / sum(visit_counts)
   ```
   Why: PUCT's visit distribution is better than raw policy

3. **Outcome as Value Target**
   ```python
   target_value = 1 if won else -1 if lost else 0
   ```
   Why: Ground truth - actual game result

4. **Player Perspective in Values**
   ```python
   value = 1.0 if winner == player else -1.0
   ```
   Why: Each position evaluated from mover's perspective

### 6. `gui_game.py` - Graphical Interface

**Purpose:** User-friendly GUI for playing the game.

**Key Features:**
```python
class SOSGameGUI:
    # Game Modes
    - Human vs Trained AI
    - Human vs MCTS AI
    - Human vs Human
    
    # UI Components
    - 8x8 button grid (board)
    - Score display
    - Letter selection (S/O)
    - Status label
    - Mode selection buttons
    
    # Threading
    - AI runs in separate thread
    - UI doesn't freeze during AI thinking
```

**Design Decisions:**

1. **Threaded AI**
   ```python
   threading.Thread(target=self.ai_move, daemon=True).start()
   ```
   Why: Prevents UI freezing during long AI searches

2. **Reduced Simulations for GUI**
   ```python
   # Training: 200-300 simulations
   # GUI play: 50-100 simulations
   ```
   Why: Faster response time, better user experience

3. **Disabled Board During AI Turn**
   ```python
   if ai_thinking:
       disable_board()
   ```
   Why: Prevents invalid human moves during AI's turn

4. **Visual Feedback**
   ```python
   status_label.config(text="🤖 AI is thinking... ⏳")
   ```
   Why: User knows AI is working, not frozen

## 🔄 Data Flow

### 1. Training Flow
```
MCTS Self-Play
    ↓
Generate Games → (state, visit_probs, outcome)
    ↓
Network Training
    ↓
Improved Network
    ↓
PUCT Self-Play (uses improved network)
    ↓
More Games → Better Data
    ↓
[Repeat]
```

### 2. Play Flow
```
User makes move
    ↓
Update game state
    ↓
Check if AI's turn
    ↓
PUCT search (uses trained network)
    ↓
Network predicts: policy + value
    ↓
PUCT builds tree using predictions
    ↓
Select best move
    ↓
Update game state
    ↓
Check game over
```

## 🎯 Design Principles

### 1. **Separation of Concerns**
- Game rules (main.py) ← Independent
- Search algorithms (mcts.py, puct.py) ← Use game interface
- Neural network (network.py) ← Separate from search
- UI (gui_game.py) ← Uses everything

### 2. **Immutability Where Possible**
- Moves are tuples (immutable)
- States are cloned for search (no side effects)

### 3. **Clean Interfaces**
```python
# All AI players implement same interface
class Player:
    def get_move(self, game_state) -> move
```

### 4. **Educational Code**
- Extensive comments
- Clear variable names
- Separate files for each concept
- Test functions in each file

## 📊 Performance Considerations

### Time Complexity

| Operation | MCTS | PUCT |
|-----------|------|------|
| Single simulation | O(depth + remaining_moves) | O(depth) |
| Total search | O(sims × game_length) | O(sims × depth) |
| Network prediction | N/A | O(1) - constant NN forward pass |

### Space Complexity

| Component | Memory |
|-----------|--------|
| Game state | O(1) - fixed 8×8 board |
| MCTS tree | O(simulations) - ~100 bytes/node |
| PUCT tree | O(simulations × branching) - expands more |
| Neural network | O(1) - fixed model size (~500KB) |

### Optimizations Applied

1. **Lazy Expansion** (MCTS): One child at a time
2. **State Cloning**: Only when needed (not during selection)
3. **Rollout Limit**: Max 64 moves (prevents infinite loops)
4. **Move Prioritization**: Try good moves first
5. **Threaded GUI**: Don't block UI during AI thinking

## 🧪 Testing Strategy

Each file includes `if __name__ == '__main__':` tests:

- `main.py`: Test SOS detection, full game
- `mcts.py`: Test MCTS player
- `puct.py`: Test PUCT player
- `network.py`: Test network save/load, prediction
- `training.py`: Test self-play generation

## 🔧 Extensibility

Easy to add:
- **New AI algorithms**: Implement `get_move(game_state)` interface
- **Different board sizes**: Change constants in `SOSGame`
- **New network architectures**: Subclass `GameNetwork`
- **Different games**: Implement similar `Game` interface

## 📚 Key Takeaways

1. **Modular design** enables easy testing and modification
2. **Clean interfaces** between components
3. **MCTS and PUCT** share similar structure but different evaluation
4. **Neural network** learns from self-play experience
5. **GUI uses threading** for responsive UI
6. **State cloning** ensures search doesn't affect game state
7. **Domain knowledge** (SOS priorities) speeds up learning

---

**This architecture makes the codebase maintainable, extensible, and educational! 🏗️**
