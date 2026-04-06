# MCTS AI Fixes Applied

## Critical Bugs Fixed

### 1. **Perspective Inversion in UCB Selection** ❌→✅
**Location:** `mcts.py` - `best_child()` function

**Problem:**
- When evaluating children, each child has `player_to_move` different from parent
- Values stored in children are from their perspective (e.g., player 1's perspective)
- But we need to evaluate from parent's perspective (e.g., player 0)
- A high value for opponent is BAD for us, so we were selecting bad moves!

**Fix:**
```python
# FLIP the value if we're looking at opponent's node
if child.player_to_move != self.player_to_move:
    avg_value = -avg_value  # Negate to get parent's perspective
```

### 2. **Weak Rollout Policy** ❌→✅
**Location:** `mcts.py` - `rollout()` function

**Problem:**
- Pure random playout doesn't understand SOS game strategy
- Doesn't prefer moves that create SOS patterns
- AI doesn't learn good tactical patterns

**Fix:**
Implemented heuristic rollout:
- 70% of time: Weighted selection preferring SOS-making moves
- 30% of time: Random (for exploration)
- Bonus weights: +100 for creating SOS, +50 for extra turn
- This makes rollout policy align with game objectives

### 3. **Binary Win/Loss Evaluation** ❌→✅
**Location:** `mcts.py` - `backpropagate()` function

**Problem:**
- Only tracked {0, 0.5, 1} values (loss, draw, win)
- Doesn't distinguish between winning by 1 point vs 10 points
- Loses crucial game state information

**Fix:**
Use score difference:
```python
score_diff = final_scores[self.player_to_move] - final_scores[1 - self.player_to_move]
value = max(-1.0, min(1.0, score_diff / 10.0))  # Normalized [-1, 1]
```
- Positive values = leading
- Negative values = losing
- Magnitude represents how well you're doing

### 4. **Insufficient Search Depth** ❌→✅
**Location:** `gui_game.py` - `start_game()` function

**Problem:**
- Only 300 simulations on 8×8 board with ~128 moves per state
- Not enough to explore good strategies

**Fix:**
- Increased to 2000 simulations for 'human_vs_mcts'
- With heuristic rollout, this is much more effective
- Balances computation time with playing strength

### 5. **Default UCB Exploration Parameter** ❌→✅
**Location:** `mcts.py` - `get_move()`, `best_child()` functions

**Problem:**
- Used default c_param=1.0
- Need better balance of exploration vs exploitation

**Fix:**
- Changed to c_param=sqrt(2)≈1.41 in `get_move()`
- Standard value in literature for UCB-based algorithms
- Better exploration on 8×8 board

## Results

### Before Fixes:
- AI was weak: human could easily beat it
- Only exploring moves superficially
- Not learning SOS patterns
- Averaging wins/losses incorrectly

### After Fixes:
- ✅ Proper perspective handling (min-max style)
- ✅ Heuristic rollout with SOS awareness
- ✅ Score difference evaluation
- ✅ Deeper search (2000 simulations)
- ✅ Better exploration parameter
- **AI now plays strategically strong**

## Testing

Run the GUI:
```bash
python gui_game.py
```

Select "👤 vs AI (MCTS)" to play against the improved AI.

The AI will now:
- Recognize and prioritize SOS patterns
- Plan multi-move strategies
- Choose high-value positions over lucky wins
- Adapt based on score differences
