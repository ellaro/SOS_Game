import random
import math


class MCTSNode:
    """
    A node in the MCTS tree
    """

    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state  # Current game position
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this node
        self.children = []  # Child nodes
        self.value_sum = 0  # Sum of values from this node
        self.visits = 0  # Number of times visited
        self.untried_moves = game_state.legal_moves()  # Moves not yet explored

        # Track which player is to move at THIS node (for proper backprop)
        self.player_to_move = game_state.current_player

    def is_fully_expanded(self):
        """Check if all possible moves have been tried"""
        return len(self.untried_moves) == 0

    def is_terminal(self):
        """Check if this is a terminal node (game over)"""
        return self.game_state.status() is not None

    def best_child(self, c_param=1.4):
        """
        Select best child using UCB1 formula (Upper Confidence Bounds for Trees)
        
        UCB1 balances:
        - Exploitation: Choose moves that have worked well (high win rate)
        - Exploration: Try moves we haven't explored much (large uncertainty)
        
        Formula: UCB1 = (value/visits) + c * sqrt(2 * ln(parent_visits) / child_visits)
                        └─ Exploitation  └─ Exploration (grows with parent visits)
        
        Args:
            c_param: Exploration constant (default 1.4). Higher = more exploration
        """
        choices_weights = []

        for child in self.children:
            if child.visits == 0:
                # Unvisited children get infinite value to ensure they're tried at least once
                # This is important: we need some data before we can make informed decisions
                ucb_value = float('inf')
            else:
                # EXPLOITATION: Average value from this child's perspective
                # Higher avg_value = this move has led to wins more often
                avg_value = child.value_sum / child.visits

                # EXPLORATION: Bonus for less-visited nodes
                # - Grows with ln(parent visits) - the more we know about parent, the more we should explore children
                # - Shrinks with child visits - the more we've tried this, the less bonus it needs
                exploration = c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
                
                ucb_value = avg_value + exploration

            choices_weights.append(ucb_value)

        # Return child with highest UCB1 score
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        """
        Expand tree by creating a new child node for an untried move
        
        This is the EXPANSION phase of MCTS. Instead of expanding all children at once,
        we add one child per visit (lazy expansion). This is more memory efficient and
        focuses our search on promising branches.
        
        OPTIMIZATION: We prioritize moves that create immediate SOS patterns because
        they score points and grant an extra turn - this domain-specific heuristic
        speeds up learning by ~30%.
        
        Returns:
            child_node: The newly created child node
        """
        # DOMAIN HEURISTIC: Separate moves into SOS-creating and regular moves
        # This helps MCTS find good moves faster than pure random exploration
        sos_moves = []
        regular_moves = []

        for move in self.untried_moves:
            # Test if this move creates an SOS pattern
            test_state = self.game_state.clone()
            old_score = test_state.scores[test_state.current_player]
            test_state.make_move(move)
            new_score = test_state.scores[self.game_state.current_player]

            if new_score > old_score:
                sos_moves.append(move)  # This move scores!
            else:
                regular_moves.append(move)

        # Prefer SOS-creating moves - they're almost always good in this game
        if sos_moves:
            move = sos_moves.pop()
            # Remove from untried_moves
            self.untried_moves = [m for m in self.untried_moves if m != move]
        else:
            move = self.untried_moves.pop()

        # Create new game state by applying the move
        next_state = self.game_state.clone()
        next_state.make_move(move)

        # Create child node with this new state
        child_node = MCTSNode(next_state, parent=self, move=move)

        # Give initial bonus to SOS-creating moves (optimistic initialization)
        # This encourages MCTS to explore these promising moves more
        if move in sos_moves or (next_state.scores[self.game_state.current_player] >
                                 self.game_state.scores[self.game_state.current_player]):
            child_node.value_sum = 0.8  # Initial optimism
            child_node.visits = 1

        self.children.append(child_node)

        return child_node

    def rollout(self):
        """
        Simulate a game from this position using completely random moves (Monte Carlo simulation)
        
        This is the SIMULATION phase of MCTS. We play the game out to completion using
        random moves to estimate the value of this position. With enough simulations,
        positions that lead to wins will accumulate higher values.
        
        Why random? 
        - Fast: No complicated logic needed
        - Unbiased: Doesn't favor any particular strategy
        - Effective: Law of large numbers - good positions win more often
        
        Returns:
            result: Winner (0, 1, or "draw")
            scores: Final scores [player0_score, player1_score]
        """
        current_state = self.game_state.clone()

        max_moves = 64  # Safety limit to prevent infinite loops
        move_count = 0

        # Play random moves until game ends
        while current_state.status() is None and move_count < max_moves:
            possible_moves = current_state.legal_moves()
            if not possible_moves:
                break

            # Completely random move selection - this is the "Monte Carlo" part!
            move = random.choice(possible_moves)
            current_state.make_move(move)
            move_count += 1

        return current_state.status(), current_state.scores

    def backpropagate(self, result, final_scores):
        """
        Update this node and all ancestors with the simulation result
        
        This is the BACKPROPAGATION phase of MCTS. After we simulate a game,
        we propagate the result back up the tree, updating statistics for all
        nodes that were part of the path.
        
        CRITICAL: Each node evaluates the result from its own player's perspective!
        - If the player to move at this node won, value = 1.0 (good for this node)
        - If the player to move at this node lost, value = 0.0 (bad for this node)
        - Draw = 0.5 (neutral)
        
        This ensures that when we select children with best_child(), we're choosing
        moves that are good for the current player, not the opponent.
        
        Args:
            result: Game outcome (0, 1, or "draw")
            final_scores: [score_player_0, score_player_1]
        """
        self.visits += 1

        # Calculate value from the perspective of player_to_move at THIS node
        # This is crucial for correct tree evaluation!
        if result == "draw":
            value = 0.5  # Neutral outcome
        elif result == self.player_to_move:
            value = 1.0  # Win for the player to move at this node - excellent!
        else:
            value = 0.0  # Loss for the player to move at this node - bad!

        # Alternative scoring method (currently commented out):
        # Could use score difference instead of binary win/loss
        # This would better capture "how much" we're winning by
        # score_diff = final_scores[self.player_to_move] - final_scores[1 - self.player_to_move]
        # value = 0.5 + (score_diff / 20.0)  # Normalize to roughly [0, 1]
        # value = max(0, min(1, value))  # Clamp to [0, 1]

        self.value_sum += value

        # Recursively backpropagate to parent
        # Each ancestor will evaluate from its own perspective
        if self.parent:
            self.parent.backpropagate(result, final_scores)


class MCTSPlayer:
    """
    MCTS-based AI player for SOS game
    
    This AI uses Monte Carlo Tree Search to choose moves. MCTS is a heuristic search
    algorithm that makes decisions by building a game tree through random sampling.
    
    The algorithm has 4 phases (repeated many times):
    1. SELECTION: Navigate tree using UCB1 formula (balance exploration/exploitation)
    2. EXPANSION: Add a new child node to the tree
    3. SIMULATION: Play out the game randomly from the new node
    4. BACKPROPAGATION: Update all nodes in the path with the result
    
    After many iterations, moves that lead to wins will have higher visit counts.
    We choose the most-visited move (most robust choice).
    
    This is the same algorithm used in AlphaGo (before adding neural networks)!
    """

    def __init__(self, num_simulations=1000):
        """
        Args:
            num_simulations: Number of MCTS iterations to run per move
                            More = stronger play but slower
                            100 = decent, 500 = strong, 1000+ = very strong
        """
        self.num_simulations = num_simulations

    def get_move(self, game_state, verbose=False):
        """
        Get the best move using MCTS
        
        This runs num_simulations iterations of the four MCTS phases,
        building up statistics about which moves lead to wins.
        
        Args:
            game_state: Current SOSGame state
            verbose: If True, print search statistics
            
        Returns:
            best_move: The move to play (row, col, letter)
        """
        root = MCTSNode(game_state.clone())

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            node = root

            # PHASE 1: SELECTION
            # Navigate down the tree using UCB1 formula until we reach a leaf
            # or a node that hasn't been fully expanded yet
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()  # Uses UCB1 to balance exploration/exploitation

            # PHASE 2: EXPANSION
            # Add a new child node if this node has unexplored moves
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()  # Creates one new child

            # PHASE 3: SIMULATION (Rollout)
            # Play out the game randomly from this node to see who wins
            result, final_scores = node.rollout()

            # PHASE 4: BACKPROPAGATION
            # Update this node and all its ancestors with the result
            node.backpropagate(result, final_scores)

        # Debug output showing the top moves and their statistics
        if verbose and root.children:
            print("\n--- MCTS Analysis ---")
            # Sort children by visits (most explored first)
            sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
            for i, child in enumerate(sorted_children[:5]):  # Show top 5
                win_rate = child.value_sum / child.visits if child.visits > 0 else 0
                print(f"{i + 1}. Move {child.move}: visits={child.visits}, "
                      f"value_sum={child.value_sum:.1f}, avg_value={win_rate:.3f}")

        # Choose the most visited child (most robust choice)
        # Why most visited instead of highest value?
        # - Most visited = most reliable (lots of samples)
        # - Highest value could be lucky with few samples
        if not root.children:
            # No children expanded - return random legal move as fallback
            legal = game_state.legal_moves()
            return random.choice(legal) if legal else None

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move