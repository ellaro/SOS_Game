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
        Select best child using UCB1 formula
        UCB1 = value/visits + c * sqrt(ln(parent_visits) / visits)
        """
        choices_weights = []

        for child in self.children:
            if child.visits == 0:
                # Unvisited children get infinite value
                ucb_value = float('inf')
            else:
                # Average value from this child's perspective
                avg_value = child.value_sum / child.visits

                # UCB1 formula
                exploration = c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
                ucb_value = avg_value + exploration

            choices_weights.append(ucb_value)

        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        """
        Expand tree by creating a new child node
        PRIORITIZE moves that create immediate SOS
        """
        # Separate moves into SOS-creating and regular moves
        sos_moves = []
        regular_moves = []

        for move in self.untried_moves:
            test_state = self.game_state.clone()
            old_score = test_state.scores[test_state.current_player]
            test_state.make_move(move)
            new_score = test_state.scores[self.game_state.current_player]

            if new_score > old_score:
                sos_moves.append(move)
            else:
                regular_moves.append(move)

        # Prefer SOS moves
        if sos_moves:
            move = sos_moves.pop()
            # Remove from untried_moves
            self.untried_moves = [m for m in self.untried_moves if m != move]
        else:
            move = self.untried_moves.pop()

        # Create new game state
        next_state = self.game_state.clone()
        next_state.make_move(move)

        # Create child node with bonus for immediate SOS
        child_node = MCTSNode(next_state, parent=self, move=move)

        # Give initial bonus to SOS-creating moves
        if move in sos_moves or (next_state.scores[self.game_state.current_player] >
                                 self.game_state.scores[self.game_state.current_player]):
            child_node.value_sum = 0.8  # Initial optimism
            child_node.visits = 1

        self.children.append(child_node)

        return child_node

    def rollout(self):
        """Simulate a completely RANDOM game (fastest)"""
        current_state = self.game_state.clone()

        max_moves = 64
        move_count = 0

        while current_state.status() is None and move_count < max_moves:
            possible_moves = current_state.legal_moves()
            if not possible_moves:
                break

            move = random.choice(possible_moves)
            current_state.make_move(move)
            move_count += 1

        return current_state.status(), current_state.scores

    def backpropagate(self, result, final_scores):
        """
        Update this node and all ancestors with the result

        result: 0, 1, or "draw"
        final_scores: [score_player_0, score_player_1]
        """
        self.visits += 1

        # Calculate value from the perspective of player_to_move at this node
        if result == "draw":
            value = 0.5
        elif result == self.player_to_move:
            value = 1.0  # Win for the player to move at this node
        else:
            value = 0.0  # Loss for the player to move at this node

        # Alternative: use score difference (better for SOS)
        # score_diff = final_scores[self.player_to_move] - final_scores[1 - self.player_to_move]
        # value = 0.5 + (score_diff / 20.0)  # Normalize to roughly [0, 1]
        # value = max(0, min(1, value))  # Clamp to [0, 1]

        self.value_sum += value

        # Recursively backpropagate to parent
        if self.parent:
            self.parent.backpropagate(result, final_scores)


class MCTSPlayer:
    """
    MCTS-based AI player
    """

    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations

    def get_move(self, game_state, verbose=False):
        """
        Get the best move using MCTS
        """
        root = MCTSNode(game_state.clone())

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # Selection: traverse tree using UCB1
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion: add a new child if possible
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation: play out randomly
            result, final_scores = node.rollout()

            # Backpropagation: update statistics
            node.backpropagate(result, final_scores)

        # Debug output
        if verbose and root.children:
            print("\n--- MCTS Analysis ---")
            # Sort children by visits
            sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
            for i, child in enumerate(sorted_children[:5]):  # Show top 5
                win_rate = child.value_sum / child.visits if child.visits > 0 else 0
                print(f"{i + 1}. Move {child.move}: visits={child.visits}, "
                      f"value_sum={child.value_sum:.1f}, avg_value={win_rate:.3f}")

        # Choose the most visited child (most robust choice)
        if not root.children:
            # No children expanded - return random legal move
            legal = game_state.legal_moves()
            return random.choice(legal) if legal else None

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move