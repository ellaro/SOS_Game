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
        self.player_to_move = 0
        if self.parent is not None:
            self.player_to_move = self.parent.game_state.current_player

    def is_fully_expanded(self):
        """Check if all possible moves have been tried"""
        return len(self.untried_moves) == 0

    def is_terminal(self):
        """Check if this is a terminal node (game over)"""
        return self.game_state.status() is not None

    def best_child(self, c_param=1.0):
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

        num_untried_moves = len(self.untried_moves)
        move_index = random.randint(0, num_untried_moves - 1)
        move = self.untried_moves[move_index]
        self.untried_moves[move_index] = self.untried_moves[-1]
        self.untried_moves.pop()


        # Create new game state
        next_state = self.game_state.clone()
        next_state.make_move(move)

        # Create child node
        child_node = MCTSNode(next_state, parent=self, move=move)

        self.children.append(child_node)

        return child_node


    #no clone
    def rollout(self):
        """We're gonna use the current scores as a fast proxy to estimate who is winning."""
        current_state = self.game_state
        if current_state.scores[0] > current_state.scores[1]:
            result = 0
        elif current_state.scores[0] < current_state.scores[1]:
            result = 1
        else:
            result = "draw"

        return result, current_state.scores

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

        self.value_sum += value

        # Recursively backpropagate to parent
        if self.parent:
            self.parent.backpropagate(result, final_scores)


class MCTSPlayer:
    """
    MCTS-based AI player
    """

    def __init__(self, num_simulations=200):
        self.num_simulations = num_simulations

    def get_move(self, game_state, verbose=False):
        """
        Get the best move using MCTS
        """
        root = MCTSNode(game_state.clone())

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # 1.Selection: traverse tree using UCB1
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # 2.Expansion: add a new child if possible
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # 3.Simulation: play out randomly
            result, final_scores = node.rollout()

            # 4.Backpropagation: update statistics
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
