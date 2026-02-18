import random
import math
import time


def in_bounds(r, c):
    """Return True if (r,c) is inside the 8x8 board."""
    return 0 <= r < 8 and 0 <= c < 8


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
        # Use the game state's current player directly (was incorrect before)
        self.player_to_move = game_state.current_player

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
                # Unvisited children get infinite value (encourage exploration)
                ucb_value = float('inf')
            else:
                # Average value from this child's stored perspective
                child_avg = child.value_sum / child.visits

                # Convert child's average value to the parent's perspective.
                # child_avg is from perspective of child.player_to_move.
                # We want value from perspective of the player to move at this node.
                if child.player_to_move == self.player_to_move:
                    avg_value = child_avg
                else:
                    avg_value = 1.0 - child_avg

                # UCB1 formula (guard against log(0) and div-by-zero)
                exploration = c_param * math.sqrt((2 * math.log(self.visits + 1)) / (child.visits + 1))
                ucb_value = avg_value + exploration

            choices_weights.append(ucb_value)

        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):

        num_untried_moves = len(self.untried_moves)
        move_index = random.randint(0, num_untried_moves - 1)
        move = self.untried_moves[move_index]
        self.untried_moves[move_index] = self.untried_moves[-1]
        self.untried_moves.pop()


        # Create new game state and child node
        next_state = self.game_state.clone()
        next_state.make_move(move)

        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)

        return child_node


    #no clone
    def rollout(self):
        """Perform a random playout until terminal and return the result.

        This replaces the earlier flawed shortcut that used the current
        partial scores instead of simulating to the end.
        """
        state = self.game_state.clone()

        # Play until terminal. Prefer moves that immediately create SOS.
        while state.status() is None:
            moves = state.legal_moves()
            if not moves:
                break

            # Heuristic: if any move creates an SOS, prefer those
            sos_moves = []
            # We can't call unmake_move here without extra bookkeeping,
            # so use a lightweight heuristic: prefer moves where placing 'S' or 'O'
            # would form the pattern by checking neighbors directly.
            sos_moves = []
            for m in moves:
                r, c, letter = m
                # quick local check for possible SOS formation
                if letter == 'S':
                    directions = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
                    for dr, dc in directions:
                        r2, c2 = r + dr, c + dc
                        r3, c3 = r + 2*dr, c + 2*dc
                        if in_bounds(r2, c2) and in_bounds(r3, c3):
                            if state.board[r2][c2] == 'O' and state.board[r3][c3] == 'S':
                                sos_moves.append(m)
                                break
                else:  # 'O'
                    basic_directions = [(0,1),(1,0),(1,1),(1,-1)]
                    for dr, dc in basic_directions:
                        r_before, c_before = r - dr, c - dc
                        r_after, c_after = r + dr, c + dc
                        if in_bounds(r_before, c_before) and in_bounds(r_after, c_after):
                            if state.board[r_before][c_before] == 'S' and state.board[r_after][c_after] == 'S':
                                sos_moves.append(m)
                                break

            if sos_moves:
                move = random.choice(sos_moves)
            else:
                move = random.choice(moves)

            state.make_move(move)

        result = state.status()
        return result, state.scores

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

    def __init__(self, num_simulations=200, time_limit=None):
        """
        Args:
            num_simulations: maximum number of simulations to run
            time_limit: optional wall-clock time limit in seconds (overrides simulations if set)
        """
        self.num_simulations = num_simulations
        self.time_limit = time_limit

    def get_move(self, game_state, verbose=False):
        """
        Get the best move using MCTS
        """
        root = MCTSNode(game_state.clone())

        # QUICK WIN CHECK: if any legal root move immediately creates SOS,
        # play it instantly (fast tactical awareness).
        for mv in game_state.legal_moves():
            tmp = game_state.clone()
            move_info = tmp.make_move(mv)
            # make_move returns (sos_count, changed_player)
            if move_info[0] > 0:
                return mv

        # Run simulations (either fixed count or until time limit)
        start_time = time.time()

        if self.time_limit is None:
            sim_iter = range(self.num_simulations)
        else:
            sim_iter = iter(int, 1)  # infinite iterator; we'll break on time

        for sim_i, _ in enumerate(sim_iter):
            # Stop if time limit reached
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break

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

            # Early stopping: if one child dominates visits, stop early
            if sim_i % 10 == 0 and root.children:
                total_visits = sum(c.visits for c in root.children)
                if total_visits > 0:
                    top_visits = max(c.visits for c in root.children)
                    if top_visits / total_visits > 0.95:
                        break

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
