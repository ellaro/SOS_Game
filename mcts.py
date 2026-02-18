import random
import math

# Constants for MCTS tuning
MAX_ROLLOUT_MOVES_TO_CHECK = 10  # Number of moves to sample for SOS detection in rollouts
SCORE_NORMALIZATION_FACTOR = 30.0  # Normalizes score differences to [0,1] range
SOS_MOVE_VISIT_THRESHOLD = 0.7  # Minimum visit ratio to prefer immediate SOS moves


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

        # Prefer SOS moves - track if this was an SOS move
        is_sos_move = False
        if sos_moves:
            move = sos_moves.pop()
            is_sos_move = True
            # Remove from untried_moves
            self.untried_moves = [m for m in self.untried_moves if m != move]
        else:
            move = self.untried_moves.pop()

        # Create new game state
        next_state = self.game_state.clone()
        next_state.make_move(move)

        # Create child node
        child_node = MCTSNode(next_state, parent=self, move=move)

        # Give initial bonus to SOS-creating moves (fixed bug)
        if is_sos_move:
            # Strong initial bias for SOS moves to ensure they get explored
            child_node.value_sum = 5.0  # Strong initial optimism
            child_node.visits = 5  # Act as if we've already explored it successfully

        self.children.append(child_node)

        return child_node

    def rollout(self):
        """Simulate a semi-smart game - prefer SOS-creating moves with fast heuristic"""
        current_state = self.game_state.clone()

        max_moves = 64
        move_count = 0

        while current_state.status() is None and move_count < max_moves:
            possible_moves = current_state.legal_moves()
            if not possible_moves:
                break

            # Quick heuristic: Check for SOS-creating moves (sampling for speed)
            # Only check a subset of moves to keep rollout fast
            sample_size = min(MAX_ROLLOUT_MOVES_TO_CHECK, len(possible_moves))
            moves_to_check = random.sample(possible_moves, sample_size) if len(possible_moves) > sample_size else possible_moves
            
            sos_creating_move = None
            for move in moves_to_check:
                r, c, letter = move
                # Quick check: Does this move complete an SOS?
                # This is a simplified heuristic check
                if self._quick_sos_check(current_state, r, c, letter):
                    sos_creating_move = move
                    break
            
            # Use SOS move if found, otherwise random
            if sos_creating_move:
                move = sos_creating_move
            else:
                move = random.choice(possible_moves)
            
            current_state.make_move(move)
            move_count += 1

        return current_state.status(), current_state.scores
    
    def _quick_sos_check(self, state, r, c, letter):
        """Quick heuristic to check if a move likely creates SOS"""
        board = state.board
        
        if letter == 'S':
            # Check if placing S creates SOS pattern (S-O-S or O-S)
            # Look for existing O adjacent
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            for dr, dc in directions:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < 8 and 0 <= c2 < 8 and board[r2][c2] == 'O':
                    # Check if there's an S on the other side
                    r3, c3 = r2 + dr, c2 + dc
                    if 0 <= r3 < 8 and 0 <= c3 < 8 and board[r3][c3] == 'S':
                        return True
        elif letter == 'O':
            # Check if placing O creates SOS (between two S's)
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                r_before, c_before = r - dr, c - dc
                r_after, c_after = r + dr, c + dc
                if (0 <= r_before < 8 and 0 <= c_before < 8 and
                    0 <= r_after < 8 and 0 <= c_after < 8 and
                    board[r_before][c_before] == 'S' and board[r_after][c_after] == 'S'):
                    return True
        
        return False

    def backpropagate(self, result, final_scores):
        """
        Update this node and all ancestors with the result

        result: 0, 1, or "draw"
        final_scores: [score_player_0, score_player_1]
        """
        self.visits += 1

        # Use score difference for better evaluation (critical for SOS game)
        # This captures not just wins/losses but also the margin of victory
        score_diff = final_scores[self.player_to_move] - final_scores[1 - self.player_to_move]
        
        # Normalize to [0, 1] range
        # SOS games typically have scores 0-20, normalization factor allows for larger differences
        value = 0.5 + (score_diff / SCORE_NORMALIZATION_FACTOR)
        value = max(0, min(1, value))  # Clamp to [0, 1]

        self.value_sum += value

        # Recursively backpropagate to parent
        if self.parent:
            self.parent.backpropagate(result, final_scores)


class MCTSPlayer:
    """
    MCTS-based AI player
    """

    def __init__(self, num_simulations=3500):
        self.num_simulations = num_simulations

    def get_move(self, game_state, verbose=False):
        """
        Get the best move using MCTS
        """
        # Quick pre-check: If there's an immediate SOS-creating move, strongly prefer it
        legal_moves = game_state.legal_moves()
        immediate_sos_moves = []
        
        for move in legal_moves:
            test_state = game_state.clone()
            old_score = test_state.scores[test_state.current_player]
            test_state.make_move(move)
            new_score = test_state.scores[game_state.current_player]
            
            if new_score > old_score:
                immediate_sos_moves.append((move, new_score - old_score))
        
        # If we found immediate SOS moves, heavily bias toward them
        if immediate_sos_moves:
            if verbose:
                print(f"\n💡 Found {len(immediate_sos_moves)} immediate SOS-creating moves!")
                for move, score_gain in immediate_sos_moves:
                    print(f"   {move} would gain {score_gain} points")
        
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

        # Choose the best move
        if not root.children:
            # No children expanded - return random legal move
            legal = game_state.legal_moves()
            return random.choice(legal) if legal else None

        # Special handling for immediate SOS moves
        # If we have immediate SOS moves, prefer them if they're in top candidates
        if immediate_sos_moves:
            sos_move_set = set(move for move, _ in immediate_sos_moves)
            
            # Find SOS moves among explored children
            sos_children = [c for c in root.children if c.move in sos_move_set]
            
            if sos_children:
                # Get the most visited non-SOS child for comparison
                non_sos_children = [c for c in root.children if c.move not in sos_move_set]
                max_non_sos_visits = max((c.visits for c in non_sos_children), default=0)
                
                # If any SOS child has at least the threshold of max visits, choose the best SOS move
                best_sos_child = max(sos_children, key=lambda c: (c.visits, c.value_sum / max(c.visits, 1)))
                
                if best_sos_child.visits >= SOS_MOVE_VISIT_THRESHOLD * max_non_sos_visits:
                    if verbose:
                        print(f"\n🎯 Selecting immediate SOS move {best_sos_child.move}")
                    return best_sos_child.move
        
        # Default: choose the most visited child (most robust choice)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move