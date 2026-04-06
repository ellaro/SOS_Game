import math
import numpy as np


# ============================================================================
# Canonical move <-> action index mapping (shared with training code)
# ============================================================================

def move_to_action_index(move):
    """
    Convert a move (row, col, letter) to an action index (0-127).
    
    Action space: 0-127 (64 cells × 2 letters)
    - Cells are ordered: row 0→7, col 0→7 (left-to-right, top-to-bottom)
    - Even indices = 'S', Odd indices = 'O'
    
    Example:
        (0, 0, 'S') -> 0
        (0, 0, 'O') -> 1
        (0, 1, 'S') -> 2
        (0, 1, 'O') -> 3
        (7, 7, 'S') -> 126
        (7, 7, 'O') -> 127
    """
    row, col, letter = move
    cell_idx = row * 8 + col
    action_idx = cell_idx * 2 + (0 if letter == 'S' else 1)
    return action_idx


def action_index_to_move(action_idx):
    """
    Convert an action index (0-127) to a move (row, col, letter).
    Inverse of move_to_action_index().
    """
    idx = int(action_idx)
    cell_idx = idx // 2
    row = cell_idx // 8
    col = cell_idx % 8
    letter = 'S' if idx % 2 == 0 else 'O'
    return (row, col, letter)


def mask_illegal_moves(policy_probs, legal_moves):
    """
    Mask out illegal moves and renormalize policy probabilities.
    
    Args:
        policy_probs: numpy array of shape (128,) from network
        legal_moves: list of legal moves as (row, col, letter) tuples
    
    Returns:
        masked_probs: numpy array of shape (128,) with illegal moves zeroed
                      and probabilities renormalized to sum to 1
    """
    masked = np.zeros(128, dtype=np.float32)
    
    # Set probabilities only for legal moves
    for move in legal_moves:
        action_idx = move_to_action_index(move)
        masked[action_idx] = policy_probs[action_idx]
    
    # Renormalize
    total = masked.sum()
    if total > 0:
        masked /= total
    else:
        # Fallback: uniform distribution over legal moves
        legal_action_indices = [move_to_action_index(m) for m in legal_moves]
        for idx in legal_action_indices:
            masked[idx] = 1.0 / len(legal_action_indices)
    
    return masked


class PUCTNode:
    """
    A node in the PUCT search tree
    Similar to MCTSNode but uses neural network guidance
    """

    def __init__(self, game_state, parent=None, move=None, prior_prob=1.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []

        # PUCT specific attributes
        self.prior_prob = prior_prob  # P(s,a) from parent's policy network
        self.visit_count = 0  # N(s,a)
        self.value_sum = 0  # W(s,a) - sum of values
        self.Q = 0  # Q(s,a) = W(s,a) / N(s,a) - average value

        self.player_to_move = game_state.current_player
        self.is_expanded = False

    def is_terminal(self):
        """Check if this is a terminal node (game over)"""
        return self.game_state.status() is not None

    def select_child(self, c_puct=1.0):
        """
        Select child with highest PUCT score

        PUCT formula: U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_child = None

        # Use (visit_count + 1) to avoid zero and stabilize the exploration term
        sqrt_parent_visits = math.sqrt(self.visit_count + 1)

        for child in self.children:
            # Q value (average value from this child's perspective)
            q_value = 0
            if child.player_to_move == self.player_to_move:
                q_value = child.Q
            else:
                q_value = -child.Q

            # Exploration bonus
            u_value = c_puct * child.prior_prob * sqrt_parent_visits / (1 + child.visit_count)

            # PUCT score
            puct_score = q_value + u_value

            if puct_score > best_score:
                best_score = puct_score
                best_child = child

        return best_child

    def expand(self, network):
        """
        Expand this node by creating children for all legal moves.
        Uses network to get prior probabilities, masks illegal moves, and renormalizes.
        """
        if self.is_expanded or self.is_terminal():
            return

        # Get policy and value from network (policy is shape 128)
        policy_probs, value = network.predict(self.game_state)

        # Get legal moves
        legal_moves = self.game_state.legal_moves()

        if not legal_moves:
            self.is_expanded = True
            return value

        # Mask illegal moves and renormalize probabilities
        children = []
        priors = []

        for move in legal_moves:
            action_idx = move_to_action_index(move)
            prior = policy_probs[action_idx]

            next_state = self.game_state.clone()
            next_state.make_move(move)

            child = PUCTNode(
                next_state,
                parent=self,
                move=move,
                prior_prob=prior
            )

            children.append(child)
            priors.append(prior)

        # 🔥 Normalize ONLY over legal moves
        priors = np.array(priors, dtype=np.float32)

        if priors.sum() > 0:
            priors /= priors.sum()
        else:
            priors = np.ones_like(priors) / len(priors)

        # Assign back
        for i, child in enumerate(children):
            child.prior_prob = priors[i]

        self.children = children

        # Create a child for each legal move
        # for move in legal_moves:
        #     # Get action index for this move using canonical mapping
        #     action_idx = move_to_action_index(move)
        #
        #     # Get prior probability for this action (from masked, renormalized policy)
        #     prior = masked_policy[action_idx]
        #
        #     # Create new game state
        #     next_state = self.game_state.clone()
        #     next_state.make_move(move)
        #
        #     # Create child node
        #     child = PUCTNode(next_state, parent=self, move=move, prior_prob=prior)
        #     self.children.append(child)

        # Priors should already sum to ~1 after masking, but double-check
        # total_prior = sum(child.prior_prob for child in self.children)
        # if total_prior > 0 and abs(total_prior - 1.0) > 1e-6:
        #     for child in self.children:
        #         child.prior_prob /= total_prior

        self.is_expanded = True
        return value


    def update(self, value):
        """
        Update node statistics with the result of a simulation

        Args:
            value: value from perspective of player to move at this node
                   (+1 = win, -1 = loss, 0 = draw)
        """
        self.visit_count += 1
        self.value_sum += value
        self.Q = self.value_sum / self.visit_count

    def backpropagate(self, value):
        """
        Backpropagate value up the tree
        Flips value for opponent's perspective
        """
        self.update(value)

        if self.parent:
            # Flip value for opponent
            self.parent.backpropagate(-value)


class PUCTPlayer:
    """
    PUCT-based AI player using neural network guidance
    """

    def __init__(self, network, num_simulations=800, c_puct=1.0, temperature=1.0, time_limit=None):
        """
        Args:
            network: GameNetwork for policy and value predictions
            num_simulations: number of PUCT simulations per move
            c_puct: exploration constant
            temperature: temperature for move selection (higher = more random)
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.time_limit = time_limit

    def get_move(self, game_state, return_probs=False, verbose=False):
        """
        Get best move using PUCT search

        Args:
            game_state: current game state
            return_probs: if True, also return visit count distribution
            verbose: if True, print search statistics

        Returns:
            best_move: chosen move
            visit_probs (optional): distribution of visit counts
        """
        root = PUCTNode(game_state.clone())

        # QUICK WIN CHECK: if any legal root move immediately creates SOS,
        # play it instantly (fast tactical awareness).
        for mv in game_state.legal_moves():
            tmp = game_state.clone()
            move_info = tmp.make_move(mv)
            if move_info[0] > 0:
                if return_probs:
                    # Must return visit_probs too — use one-hot for the chosen move
                    visit_probs_full = np.zeros(128, dtype=np.float32)
                    visit_probs_full[move_to_action_index(mv)] = 1.0
                    return mv, visit_probs_full
                return mv

        # Expand root once to get priors and optionally add Dirichlet noise
        root.expand(self.network)
        # If root has children, add Dirichlet noise to encourage exploration
        if root.children:
            eps = 0.25
            alpha = 0.03
            noise = np.random.dirichlet([alpha] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior_prob = child.prior_prob * (1 - eps) + noise[i] * eps

        # Run simulations (count or time-limited)
        import time
        start_time = time.time()

        if self.time_limit is None:
            sim_iter = range(self.num_simulations)
        else:
            sim_iter = iter(int, 1)

        for sim in sim_iter:
            # Stop if time limit reached
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break
            node = root
            search_path = [node]

            # Selection: traverse tree using PUCT
            while node.is_expanded and not node.is_terminal():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            # Expansion and evaluation
            if not node.is_terminal():
                # Expand node using network
                value = node.expand(self.network)

            # Evaluation: use network to evaluate leaf node
            else:
                # Terminal node - use actual game result
                result = node.game_state.status()
                if result == "draw":
                    value = 0
                elif result == node.player_to_move:
                    value = 1
                else:
                    value = -1

            # Backpropagation
            for node_in_path in reversed(search_path):
                node_in_path.update(value)
                value = -value  # Flip for opponent

            # Early stopping: if one root child dominates visits, stop early
            if sim % 10 == 0 and root.children:
                visits = np.array([child.visit_count for child in root.children])
                total = visits.sum()
                if total > 0:
                    if visits.max() / total > 0.95:
                        break

        # Select move based on visit counts
        if not root.children:
            # No legal moves
            return None

        # Get visit counts
        visits = np.array([child.visit_count for child in root.children])
        moves = [child.move for child in root.children]

        if verbose:
            print("\n--- PUCT Analysis ---")
            sorted_indices = np.argsort(visits)[::-1]
            for i, idx in enumerate(sorted_indices[:5]):
                child = root.children[idx]
                print(f"{i + 1}. Move {child.move}: visits={child.visit_count}, "
                      f"Q={child.Q:.3f}, P={child.prior_prob:.3f}")

        # Select move based on temperature
        if self.temperature == 0:
            # Greedy: always pick most visited
            best_idx = np.argmax(visits)
        else:
            # Sample from visit distribution with temperature
            visits_temp = visits ** (1.0 / self.temperature)
            probs = visits_temp / visits_temp.sum()
            best_idx = np.random.choice(len(visits), p=probs)

        best_move = moves[best_idx]

        if return_probs:
            # Return visit count distribution aligned with full action space (128 actions)
            # Illegal moves will have probability 0
            visit_probs_full = np.zeros(128, dtype=np.float32)
            for child in root.children:
                action_idx = move_to_action_index(child.move)
                visit_probs_full[action_idx] = child.visit_count
            
            # Normalize to sum to 1
            total_visits = visit_probs_full.sum()
            if total_visits > 0:
                visit_probs_full /= total_visits
            
            return best_move, visit_probs_full

        return best_move


# No top-level test code in this module; tests are run via separate scripts.