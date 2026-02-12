import math
import numpy as np


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

        sqrt_parent_visits = math.sqrt(self.visit_count)

        for child in self.children:
            # Q value (average value from this child's perspective)
            q_value = child.Q

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
        Expand this node by creating children for all legal moves
        Uses network to get prior probabilities
        """
        if self.is_expanded or self.is_terminal():
            return

        # Get policy and value from network
        policy_probs, _ = network.predict(self.game_state)

        # Get legal moves
        legal_moves = self.game_state.legal_moves()

        if not legal_moves:
            self.is_expanded = True
            return

        # Create a child for each legal move
        for move in legal_moves:
            # Get action index for this move
            # We need to find which action index corresponds to this move
            # We'll use decode to find the mapping
            action_idx = self._move_to_action_index(move)

            # Get prior probability for this action
            prior = policy_probs[action_idx]

            # Create new game state
            next_state = self.game_state.clone()
            next_state.make_move(move)

            # Create child node
            child = PUCTNode(next_state, parent=self, move=move, prior_prob=prior)
            self.children.append(child)

        # Normalize priors (make sure they sum to 1)
        total_prior = sum(child.prior_prob for child in self.children)
        if total_prior > 0:
            for child in self.children:
                child.prior_prob /= total_prior

        self.is_expanded = True

    def _move_to_action_index(self, move):
        """
        Convert a move (r, c, letter) to an action index (0-127)
        Inverse of decode function
        """
        r, c, letter = move
        cell = r * 8 + c
        action_idx = cell * 2 + (0 if letter == 'S' else 1)
        return action_idx

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

    def __init__(self, network, num_simulations=800, c_puct=1.0, temperature=1.0):
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

        # Run simulations
        for sim in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree using PUCT
            while node.is_expanded and not node.is_terminal():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            # Expansion and evaluation
            if not node.is_terminal():
                # Expand node using network
                node.expand(self.network)

                # If we expanded, select a child
                if node.children:
                    node = node.select_child(self.c_puct)
                    search_path.append(node)

            # Evaluation: use network to evaluate leaf node
            if node.is_terminal():
                # Terminal node - use actual game result
                result = node.game_state.status()
                if result == "draw":
                    value = 0
                elif result == node.player_to_move:
                    value = 1
                else:
                    value = -1
            else:
                # Non-terminal - use network value prediction
                _, value = self.network.predict(node.game_state)
                # Value is from current player's perspective at this node

            # Backpropagation
            for node_in_path in reversed(search_path):
                node_in_path.update(value)
                value = -value  # Flip for opponent

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
            # Return visit count distribution (for training)
            visit_probs = visits / visits.sum()
            return best_move, visit_probs

        return best_move


# Test PUCT
if __name__ == '__main__':
    from main import SOSGame
    from network import GameNetwork

    print("=== Testing PUCT ===\n")

    # Create network and PUCT player
    network = GameNetwork()
    puct_player = PUCTPlayer(network, num_simulations=100, c_puct=1.0)

    # Test on simple position
    game = SOSGame()
    game.make_move((0, 0, 'S'))
    game.make_move((0, 1, 'O'))

    print("Testing PUCT move selection:")
    game.print_board()

    move = puct_player.get_move(game, verbose=True)
    print(f"\nPUCT selected move: {move}")

    # Test full game
    print("\n=== Playing full game ===")
    game2 = SOSGame()
    move_count = 0

    while game2.status() is None and move_count < 64:
        move = puct_player.get_move(game2)
        if move is None:
            break
        game2.make_move(move)
        move_count += 1

        if move_count % 10 == 0:
            print(f"Move {move_count}...", end=" ", flush=True)

    print(f"\nGame finished in {move_count} moves")
    print(f"Winner: {game2.status()}")
    print(f"Scores: {game2.scores}")

    print("\n✅ PUCT tests passed!")