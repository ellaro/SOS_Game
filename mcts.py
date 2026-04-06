import random
import math
import time
import numpy as np


def in_bounds(r, c):
    """Return True if (r,c) is inside the 8x8 board."""
    return 0 <= r < 8 and 0 <= c < 8


def move_to_action_index(move):
    """Convert (row, col, letter) to action index 0-127."""
    r, c, letter = move
    cell = r * 8 + c
    return cell * 2 + (0 if letter == 'S' else 1)


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.value_sum = 0
        self.visits = 0
        self.untried_moves = game_state.legal_moves()
        self.player_to_move = game_state.current_player

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.game_state.status() is not None

    def best_child(self, c_param=1.0):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                ucb_value = float('inf')
            else:
                child_avg = child.value_sum / child.visits
                if child.player_to_move == self.player_to_move:
                    avg_value = child_avg
                else:
                    avg_value = 1.0 - child_avg
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

        next_state = self.game_state.clone()
        next_state.make_move(move)
        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        state = self.game_state.clone()
        while state.status() is None:
            moves = state.legal_moves()
            if not moves:
                break

            sos_moves = []
            for m in moves:
                r, c, letter = m
                if letter == 'S':
                    directions = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
                    for dr, dc in directions:
                        r2, c2 = r + dr, c + dc
                        r3, c3 = r + 2*dr, c + 2*dc
                        if in_bounds(r2, c2) and in_bounds(r3, c3):
                            if state.board[r2][c2] == 'O' and state.board[r3][c3] == 'S':
                                sos_moves.append(m)
                                break
                else:
                    basic_directions = [(0,1),(1,0),(1,1),(1,-1)]
                    for dr, dc in basic_directions:
                        r_before, c_before = r - dr, c - dc
                        r_after, c_after = r + dr, c + dc
                        if in_bounds(r_before, c_before) and in_bounds(r_after, c_after):
                            if state.board[r_before][c_before] == 'S' and state.board[r_after][c_after] == 'S':
                                sos_moves.append(m)
                                break

            move = random.choice(sos_moves) if sos_moves else random.choice(moves)
            state.make_move(move)

        result = state.status()
        return result, state.scores

    def backpropagate(self, result, final_scores):
        self.visits += 1
        if result == "draw":
            value = 0.5
        elif result == self.player_to_move:
            value = 1.0
        else:
            value = 0.0
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(result, final_scores)


class MCTSPlayer:
    def __init__(self, num_simulations=200, time_limit=None):
        self.num_simulations = num_simulations
        self.time_limit = time_limit

    def get_move(self, game_state, return_probs=False, verbose=False):
        """
        Get the best move using MCTS.
        If return_probs=True, also returns a 128-dim visit count distribution
        (used for AlphaZero training as the policy target).
        """
        root = MCTSNode(game_state.clone())

        # QUICK WIN CHECK: if any move immediately creates SOS, play it.
        best_move = None
        max_sos = 0
        for mv in game_state.legal_moves():
            tmp = game_state.clone()
            move_info = tmp.make_move(mv)
            if move_info[0] > max_sos:
                max_sos = move_info[0]
                best_move = mv

        if best_move is not None:
            if return_probs:
                # Return a one-hot distribution for the winning move
                visit_probs = np.zeros(128, dtype=np.float32)
                visit_probs[move_to_action_index(best_move)] = 1.0
                return best_move, visit_probs
            return best_move

        # Run simulations
        start_time = time.time()
        if self.time_limit is None:
            sim_iter = range(self.num_simulations)
        else:
            sim_iter = iter(int, 1)

        for sim_i, _ in enumerate(sim_iter):
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break

            node = root

            # 1. Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # 2. Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # 3. Simulation
            result, final_scores = node.rollout()

            # 4. Backpropagation
            node.backpropagate(result, final_scores)

            # Early stopping
            if sim_i % 10 == 0 and root.children:
                total_visits = sum(c.visits for c in root.children)
                if total_visits > 0:
                    top_visits = max(c.visits for c in root.children)
                    if top_visits / total_visits > 0.95:
                        break

        if verbose and root.children:
            print("\n--- MCTS Analysis ---")
            sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
            for i, child in enumerate(sorted_children[:5]):
                win_rate = child.value_sum / child.visits if child.visits > 0 else 0
                print(f"{i+1}. Move {child.move}: visits={child.visits}, avg_value={win_rate:.3f}")

        if not root.children:
            legal = game_state.legal_moves()
            best_move = random.choice(legal) if legal else None
            if return_probs:
                visit_probs = np.zeros(128, dtype=np.float32)
                if best_move:
                    visit_probs[move_to_action_index(best_move)] = 1.0
                return best_move, visit_probs
            return best_move

        best_child = max(root.children, key=lambda c: c.visits)

        if return_probs:
            # Build the policy target: visit count distribution over all 128 actions
            visit_probs = np.zeros(128, dtype=np.float32)
            for child in root.children:
                visit_probs[move_to_action_index(child.move)] = child.visits
            total = visit_probs.sum()
            if total > 0:
                visit_probs /= total
            return best_child.move, visit_probs

        return best_child.move