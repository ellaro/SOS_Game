from main import SOSGame
from puct import PUCTPlayer, action_index_to_move, move_to_action_index
import torch
import numpy as np


def play_game(player1, player2):
    game = SOSGame()
    while game.status() is None:
        if game.current_player == 0:
            move = player1.get_move(game)
        else:
            move = player2.get_move(game)
        game.make_move(move)
    return game.status()


def generate_self_play_game(network, num_simulations=200):
    """
    Plays one self-play game using PUCT and collects training data.
    Returns:
        list of (state_tensor, policy_target, value_target)
    """
    game = SOSGame()
    player = PUCTPlayer(network, num_simulations=num_simulations, temperature=1.0)

    memory = []  # (state_tensor, visit_probs, player)

    while game.status() is None:
        state_tensor = game.to_tensor()

        result = player.get_move(game, return_probs=True)
        move = result[0]
        visit_probs = result[1]

        # Assert BEFORE making the move
        legal_moves = game.legal_moves()
        assert move in legal_moves, (
            f"BUG: PUCT returned illegal move {move}. "
            f"Legal moves: {legal_moves}"
        )
        assert isinstance(visit_probs, np.ndarray), (
            f"visit_probs must be numpy array, got {type(visit_probs)}"
        )
        assert visit_probs.shape == (128,), (
            f"visit_probs must have shape (128,), got {visit_probs.shape}"
        )

        memory.append((
            state_tensor,
            visit_probs,
            game.current_player
        ))

        game.make_move(move)  # Only ONE make_move

    # Game finished
    result = game.status()  # 0 / 1 / draw

    training_data = []

    for state_tensor, visit_probs, player_at_state in memory:
        if result not in [0, 1]:
            value = 0
        elif result == player_at_state:
            value = 1
        else:
            value = -1

        training_data.append((
            state_tensor,
            torch.tensor(visit_probs, dtype=torch.float32),
            torch.tensor(value, dtype=torch.float32)
        ))

    return training_data


if __name__ == "__main__":
    from network import GameNetwork

    net = GameNetwork()
    data = generate_self_play_game(net, num_simulations=50)
    print("Number of samples:", len(data))
    print("First sample:")
    print(data[0])