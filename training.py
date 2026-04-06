import numpy as np
import random
from main import SOSGame
from mcts import MCTSPlayer
from puct import PUCTPlayer
from network import GameNetwork, NetworkTrainer
import pickle
import os
from datetime import datetime


class SelfPlayTrainer:
    """
    Manages self-play games and network training
    """

    def __init__(self, network=None):
        self.network = network if network else GameNetwork()
        self.trainer = NetworkTrainer(self.network, lr=0.001)
        self.training_data = []

    def generate_mcts_games(self, num_games=100, num_simulations=300, verbose=True):
        """
        Generate games using pure MCTS (no network).
        Policy target = visit count distribution (not one-hot).
        Value target = normalized score difference.
        """
        print(f"Generating {num_games} MCTS self-play games")

        mcts_player = MCTSPlayer(num_simulations=num_simulations)
        game_data = []

        for game_num in range(num_games):
            game = SOSGame()
            game_history = []  # stores (state, visit_probs, current_player)
            move_count = 0

            while game.status() is None and move_count < 64:
                state = game.encode()
                current_player = game.current_player

                # Get move AND visit count distribution from MCTS
                move, visit_probs = mcts_player.get_move(game, return_probs=True)

                # Store state, policy (visit counts), and player
                game_history.append((state, visit_probs, current_player))

                game.make_move(move)
                move_count += 1

            # Game finished
            result = game.status()
            final_scores = game.scores

            for state, policy, player in game_history:
                # Normalized score difference as value target
                score_diff = final_scores[player] - final_scores[1 - player]
                value = max(-1.0, min(1.0, score_diff / 10.0))

                game_data.append((state, policy, value))

            if verbose and (game_num + 1) % 10 == 0:
                print(f"{game_num + 1}/{num_games} games generated; last winner: {result}")

        print(f"Generated {len(game_data)} training examples")
        return game_data

    def generate_puct_games(self, num_games=100, num_simulations=400, temperature=1.0, verbose=True):
        """
        Generate games using PUCT with current network.
        Policy target = visit count distribution from PUCT (already 128-dim).
        Value target = normalized score difference.
        """
        print(f"Generating {num_games} PUCT self-play games")

        puct_player = PUCTPlayer(
            self.network,
            num_simulations=num_simulations,
            temperature=temperature
        )
        game_data = []

        for game_num in range(num_games):
            game = SOSGame()
            game_history = []
            move_count = 0

            while game.status() is None and move_count < 64:
                state = game.encode()
                current_player = game.current_player

                # Get move and visit count distribution (already 128-dim normalized)
                move, visit_probs = puct_player.get_move(game, return_probs=True)

                if move is None:
                    break

                # visit_probs is already a 128-dim normalized array from PUCT
                policy = visit_probs

                game_history.append((state, policy, current_player))
                game.make_move(move)
                move_count += 1

            # Game finished
            result = game.status()
            final_scores = game.scores

            for state, policy, player in game_history:
                # Normalized score difference as value target
                score_diff = final_scores[player] - final_scores[1 - player]
                value = max(-1.0, min(1.0, score_diff / 10.0))

                game_data.append((state, policy, value))

            if verbose and (game_num + 1) % 10 == 0:
                print(f"{game_num + 1}/{num_games} games generated; last winner: {result}")

        print(f"Generated {len(game_data)} training examples")
        return game_data

    def _move_to_action_index(self, move):
        """Convert move to action index"""
        r, c, letter = move
        cell = r * 8 + c
        return cell * 2 + (0 if letter == 'S' else 1)

    def train(self, training_data, epochs=10, batch_size=32, verbose=True):
        """
        Train network on collected data
        """
        print(f"Training network for {epochs} epochs")
        print(f"Training data size: {len(training_data)}")

        for epoch in range(epochs):
            avg_loss, avg_p_loss, avg_v_loss = self.trainer.train_epoch(
                training_data,
                batch_size=batch_size
            )

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss {avg_loss:.4f} "
                      f"(policy: {avg_p_loss:.4f}, value: {avg_v_loss:.4f})")

        print("Training complete")

    def save_training_data(self, data, filename):
        """Save training data to file"""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Training data saved to {filename}")

    def load_training_data(self, filename):
        """Load training data from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Training data loaded: {len(data)} examples")
        return data


def initial_training_pipeline(num_mcts_games=1000, epochs=20):
    """
    Phase 1: Generate games with pure MCTS, train network, save.
    """
    print("=" * 60)
    print("INITIAL TRAINING PIPELINE")
    print("=" * 60)

    trainer = SelfPlayTrainer()

    print("\nStep 1: Generating initial training data with MCTS")
    training_data = trainer.generate_mcts_games(
        num_games=num_mcts_games,
        num_simulations=200
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = f"training_data_mcts_{timestamp}.pkl"
    trainer.save_training_data(training_data, data_file)

    print("\nStep 2: Training network")
    trainer.train(training_data, epochs=epochs, batch_size=64)

    model_file = f"network_mcts_{timestamp}.pth"
    trainer.network.save(model_file)

    print("\n" + "=" * 60)
    print("Initial training complete")
    print(f"Model saved to: {model_file}")
    print(f"Data saved to:  {data_file}")
    print("=" * 60)

    return trainer.network


def iterative_training_pipeline(network, num_iterations=5, games_per_iter=100):
    """
    Phase 2: Iteratively improve network using PUCT self-play.
    """
    print("=" * 60)
    print("ITERATIVE TRAINING PIPELINE")
    print("=" * 60)

    trainer = SelfPlayTrainer(network)
    all_training_data = []

    for iteration in range(num_iterations):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'=' * 60}")

        new_data = trainer.generate_puct_games(
            num_games=games_per_iter,
            num_simulations=300,
            temperature=1.0
        )

        all_training_data.extend(new_data)

        # Keep only the most recent 5000 examples (replay buffer)
        if len(all_training_data) > 5000:
            all_training_data = all_training_data[-5000:]

        trainer.train(all_training_data, epochs=5, batch_size=64)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trainer.network.save(f"network_iter_{iteration + 1}_{timestamp}.pth")

    print("\n" + "=" * 60)
    print("Iterative training complete")
    print("=" * 60)

    return trainer.network


if __name__ == '__main__':
    print("Quick test")

    trainer = SelfPlayTrainer()
    data = trainer.generate_mcts_games(num_games=5, num_simulations=100)
    trainer.train(data, epochs=3, batch_size=16)

    puct_data = trainer.generate_puct_games(num_games=3, num_simulations=100)

    print("All tests passed")