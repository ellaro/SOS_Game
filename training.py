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
        Generate games using pure MCTS (no network)
        This is for initial training data

        Returns:
            training_data: list of (state, policy, value) tuples
        """
        print(f"\n=== Generating {num_games} MCTS self-play games ===")

        mcts_player = MCTSPlayer(num_simulations=num_simulations)
        game_data = []

        for game_num in range(num_games):
            game = SOSGame()
            game_history = []  # Store (state, move, player) for this game
            move_count = 0

            while game.status() is None and move_count < 64:
                # Get current state
                state = game.encode()
                current_player = game.current_player

                # Get move from MCTS
                move = mcts_player.get_move(game)

                # Store state and move
                game_history.append((state, move, current_player))

                # Make move
                game.make_move(move)
                move_count += 1

            # Game finished - assign values
            result = game.status()

            # Convert game history to training data
            for state, move, player in game_history:
                # Determine value from this player's perspective
                if result == "draw":
                    value = 0
                elif result == player:
                    value = 1
                else:
                    value = -1

                # Create policy target (one-hot for the move that was made)
                policy = np.zeros(128)
                action_idx = self._move_to_action_index(move)
                policy[action_idx] = 1.0

                game_data.append((state, policy, value))

            if verbose and (game_num + 1) % 10 == 0:
                print(f"Generated {game_num + 1}/{num_games} games | "
                      f"Last game: {move_count} moves, winner: {result}")

        print(f"✅ Generated {len(game_data)} training examples from {num_games} games")
        return game_data

    def generate_puct_games(self, num_games=100, num_simulations=400, temperature=1.0, verbose=True):
        """
        Generate games using PUCT with current network
        Stores visit count distributions as policy targets

        Returns:
            training_data: list of (state, policy, value) tuples
        """
        print(f"\n=== Generating {num_games} PUCT self-play games ===")

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

                # Get move and visit count distribution
                move, visit_probs = puct_player.get_move(game, return_probs=True)

                if move is None:
                    break

                # Convert visit_probs to full 128-dim policy
                # (visit_probs only has probabilities for legal moves)
                legal_moves = game.legal_moves()
                policy = np.zeros(128)

                for legal_move, prob in zip(legal_moves, visit_probs):
                    action_idx = self._move_to_action_index(legal_move)
                    policy[action_idx] = prob

                game_history.append((state, policy, current_player))
                game.make_move(move)
                move_count += 1

            # Assign values
            result = game.status()

            for state, policy, player in game_history:
                if result == "draw":
                    value = 0
                elif result == player:
                    value = 1
                else:
                    value = -1

                game_data.append((state, policy, value))

            if verbose and (game_num + 1) % 10 == 0:
                print(f"Generated {game_num + 1}/{num_games} games | "
                      f"Last game: {move_count} moves, winner: {result}")

        print(f"✅ Generated {len(game_data)} training examples from {num_games} games")
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
        print(f"\n=== Training network for {epochs} epochs ===")
        print(f"Training data size: {len(training_data)}")

        for epoch in range(epochs):
            avg_loss, avg_p_loss, avg_v_loss = self.trainer.train_epoch(
                training_data,
                batch_size=batch_size
            )

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Policy: {avg_p_loss:.4f} | "
                      f"Value: {avg_v_loss:.4f}")

        print("✅ Training complete!")

    def save_training_data(self, data, filename):
        """Save training data to file"""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Training data saved to {filename}")

    def load_training_data(self, filename):
        """Load training data from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Training data loaded from {filename} ({len(data)} examples)")
        return data


def initial_training_pipeline(num_mcts_games=1000, epochs=20):
    """
    Complete initial training pipeline:
    1. Generate games with pure MCTS
    2. Train network on these games
    3. Save network
    """
    print("=" * 60)
    print("INITIAL TRAINING PIPELINE")
    print("=" * 60)

    # Create trainer
    trainer = SelfPlayTrainer()

    # Step 1: Generate MCTS games
    print("\nStep 1: Generating initial training data with MCTS")
    training_data = trainer.generate_mcts_games(
        num_games=num_mcts_games,
        num_simulations=200
    )

    # Save training data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = f"training_data_mcts_{timestamp}.pkl"
    trainer.save_training_data(training_data, data_file)

    # Step 2: Train network
    print("\nStep 2: Training network")
    trainer.train(training_data, epochs=epochs, batch_size=64)

    # Step 3: Save network
    model_file = f"network_mcts_{timestamp}.pth"
    trainer.network.save(model_file)

    print("\n" + "=" * 60)
    print("✅ Initial training complete!")
    print(f"Model saved to: {model_file}")
    print(f"Data saved to: {data_file}")
    print("=" * 60)

    return trainer.network


def iterative_training_pipeline(network, num_iterations=5, games_per_iter=100):
    """
    Iterative improvement:
    1. Generate games with PUCT using current network
    2. Train network on new games
    3. Repeat
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

        # Generate games with current network
        new_data = trainer.generate_puct_games(
            num_games=games_per_iter,
            num_simulations=300,
            temperature=1.0
        )

        # Add to training pool
        all_training_data.extend(new_data)

        # Keep only recent data (e.g., last 5000 examples)
        if len(all_training_data) > 5000:
            all_training_data = all_training_data[-5000:]

        # Train on all data
        trainer.train(all_training_data, epochs=5, batch_size=64)

        # Save checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trainer.network.save(f"network_iter_{iteration + 1}_{timestamp}.pth")

    print("\n" + "=" * 60)
    print("✅ Iterative training complete!")
    print("=" * 60)

    return trainer.network


if __name__ == '__main__':
    # Quick test with small numbers
    print("=== Quick Test ===\n")

    # Test 1: Generate small amount of MCTS data
    trainer = SelfPlayTrainer()
    data = trainer.generate_mcts_games(num_games=5, num_simulations=100)

    # Test 2: Train on it
    trainer.train(data, epochs=3, batch_size=16)

    # Test 3: Generate PUCT games with trained network
    puct_data = trainer.generate_puct_games(num_games=3, num_simulations=100)

    print("\n✅ All tests passed!")
    print("\nTo run full training, uncomment one of these:")
    print("# initial_training_pipeline(num_mcts_games=1000, epochs=20)")
    print("# network = GameNetwork.load('your_network.pth')")
    print("# iterative_training_pipeline(network, num_iterations=5, games_per_iter=100)")