from main import SOSGame
from puct import PUCTPlayer
from network import GameNetwork
import os
import glob


def find_latest_network():
    """Find the most recent trained network file"""
    network_files = glob.glob("network_*.pth")
    if not network_files:
        return None
    # Sort by modification time
    latest = max(network_files, key=os.path.getmtime)
    return latest


def play_human_vs_network(network_file=None):
    """Play against the trained network"""

    # Load network
    if network_file is None:
        network_file = find_latest_network()
        if network_file is None:
            print("No trained network found!")
            print("Please run: python run_training.py first")
            return

    print(f"Loading network from: {network_file}")
    network = GameNetwork.load(network_file)

    # Create AI player
    ai = PUCTPlayer(network, num_simulations=400, temperature=0)

    print("\n" + "=" * 60)
    print("  PLAY AGAINST THE TRAINED NEURAL NETWORK!")
    print("=" * 60)
    print("\nYou are Player 0 (starts first)")
    print("AI is Player 1")
    print("Enter moves as: row col letter (e.g., '2 3 S')")
    print("=" * 60 + "\n")

    game = SOSGame()
    game.print_board()

    while game.status() is None:
        if game.current_player == 0:
            # Human's turn
            print(f"\n{'=' * 40}")
            print("YOUR TURN (Player 0)")
            print(f"{'=' * 40}")

            while True:
                try:
                    user_input = input("Enter move (row col letter): ").strip().split()
                    if len(user_input) != 3:
                        print("Invalid format! Use: row col letter (e.g., '2 3 S')")
                        continue

                    r, c = int(user_input[0]), int(user_input[1])
                    letter = user_input[2].upper()

                    if letter not in ['S', 'O']:
                        print("Letter must be 'S' or 'O'")
                        continue

                    move = (r, c, letter)

                    if move in game.legal_moves():
                        game.make_move(move)
                        break
                    else:
                        print("❌ Illegal move! Cell is occupied or out of bounds.")
                        print(f"Legal moves: {len(game.legal_moves())} available")
                except ValueError:
                    print("❌ Invalid input! Use format: row col letter (e.g., '2 3 S')")
                except KeyboardInterrupt:
                    print("\n\nGame interrupted!")
                    return

        else:
            # AI's turn
            print(f"\n{'=' * 40}")
            print("AI THINKING... (Player 1)")
            print(f"{'=' * 40}")

            move = ai.get_move(game, verbose=True)
            if move:
                print(f"AI plays: {move}")
                game.make_move(move)
            else:
                print("AI has no legal moves!")
                break

        game.print_board()

    # Game over
    print("\n" + "=" * 60)
    print("  GAME OVER!")
    print("=" * 60)

    winner = game.status()
    if winner == 0:
        print("🎉 YOU WON! 🎉")
    elif winner == 1:
        print("🤖 AI WON! 🤖")
    else:
        print("🤝 IT'S A DRAW! 🤝")

    print(f"\nFinal Scores:")
    print(f"  You (Player 0): {game.scores[0]}")
    print(f"  AI  (Player 1): {game.scores[1]}")
    print("=" * 60)


if __name__ == '__main__':
    play_human_vs_network()