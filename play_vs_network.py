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
            print("No trained network found. Run training first.")
            return

    print(f"Loading network from: {network_file}")
    network = GameNetwork.load(network_file)

    # Create AI player
    ai = PUCTPlayer(network, num_simulations=400, temperature=0)

    print("Play against trained network")
    print("You are Player 0 (start); AI is Player 1")
    print("Enter moves as: row col letter (e.g., '2 3 S')")

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
                        print("Letter must be S or O")
                        continue

                    move = (r, c, letter)

                    if move in game.legal_moves():
                        game.make_move(move)
                        break
                    else:
                        print("Illegal move; cell occupied or out of bounds.")
                        print(f"Legal moves: {len(game.legal_moves())}")
                except ValueError:
                    print("Invalid input; use: row col letter (e.g., '2 3 S')")
                except KeyboardInterrupt:
                    print("Game interrupted")
                    return

        else:
            # AI's turn
            print("AI thinking...")
            move = ai.get_move(game, verbose=True)
            if move:
                print(f"AI plays: {move}")
                game.make_move(move)
            else:
                print("AI has no legal moves")
                break

        game.print_board()

    # Game over
    print("Game over")

    winner = game.status()
    if winner == 0:
        print("You won")
    elif winner == 1:
        print("AI won")
    else:
        print("Draw")

    print("Final scores:")
    print(f"  You: {game.scores[0]}")
    print(f"  AI:  {game.scores[1]}")


if __name__ == '__main__':
    play_human_vs_network()