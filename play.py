from main import SOSGame
from mcts import MCTSPlayer
import random


def play_ai_vs_ai(num_simulations=500, verbose=True):
    """Play a game between two MCTS AI players"""
    game = SOSGame()
    ai_player = MCTSPlayer(num_simulations=num_simulations)

    move_count = 0

    if verbose:
        print("=== AI vs AI Game ===\n")
        game.print_board()

    while game.status() is None:
        move_count += 1

        if verbose:
            print(f"\n--- Move {move_count}: Player {game.current_player} ---")

        # AI chooses move
        move = ai_player.get_move(game)

        if verbose:
            print(f"AI chooses: {move}")

        game.make_move(move)

        if verbose:
            game.print_board()

    if verbose:
        print(f"\n🎮 Game Over after {move_count} moves!")
        print(f"Winner: Player {game.status()}")

    return game.status(), move_count


def play_human_vs_ai(num_simulations=500):
    """Play a game: Human vs AI"""
    game = SOSGame()
    ai_player = MCTSPlayer(num_simulations=num_simulations)

    print("=== Human vs AI ===")
    print("You are Player 0, AI is Player 1")
    game.print_board()

    while game.status() is None:
        if game.current_player == 0:
            # Human's turn
            print("\nYour turn! Enter move as: row col letter (e.g., '2 3 S')")

            while True:
                try:
                    user_input = input("> ").strip().split()
                    r, c = int(user_input[0]), int(user_input[1])
                    letter = user_input[2].upper()

                    move = (r, c, letter)

                    if move in game.legal_moves():
                        game.make_move(move)
                        break
                    else:
                        print("❌ Illegal move! Try again.")
                except:
                    print("❌ Invalid input! Format: row col letter (e.g., '2 3 S')")
        else:
            # AI's turn
            print("\n🤖 AI is thinking...")
            move = ai_player.get_move(game)
            print(f"AI plays: {move}")
            game.make_move(move)

        game.print_board()

    print(f"\n🎮 Game Over!")
    winner = game.status()
    if winner == 0:
        print("🎉 You won!")
    elif winner == 1:
        print("😔 AI won!")
    else:
        print("🤝 It's a draw!")


def benchmark_ai(num_games=10, num_simulations=500):
    """Run multiple games to benchmark the AI"""
    print(f"=== Running {num_games} AI vs AI games ===")
    print(f"Simulations per move: {num_simulations}\n")

    results = {0: 0, 1: 0, "draw": 0}
    total_moves = 0

    for i in range(num_games):
        print(f"Game {i + 1}/{num_games}...", end=" ")
        winner, moves = play_ai_vs_ai(num_simulations=num_simulations, verbose=False)
        results[winner] += 1
        total_moves += moves
        print(f"Winner: {winner}, Moves: {moves}")

    print("\n=== Results ===")
    print(f"Player 0 wins: {results[0]}")
    print(f"Player 1 wins: {results[1]}")
    print(f"Draws: {results['draw']}")
    print(f"Average moves per game: {total_moves / num_games:.1f}")


def test_mcts_quality():
    """Test if MCTS finds obvious winning moves"""
    print("=== Testing MCTS Quality ===\n")

    # Test 1: Should complete an obvious SOS
    print("Test 1: Complete obvious SOS")
    game = SOSGame()
    game.make_move((0, 0, 'S'))
    game.make_move((0, 1, 'O'))
    # Now (0, 2, 'S') should complete SOS and give immediate reward

    game.print_board()

    # Try with different simulation counts
    for num_sims in [100, 500, 1000]:
        ai = MCTSPlayer(num_simulations=num_sims)
        move = ai.get_move(game, verbose=(num_sims >= 500))

        print(f"\nWith {num_sims} simulations:")
        print(f"  AI chose: {move}")
        print(f"  Expected: (0, 2, 'S')")

        # Test the move
        test_game = game.clone()
        test_game.make_move((0, 2, 'S'))
        print(
            f"  Score after (0,2,S): Player {game.current_player} would get {test_game.scores[game.current_player]} points")

        if move == (0, 2, 'S'):
            print(f"  ✅ Found it with {num_sims} simulations!")
            return
        else:
            print(f"  ❌ Didn't find it")

    print("\n❌ MCTS failed to find obvious winning move even with 1000 sims")


def test_mcts_basic():
    """Test that MCTS can play a complete game without errors"""
    print("\n=== Test 2: Complete Game ===")

    game = SOSGame()
    ai = MCTSPlayer(num_simulations=100)

    move_count = 0
    max_moves = 64  # Board size

    while game.status() is None and move_count < max_moves:
        move = ai.get_move(game)
        game.make_move(move)
        move_count += 1

    print(f"Game completed in {move_count} moves")
    print(f"Winner: {game.status()}")
    print(f"Scores: Player 0: {game.scores[0]}, Player 1: {game.scores[1]}")

    if game.status() is not None:
        print("✅ Game completed successfully!\n")
    else:
        print("❌ Game didn't finish\n")

if __name__ == '__main__':
    # Uncomment the test you want to run:

    # Option 1: Watch one AI vs AI game
    play_ai_vs_ai(num_simulations=500, verbose=True)

    # Option 2: Play against the AI
    # play_human_vs_ai(num_simulations=500)

    # Option 3: Run benchmark
    # benchmark_ai(num_games=10, num_simulations=300)

    # Option 4: Test MCTS quality
    # test_mcts_quality()