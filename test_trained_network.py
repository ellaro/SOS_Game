from main import SOSGame
from puct import PUCTPlayer
from mcts import MCTSPlayer
from network import GameNetwork

print("TESTING TRAINED NETWORK")

# Load trained network
network = GameNetwork.load("network_mcts_20260203_223556.pth")

# Create players
puct_player = PUCTPlayer(network, num_simulations=200, temperature=0)
mcts_player = MCTSPlayer(num_simulations=200)

print("Network loaded")

# Test 1: PUCT finds obvious move
print("TEST 1: PUCT finds obvious move")

game = SOSGame()
game.make_move((0, 0, 'S'))
game.make_move((0, 1, 'O'))

game.print_board()

print("Expected move: (0, 2, 'S')")
move = puct_player.get_move(game, verbose=True)
print(f"\nPUCT chose: {move}")

if move == (0, 2, 'S'):
    print("PUCT found the winning move")
else:
    print("PUCT did not find the winning move")

# Test 2: Play a full game
print("TEST 2: PUCT vs MCTS - full game")

game2 = SOSGame()
move_count = 0

print("\nPlaying game...")
while game2.status() is None and move_count < 64:
    if game2.current_player == 0:
        move = puct_player.get_move(game2)
    else:
        move = mcts_player.get_move(game2)

    if move is None:
        break

    game2.make_move(move)
    move_count += 1

    if move_count % 10 == 0:
        print(f"Move {move_count}...", end=" ", flush=True)

print(f"\n\nGame finished in {move_count} moves")
game2.print_board()

winner = game2.status()
    if winner == 0:
        print("PUCT won")
    elif winner == 1:
        print("MCTS won")
    else:
        print("Draw")

print(f"\nScores:")
print(f"  PUCT: {game2.scores[0]}")
print(f"  MCTS: {game2.scores[1]}")

print("TESTING COMPLETE")
print("Network appears to be working")