from main import SOSGame
from puct import PUCTPlayer
from mcts import MCTSPlayer
from network import GameNetwork

print("=" * 70)
print("  TESTING TRAINED NETWORK")
print("=" * 70)

# Load the trained network
print("\nLoading trained network...")
network = GameNetwork.load("network_mcts_20260203_223556.pth")

# Create players
puct_player = PUCTPlayer(network, num_simulations=200, temperature=0)
mcts_player = MCTSPlayer(num_simulations=200)

print("\n✅ Network loaded!")

# Test 1: PUCT finds obvious move
print("\n" + "=" * 70)
print("TEST 1: Can PUCT find obvious winning move?")
print("=" * 70)

game = SOSGame()
game.make_move((0, 0, 'S'))
game.make_move((0, 1, 'O'))

game.print_board()

print("\nExpected: (0, 2, 'S') - completes SOS")
move = puct_player.get_move(game, verbose=True)
print(f"\nPUCT chose: {move}")

if move == (0, 2, 'S'):
    print("✅ SUCCESS! PUCT found the winning move!")
else:
    print("❌ PUCT didn't find it (but that's okay, it's still learning)")

# Test 2: Play a full game
print("\n" + "=" * 70)
print("TEST 2: PUCT vs MCTS - Full Game")
print("=" * 70)

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
    print("🎉 PUCT (trained network) WON!")
elif winner == 1:
    print("🤖 MCTS (pure search) won")
else:
    print("🤝 Draw")

print(f"\nScores:")
print(f"  PUCT: {game2.scores[0]}")
print(f"  MCTS: {game2.scores[1]}")

print("\n" + "=" * 70)
print("  TESTING COMPLETE")
print("=" * 70)
print("\nYour network is trained and working!")
print("You can now use it for Exercise 6.")