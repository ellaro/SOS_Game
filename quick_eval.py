from main import SOSGame
from puct import PUCTPlayer
from mcts import MCTSPlayer
from network import GameNetwork

# Quick evaluation with lower simulations for speed
network = GameNetwork.load("network_mcts_20260203_223556.pth")
pu = PUCTPlayer(network, num_simulations=50, temperature=0)
mc = MCTSPlayer(num_simulations=50)

g = SOSGame()
move_count = 0
while g.status() is None and move_count < 64:
    if g.current_player == 0:
        move = pu.get_move(g)
    else:
        move = mc.get_move(g)
    if move is None:
        break
    g.make_move(move)
    move_count += 1

g.print_board()
print(f"Game finished in {move_count} moves")
g.print_board()
print(f"Winner: {g.status()}")
