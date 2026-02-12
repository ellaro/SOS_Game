from training import initial_training_pipeline, iterative_training_pipeline
from network import GameNetwork
import time

print("=" * 70)
print("  FULL TRAINING PIPELINE - SOS GAME NEURAL NETWORK")
print("=" * 70)
print("\nThis will take approximately 45-60 minutes.")
print("You can leave it running and come back later.\n")

start_time = time.time()

# Phase 1: Initial training with pure MCTS
print("\n" + "=" * 70)
print("PHASE 1: INITIAL TRAINING WITH MCTS")
print("=" * 70)
print("Generating 1000 self-play games using pure MCTS...")
print("Then training network on these games for 20 epochs.")
print("Estimated time: 20-30 minutes\n")

phase1_start = time.time()

network = initial_training_pipeline(
    num_mcts_games=1000,  # 1000 games with MCTS
    epochs=20             # Train for 20 epochs
)

phase1_time = time.time() - phase1_start
print(f"\n✅ Phase 1 completed in {phase1_time/60:.1f} minutes")

# Phase 2: Iterative improvement with PUCT
print("\n" + "=" * 70)
print("PHASE 2: ITERATIVE IMPROVEMENT WITH PUCT")
print("=" * 70)
print("Running 5 iterations of:")
print("  - Generate 200 self-play games with PUCT + current network")
print("  - Train network on collected games for 5 epochs")
print("Estimated time: 25-30 minutes\n")

phase2_start = time.time()

network = iterative_training_pipeline(
    network=network,
    num_iterations=5,      # 5 iterations
    games_per_iter=200     # 200 games per iteration
)

phase2_time = time.time() - phase2_start
total_time = time.time() - start_time

# Final summary
print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE! 🎉")
print("=" * 70)
print(f"\nTotal training time: {total_time/60:.1f} minutes")
print(f"  - Phase 1 (MCTS): {phase1_time/60:.1f} minutes")
print(f"  - Phase 2 (PUCT): {phase2_time/60:.1f} minutes")
print("\nYour trained network has been saved!")
print("Look for files named: network_iter_5_YYYYMMDD_HHMMSS.pth")
print("\nNext steps:")
print("1. Test your network: python evaluate.py")
print("2. Play against it: python play_vs_network.py")
print("=" * 70)