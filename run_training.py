from training import initial_training_pipeline, iterative_training_pipeline
from network import GameNetwork
import time

print("FULL TRAINING PIPELINE")
print("This may take some time depending on configuration.")

start_time = time.time()

# Phase 1: Initial training with pure MCTS
print("Phase 1: generating MCTS self-play data and training")

phase1_start = time.time()

network = initial_training_pipeline(
    num_mcts_games=1000,  # 1000 games with MCTS
    epochs=20             # Train for 20 epochs
)

phase1_time = time.time() - phase1_start
print(f"Phase 1 completed in {phase1_time/60:.1f} minutes")

# Phase 2: Iterative improvement with PUCT
print("Phase 2: iterative improvement with PUCT")
print("Running iterations of self-play and training")

phase2_start = time.time()

network = iterative_training_pipeline(
    network=network,
    num_iterations=5,      # 5 iterations
    games_per_iter=200     # 200 games per iteration
)

phase2_time = time.time() - phase2_start
total_time = time.time() - start_time

# Final summary
print("Training complete")
print(f"Total training time: {total_time/60:.1f} minutes")
print(f"  - Phase 1: {phase1_time/60:.1f} minutes")
print(f"  - Phase 2: {phase2_time/60:.1f} minutes")
print("Training checkpoints saved to disk.")