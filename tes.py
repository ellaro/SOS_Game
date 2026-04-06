from training import SelfPlayTrainer

trainer = SelfPlayTrainer()

# Test MCTS data generation
print("=== Testing MCTS games ===")
data = trainer.generate_mcts_games(num_games=3, num_simulations=50, verbose=True)
print(f"Number of samples: {len(data)}")

state, policy, value = data[0]
print(f"State shape: {len(state)}")
print(f"Policy shape: {len(policy)}, sum: {sum(policy):.3f}")
print(f"Value: {value} (should be between -1 and 1)")

# Test training
print("\n=== Testing training ===")
trainer.train(data, epochs=2, batch_size=8)

print("\nAll good!")