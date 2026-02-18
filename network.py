import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class GameNetwork(nn.Module):
    """
    Neural Network for SOS Game

    Input: encoded game state (131 values)
        - Board: 8x8x2 = 128 (S/O encoding)
        - Current player: 1
        - Scores: 2

    Outputs:
        - Value head: single value in [-1, 1] (win probability)
        - Policy head: 128 values (probabilities for each action)
    """

    def __init__(self, input_size=131, hidden_size=256, action_size=128):
        super(GameNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        # Policy head (predicts move probabilities)
        self.policy_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.policy_fc2 = nn.Linear(hidden_size // 2, action_size)

        # Value head (predicts win probability)
        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.value_fc2 = nn.Linear(hidden_size // 4, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: tensor of shape (batch_size, input_size) or (input_size,)

        Returns:
            policy: tensor of shape (batch_size, action_size) - action probabilities
            value: tensor of shape (batch_size, 1) - win probability in [-1, 1]
        """
        # Handle single input (convert to batch)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Shared layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Policy head
        policy = F.relu(self.policy_fc1(x))
        policy = self.policy_fc2(policy)
        policy = F.log_softmax(policy, dim=1)  # Log probabilities

        # Value head
        value = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]

        return policy, value

    def predict(self, game_state):
        """
        Predict policy and value for a single game state

        Args:
            game_state: SOSGame object

        Returns:
            policy_probs: numpy array of shape (action_size,)
            value: float in [-1, 1]
        """
        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            # Encode game state
            state_encoded = game_state.encode()
            state_tensor = torch.FloatTensor(state_encoded)

            # Get predictions
            log_policy, value = self.forward(state_tensor)

            # Convert to probabilities
            policy_probs = torch.exp(log_policy).squeeze(0).numpy()
            value_scalar = value.item()

        return policy_probs, value_scalar

    def save(self, filepath):
        """Save model weights to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'action_size': self.action_size,
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model weights from file"""
        checkpoint = torch.load(filepath)

        # Create model with saved architecture
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            action_size=checkpoint['action_size']
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")

        return model


class NetworkTrainer:
    """
    Helper class for training the network
    """

    def __init__(self, network, lr=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    def train_step(self, states, target_policies, target_values):
        """
        Single training step

        Args:
            states: list of encoded game states
            target_policies: list of target policy distributions (from MCTS visit counts)
            target_values: list of target values (game outcomes: -1, 0, 1)

        Returns:
            total_loss, policy_loss, value_loss
        """
        self.network.train()

        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        target_policies_tensor = torch.FloatTensor(target_policies)
        target_values_tensor = torch.FloatTensor(target_values).unsqueeze(1)

        # Forward pass
        log_policies, values = self.network(states_tensor)

        # Compute losses
        # Policy loss: cross-entropy between target and predicted policy
        policy_loss = -torch.mean(torch.sum(target_policies_tensor * log_policies, dim=1))

        # Value loss: MSE between target and predicted value
        value_loss = F.mse_loss(values, target_values_tensor)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()

    def train_epoch(self, training_data, batch_size=32):
        """
        Train for one epoch

        Args:
            training_data: list of (state, policy, value) tuples
            batch_size: batch size for training

        Returns:
            average losses
        """
        # Shuffle data
        np.random.shuffle(training_data)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        # Train in batches
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]

            states = [item[0] for item in batch]
            policies = [item[1] for item in batch]
            values = [item[2] for item in batch]

            loss, p_loss, v_loss = self.train_step(states, policies, values)

            total_loss += loss
            total_policy_loss += p_loss
            total_value_loss += v_loss
            num_batches += 1

        return (total_loss / num_batches,
                total_policy_loss / num_batches,
                total_value_loss / num_batches)


# No top-level test/demo code here; use the separate test scripts if needed.