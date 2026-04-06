import random
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size=10000):
        """
        Args:
            max_size: maximum number of samples to store
        """
        self.buffer = deque(maxlen=max_size)

    def add(self, game_data):
        for sample in game_data:
            self.buffer.append(sample)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)