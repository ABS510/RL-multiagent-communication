from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(
        self,
        action,
        observation,
        reward,
        termination,
        truncation,
        info,
        next_observation,
    ):
        self.buffer.append(
            (
                action,
                observation,
                reward,
                termination,
                truncation,
                info,
                next_observation,
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        (
            action,
            observation,
            reward,
            termination,
            truncation,
            info,
            next_observation,
        ) = zip(*batch)
        return (
            action,
            observation,
            reward,
            termination,
            truncation,
            info,
            next_observation,
        )
