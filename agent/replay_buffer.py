from collections import deque
import random
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.hyperparams import BUFFER_SIZE, RANDOM_SEED

random.seed(RANDOM_SEED)

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return {
            'states':      np.array(states,      dtype=np.float32),
            'actions':     np.array(actions,     dtype=np.int64),
            'rewards':     np.array(rewards,     dtype=np.float32),
            'next_states': np.array(next_states, dtype=np.float32),
            'dones':       np.array(dones,       dtype=np.float32),
        }

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, min_size):
        return len(self.buffer) >= min_size

if __name__ == "__main__":
    from configs.hyperparams import OBS_SIZE
    buf = ReplayBuffer(capacity=100)
    dummy_state = np.random.rand(OBS_SIZE).astype(np.float32)
    for i in range(50):
        buf.push(dummy_state, i % 3, float(i) * 0.1, dummy_state, i == 49)
    assert len(buf) == 50
    batch = buf.sample(15)
    assert batch['states'].shape    == (15, OBS_SIZE)
    assert batch['rewards'].dtype   == np.float32
    assert batch['dones'].dtype     == np.float32
    print("ReplayBuffer test PASSED.")
    print(f"Sample rewards range: [{batch['rewards'].min():.2f}, {batch['rewards'].max():.2f}]")