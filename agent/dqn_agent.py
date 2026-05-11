import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer
from configs.hyperparams import (
    GAMMA, REPLAY_START, EPS_MAX, EPS_MIN, DECAY_RATE,
    BATCH_SIZE, TARGET_UPDATE, BUFFER_SIZE, LR,
    K_PATHS, RANDOM_SEED, CKPT_DIR
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class DQNAgent:
    def __init__(self, device='cpu'):
        self.device   = torch.device(device)
        self.online   = QNetwork().to(self.device)
        self.target   = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim    = optim.Adam(self.online.parameters(), lr=LR)
        self.buffer   = ReplayBuffer(BUFFER_SIZE)
        self.steps    = 0
        self.epsilon  = EPS_MAX
        self.losses   = []

    def act(self, state):
        self.epsilon = max(EPS_MIN, EPS_MAX - self.steps * DECAY_RATE)
        x = random.random()
        if x < self.epsilon:
            return random.randint(0, K_PATHS - 1)
        else:
            return self.online.best_action(state, self.device)

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.steps += 1

    def train_step(self):
        if not self.buffer.is_ready(REPLAY_START):
            return None

        batch = self.buffer.sample(BATCH_SIZE)
        S  = torch.FloatTensor(batch['states']).to(self.device)
        A  = torch.LongTensor(batch['actions']).to(self.device)
        R  = torch.FloatTensor(batch['rewards']).to(self.device)
        NS = torch.FloatTensor(batch['next_states']).to(self.device)
        D  = torch.FloatTensor(batch['dones']).to(self.device)

        q_all  = self.online(S)
        q_pred = q_all.gather(1, A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next   = self.target(NS).min(1)[0]
            q_target = R + GAMMA * q_next * (1.0 - D)

        loss = nn.MSELoss()(q_pred, q_target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optim.step()
        self.losses.append(loss.item())

        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def save(self, filename):
        path = os.path.join(CKPT_DIR, filename)
        torch.save({
            'online_state_dict': self.online.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
        }, path)
        print(f"Model saved: {path}")

    def load(self, filename):
        path = os.path.join(CKPT_DIR, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt['online_state_dict'])
        self.target.load_state_dict(ckpt['online_state_dict'])
        self.steps   = ckpt.get('steps', 0)
        self.epsilon = ckpt.get('epsilon', EPS_MIN)
        print(f"Model loaded: {path}")

class DDQNAgent(DQNAgent):
    def train_step(self):
        if not self.buffer.is_ready(REPLAY_START):
            return None

        batch = self.buffer.sample(BATCH_SIZE)
        S  = torch.FloatTensor(batch['states']).to(self.device)
        A  = torch.LongTensor(batch['actions']).to(self.device)
        R  = torch.FloatTensor(batch['rewards']).to(self.device)
        NS = torch.FloatTensor(batch['next_states']).to(self.device)
        D  = torch.FloatTensor(batch['dones']).to(self.device)

        q_all  = self.online(S)
        q_pred = q_all.gather(1, A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a   = self.online(NS).argmin(1, keepdim=True)
            q_next   = self.target(NS).gather(1, best_a).squeeze(1)
            q_target = R + GAMMA * q_next * (1.0 - D)

        loss = nn.MSELoss()(q_pred, q_target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optim.step()
        self.losses.append(loss.item())

        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

if __name__ == "__main__":
    from configs.hyperparams import OBS_SIZE
    agent = DQNAgent(device='cpu')
    dummy_state = np.random.rand(OBS_SIZE).astype(np.float32)
    for i in range(REPLAY_START + 10):
        agent.store(dummy_state, i % 3, float(i % 5) * 0.2, dummy_state, i == REPLAY_START+9)
    loss = agent.train_step()
    assert loss is not None, "train_step should return loss after REPLAY_START"
    assert loss >= 0.0, f"Loss should be non-negative, got {loss}"
    print(f"DQNAgent test PASSED. First loss: {loss:.6f}")

    ddqn = DDQNAgent(device='cpu')
    for i in range(REPLAY_START + 10):
        ddqn.store(dummy_state, i % 3, float(i % 5) * 0.2, dummy_state, False)
    loss2 = ddqn.train_step()
    assert loss2 is not None
    print(f"DDQNAgent test PASSED. First loss: {loss2:.6f}")

    agent2 = DQNAgent()
    agent2.steps = 400
    action = agent2.act(dummy_state)
    expected_eps = max(EPS_MIN, EPS_MAX - 400 * DECAY_RATE)
    assert abs(agent2.epsilon - expected_eps) < 1e-6
    print(f"Epsilon decay test PASSED. At step 400: epsilon={agent2.epsilon:.4f}")