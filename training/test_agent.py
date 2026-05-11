import sys
sys.path.insert(0, "/home/sabeeh138/projects/drl_project")
from configs.hyperparams import *
from agent.dqn_agent import DQNAgent
from env.metrics import compute_reward
import numpy as np

print("All imports OK!")
agent = DQNAgent()
print("Agent created OK!")
obs = np.concatenate([[0], np.random.rand(9)])
print(f"Observation shape: {obs.shape}")
reward = compute_reward(obs, 0)
print(f"Reward: {reward}")

print("Testing agent act...")
action = agent.act(0)
print(f"Action: {action}")
print("Testing agent store...")
agent.store(0, action, reward, 0, False)
print("Testing agent train step...")
loss = agent.train_step()
print(f"Loss: {loss}")

print("All tests passed!")
