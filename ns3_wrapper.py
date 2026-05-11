import gymnasium as gym
import numpy as np
from ns3gym import ns3env


class NS3RoutingEnv(gym.Env):
    def __init__(self, port=5555):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
        self.env = ns3env.Ns3Env(port=port, stepTime=5.0)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.array(obs, dtype=np.float32), reward, done, info

    def close(self):
        self.env.close()
