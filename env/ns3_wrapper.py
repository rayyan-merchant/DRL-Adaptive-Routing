import numpy as np
import gym
from gym import spaces
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ns3gym import ns3env
except ImportError:
    raise ImportError("ns3gym not installed. Run: pip install ns3gym")

from configs.hyperparams import (
    K_PATHS,
    OBS_SIZE,
    ZMQ_PORT,
    T_MON
)


class NS3RoutingEnv(gym.Env):

    def __init__(self, port=ZMQ_PORT, sim_seed=42, debug=False):

        super(NS3RoutingEnv, self).__init__()

        self.port = port
        self.sim_seed = sim_seed
        self.debug = debug

        self.env = None

        self._obs = np.zeros(
            OBS_SIZE,
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(K_PATHS)

    def reset(self, seed=None, options=None):

        # --------------------------------------------------
        # CLEANUP OLD CONNECTION
        # --------------------------------------------------

        if self.env is not None:

            try:
                self.env.close()
            except Exception:
                pass

            self.env = None

            # IMPORTANT:
            # Allow old ZMQ sockets to close properly
            time.sleep(3)

        # --------------------------------------------------
        # CREATE NEW NS3 ENV
        # --------------------------------------------------

        self.env = ns3env.Ns3Env(
            port=self.port,
            stepTime=T_MON,
            startSim=False,
            simSeed=self.sim_seed,
            simArgs={},
            debug=self.debug
        )

        # IMPORTANT:
        # Give simulator time to relaunch and bind socket
        time.sleep(2)

        # --------------------------------------------------
        # RESET ENVIRONMENT
        # --------------------------------------------------

        obs = self.env.reset()

        if obs is None:
            raise RuntimeError(
                "ns-3 environment reset returned None"
            )

        self._obs = np.array(
            obs,
            dtype=np.float32
        )

        return self._obs, {}

    def step(self, action):

        assert 0 <= action < K_PATHS

        obs, reward, done, info = self.env.step(
            int(action)
        )

        if obs is not None:

            self._obs = np.array(
                obs,
                dtype=np.float32
            )

        # IMPORTANT:
        # Return REAL reward from ns-3
        return (
            self._obs,
            float(reward),
            bool(done),
            False,
            info
        )

    def close(self):

        if self.env is not None:

            try:
                self.env.close()
            except Exception:
                pass

            self.env = None

            time.sleep(1)

    @property
    def current_obs(self):
        return self._obs


# ==========================================================
# SMOKE TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NS3 ROUTING ENV SMOKE TEST")
    print("=" * 60)

    print("\nIMPORTANT:")
    print("Run simulator loop in another terminal:\n")

    print(
        "cd ~/projects/ns-3-dev && "
        "while true; do "
        "./ns3 run "
        "\"routing_sim --enableRL=true --simTime=100\"; "
        "sleep 2; "
        "done\n"
    )

    import random

    env = NS3RoutingEnv(
        port=ZMQ_PORT
    )

    obs, _ = env.reset()

    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")

    total_reward = 0.0

    for step in range(20):

        action = random.randint(
            0,
            K_PATHS - 1
        )

        obs, reward, done, _, info = env.step(
            action
        )

        total_reward += reward

        print(
            f"Step {step:02d} | "
            f"action={action} | "
            f"reward={reward:.4f} | "
            f"done={done}"
        )

        if done:
            break

    env.close()

    print("-" * 60)
    print(f"Total reward: {total_reward:.4f}")
    print("Smoke test complete.")