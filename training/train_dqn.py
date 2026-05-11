import numpy as np
import csv
import os
import time
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_agent import DQNAgent
from env.ns3_wrapper import NS3RoutingEnv
from env.metrics import compute_reward
from configs.hyperparams import (
    N_EPISODES, STEPS_PER_EP, K_PATHS, BETA, LOGS_DIR,
    CKPT_DIR, RANDOM_SEED, REPLAY_START
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def train():
    print("="*60)
    print("DQN ADAPTIVE ROUTING TRAINING")
    print(f"Episodes: {N_EPISODES} | Steps/ep: {STEPS_PER_EP}")
    print(f"Replay starts after: {REPLAY_START} steps")
    print("Make sure ns-3 is running in Terminal 1.")
    print("="*60)

    env   = NS3RoutingEnv()
    agent = DQNAgent(device='cpu')
    log   = []           # list of dicts, one per episode

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        ep_cost   = 0.0
        ep_losses = []
        ep_actions= []

        for step in range(STEPS_PER_EP):
            state = obs
            action = agent.act(state)
            ep_actions.append(action)

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = compute_reward(obs, action, K_PATHS, BETA)
            next_state = next_obs

            agent.store(state, action, reward, next_state, float(done))

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            ep_cost += reward
            obs = next_obs
            if done:
                break

        avg_loss    = float(np.mean(ep_losses)) if ep_losses else 0.0
        action_dist = [ep_actions.count(a) / max(len(ep_actions),1)
                       for a in range(K_PATHS)]

        log.append({
            'episode':  ep,
            'reward':   round(ep_cost, 6),
            'avg_loss': round(avg_loss, 8),
            'epsilon':  round(agent.epsilon, 6),
            'action0_frac': round(action_dist[0], 4),
            'action1_frac': round(action_dist[1], 4),
            'action2_frac': round(action_dist[2], 4),
        })

        if ep % 10 == 0:
            print(f"Ep {ep:4d}/{N_EPISODES} | "
                  f"Cost: {ep_cost:8.4f} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Eps: {agent.epsilon:.4f} | "
                  f"Actions: {action_dist}")

        if ep % 50 == 0:
            agent.save(f"dqn_ep{ep}.pt")

    agent.save("dqn_final.pt")

    log_path = os.path.join(LOGS_DIR, "dqn_training.csv")
    with open(log_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    print(f"Training complete. Log saved to {log_path}")

    env.close()

if __name__ == "__main__":
    train()