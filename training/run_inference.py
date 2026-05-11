import argparse
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_agent import DQNAgent, DDQNAgent
from env.ns3_wrapper import NS3RoutingEnv
from env.metrics import compute_reward
from configs.hyperparams import (
    STEPS_PER_EP, K_PATHS, BETA, RANDOM_SEED, CKPT_DIR
)

def run_inference():
    parser = argparse.ArgumentParser(description="Run trained DRL agent in inference mode")
    parser.add_argument("--algo",     type=str, default="dqn",
                        choices=["dqn","ddqn"])
    parser.add_argument("--ckpt",     type=str, default=None,
                        help="Checkpoint filename in results/checkpoints/")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    # Load agent
    if args.algo == "ddqn":
        agent = DDQNAgent(device='cpu')
    else:
        agent = DQNAgent(device='cpu')

    ckpt_name = args.ckpt or f"{args.algo}_final.pt"
    agent.load(ckpt_name)
    agent.epsilon = 0.0   # No exploration during inference

    print("="*60)
    print(f"{args.algo.upper()} INFERENCE MODE")
    print(f"Checkpoint: {ckpt_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Epsilon: {agent.epsilon} (pure greedy)")
    print("="*60)

    # Run evaluation episodes
    env = NS3RoutingEnv()
    all_costs = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_cost = 0.0
        ep_actions = []
        
        for step in range(STEPS_PER_EP):
            state  = obs
            action = agent.act(state)       # epsilon=0.0, so always greedy
            ep_actions.append(action)
            next_obs, _, done, _, _ = env.step(action)
            reward = compute_reward(obs, action, K_PATHS, BETA)
            ep_cost += reward
            obs = next_obs
            if done:
                break
        
        all_costs.append(ep_cost)
        action_dist = [ep_actions.count(a)/max(len(ep_actions),1)
                      for a in range(K_PATHS)]
        print(f"Inference Ep {ep+1}: cost={ep_cost:.4f} | actions={action_dist}")

    env.close()
    print(f"\nMean cost over {args.episodes} episodes: {np.mean(all_costs):.4f}")
    print(f"Std cost: {np.std(all_costs):.4f}")

if __name__ == "__main__":
    run_inference()
