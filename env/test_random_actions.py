import random
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.ns3_wrapper import NS3RoutingEnv
from env.metrics import compute_reward
from configs.hyperparams import K_PATHS, N_EPISODES

N_TEST_EPISODES = 5
STEPS_PER_EP = 20

print("="*60)
print("INTEGRATION TEST: Random Action Loop")
print("Make sure ns-3 is running in Terminal 1:")
print("  cd ~/ns-3.38")
print("  ./ns3 run 'drl_routing/routing_sim --enableRL=true --simTime=100'")
print("="*60)
input("Press Enter when ns-3 is running...")

env = NS3RoutingEnv()
reward_history = []
reward_per_action = {0: [], 1: [], 2: []}

try:
    for ep in range(N_TEST_EPISODES):
        obs, _ = env.reset()
        ep_reward = 0.0
        for step in range(STEPS_PER_EP):
            action = random.randint(0, K_PATHS-1)
            next_obs, _, done, _, _ = env.step(action)
            reward = compute_reward(obs, action)
            
            ep_reward += reward
            reward_per_action[action].append(reward)
            
            obs = next_obs
            if done:
                break
                
        reward_history.append(ep_reward)
        print(f"Episode {ep+1}/{N_TEST_EPISODES} | Total cost: {ep_reward:.4f}")
        
    print("\nMean reward per action:")
    for act, r_list in reward_per_action.items():
        mean_r = np.mean(r_list) if r_list else 0.0
        print(f"  Action {act}: {mean_r:.4f}")
        
    assert len(reward_history) == N_TEST_EPISODES, "Not all episodes completed"
    
    rewards_flat = [r for v in reward_per_action.values() for r in v]
    assert max(rewards_flat) != min(rewards_flat), "All rewards identical — environment is not responsive"
    print("GATE: Rewards vary across actions — environment is working correctly.")
    
    print("TEST PASSED")
except Exception as e:
    print(f"TEST FAILED: {e}")
finally:
    env.close()