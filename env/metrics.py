import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.hyperparams import K_PATHS, BETA

def extract_path_metrics(obs, k=K_PATHS):
    return obs[1:].reshape(k, 3)

def minmax_normalize(path_metrics):
    normalized = np.zeros_like(path_metrics)
    for c in range(3):
        col = path_metrics[:, c]
        vmin, vmax = col.min(), col.max()
        if (vmax - vmin) < 1e-8:
            normalized[:, c] = 0.5
        else:
            normalized[:, c] = (col - vmin) / (vmax - vmin)
    return normalized

def compute_reward(obs, action, k=K_PATHS, beta=BETA):

    """
    Lower cost is better.
    Stable normalized routing cost.
    """

    path_metrics = extract_path_metrics(obs, k)

    norm = minmax_normalize(path_metrics)

    bw_n    = norm[action, 0]
    delay_n = norm[action, 1]
    loss_n  = norm[action, 2]

    # ------------------------------------------
    # SAFE COST FUNCTION
    # ------------------------------------------

    # Higher bandwidth should REDUCE cost
    bw_cost = 1.0 - bw_n

    reward = (
        beta[0] * bw_cost +
        beta[1] * delay_n +
        beta[2] * loss_n
    )

    return float(reward)



def batch_rewards(obs_batch, actions_batch, k=K_PATHS, beta=BETA):
    rewards = np.zeros(len(obs_batch), dtype=np.float32)
    for i in range(len(obs_batch)):
        rewards[i] = compute_reward(obs_batch[i], actions_batch[i], k, beta)
    return rewards

if __name__ == "__main__":
    obs_test = np.array([0.0,
                         1.0, 0.1, 0.0,
                         0.3, 0.5, 0.2,
                         0.1, 0.9, 0.8])
    r0 = compute_reward(obs_test, action=0)
    r2 = compute_reward(obs_test, action=2)
    assert r0 < r2, f"Expected r0 < r2, got r0={r0:.4f} r2={r2:.4f}"
    print(f"Sanity check PASSED: r0={r0:.4f} < r2={r2:.4f}")