import numpy as np


def compute_reward(obs, action, k=3, beta=(1.0, 1.0, 1.0)):
    path_metrics = obs[1:].reshape(k, 3).astype(np.float32)
    norm = np.zeros_like(path_metrics)
    for col in range(3):
        vals = path_metrics[:, col]
        vmin, vmax = vals.min(), vals.max()
        if (vmax - vmin) < 1e-8:
            norm[:, col] = 0.5
        else:
            norm[:, col] = (vals - vmin) / (vmax - vmin)
    bw_n, dl_n, ls_n = norm[action]
    reward = beta[0] * (1.0 / (bw_n + 1e-6)) + beta[1] * dl_n + beta[2] * ls_n
    return float(reward)
