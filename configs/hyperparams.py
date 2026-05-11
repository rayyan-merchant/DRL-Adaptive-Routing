# ── Neural Network ──────────────────────────────────────────────────────────
HIDDEN_NEURONS = 50        # neurons in single hidden layer (DRSIR Table I)

# ── Reinforcement Learning ───────────────────────────────────────────────────
GAMMA        = 0.1         # discount factor (DRSIR Table I — near-sighted)
REPLAY_START = 200         # steps before training begins (warmup)
EPS_MAX      = 1.0         # starting epsilon (full exploration)
EPS_MIN      = 0.05        # minimum epsilon floor (5% exploration always)
DECAY_RATE   = 1.0 / 32000.0 # epsilon decay per step (80% of 2000 eps * 20 steps)
BATCH_SIZE   = 15          # replay buffer mini-batch size (DRSIR Table I)
TARGET_UPDATE = 100        # sync Target NN to Online NN every N steps
BUFFER_SIZE  = 10_000      # maximum replay buffer capacity
LR           = 0.001       # Adam optimizer learning rate

# ── Environment ──────────────────────────────────────────────────────────────
K_PATHS      = 3           # candidate paths per SD pair
T_MON        = 5.0         # simulated seconds per RL decision step
SIM_TIME     = 100.0       # total simulated seconds per episode
N_EPISODES   = 2000        # total training episodes
STEPS_PER_EP = int(SIM_TIME / T_MON)   # = 20 steps per episode
N_SD_PAIRS   = 2           # number of SD pairs (S1→D1 and S2→D1)
OBS_SIZE     = 1 + K_PATHS * 3         # = 10: [sd_idx, bw0,d0,l0, bw1,d1,l1, bw2,d2,l2]
ZMQ_PORT     = 5555        # port for ns3-gym ZeroMQ communication

# ── Reward ───────────────────────────────────────────────────────────────────
BETA         = (1.0, 1.0, 1.0)   # reward weights (beta1, beta2, beta3)
                                   # beta1: BW weight, beta2: delay, beta3: loss

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED  = 42          # used in Python: random.seed, np.random.seed, torch

# ── Paths ────────────────────────────────────────────────────────────────────
import os
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR       = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR      = os.path.join(RESULTS_DIR, "plots")
RAW_DIR        = os.path.join(RESULTS_DIR, "raw")
CKPT_DIR       = os.path.join(RESULTS_DIR, "checkpoints")

# ── Create directories on import ─────────────────────────────────────────────
for _d in [LOGS_DIR, PLOTS_DIR, RAW_DIR, CKPT_DIR]:
    os.makedirs(_d, exist_ok=True)

if __name__ == "__main__":
    print("Hyperparameters loaded successfully.")
    print(f"  STEPS_PER_EP = {STEPS_PER_EP}")
    print(f"  OBS_SIZE     = {OBS_SIZE}")
    print(f"  DECAY_RATE   = {DECAY_RATE:.6f}")
    print(f"  RESULTS_DIR  = {RESULTS_DIR}")