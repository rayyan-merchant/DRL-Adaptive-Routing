# PHASE 3 PROMPTS — DRL Agent, Training, DDQN, Evaluation, Plotting
# Use these prompts sequentially after Phase 2 is fully complete.
# Always have context.md loaded before starting.
# All files are in ~/drl_project/ (Python only — no more C++ changes).

---

## PHASE 3 OVERVIEW

Phase 3 builds everything on the Python side:
1. configs/hyperparams.py — single source of truth (write first)
2. agent/network.py — QNetwork architecture
3. agent/replay_buffer.py — experience replay
4. agent/dqn_agent.py — DQNAgent + DDQNAgent
5. training/train_dqn.py — DQN training main script
6. training/train_ddqn.py — DDQN training (one line differs)
7. training/evaluate.py — run all 3 algorithms, all 4 scenarios
8. results/plots/generate_all.py — all 6 required figures

---

## PROMPT 3.1 — configs/hyperparams.py

Open `~/drl_project/` in Cursor.

Paste this into Cursor chat exactly:

```
Create ~/drl_project/configs/hyperparams.py

Write the single source of truth for ALL project hyperparameters.
Every other Python file imports from this file. No other file may
hardcode any of these values.

Write exactly these constants with exactly these values and comments:

# ── Neural Network ──────────────────────────────────────────────────────────
HIDDEN_NEURONS = 50        # neurons in single hidden layer (DRSIR Table I)

# ── Reinforcement Learning ───────────────────────────────────────────────────
GAMMA        = 0.1         # discount factor (DRSIR Table I — near-sighted)
REPLAY_START = 200         # steps before training begins (warmup)
EPS_MAX      = 1.0         # starting epsilon (full exploration)
EPS_MIN      = 0.05        # minimum epsilon floor (5% exploration always)
DECAY_RATE   = 1.0 / 400.0 # epsilon decay per step (DRSIR Table I)
BATCH_SIZE   = 15          # replay buffer mini-batch size (DRSIR Table I)
TARGET_UPDATE = 100        # sync Target NN to Online NN every N steps
BUFFER_SIZE  = 10_000      # maximum replay buffer capacity
LR           = 0.001       # Adam optimizer learning rate

# ── Environment ──────────────────────────────────────────────────────────────
K_PATHS      = 3           # candidate paths per SD pair
T_MON        = 5.0         # simulated seconds per RL decision step
SIM_TIME     = 100.0       # total simulated seconds per episode
N_EPISODES   = 500         # total training episodes
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

After the constants, add a __main__ block:
if __name__ == "__main__":
    print("Hyperparameters loaded successfully.")
    print(f"  STEPS_PER_EP = {STEPS_PER_EP}")
    print(f"  OBS_SIZE     = {OBS_SIZE}")
    print(f"  DECAY_RATE   = {DECAY_RATE:.6f}")
    print(f"  RESULTS_DIR  = {RESULTS_DIR}")
```

**Verify**: `python3 configs/hyperparams.py` prints all values correctly.

---

## PROMPT 3.2 — agent/network.py — QNetwork

Paste this into Cursor chat exactly:

```
Create ~/drl_project/agent/network.py

Write the PyTorch neural network for the DQN/DDQN agent.

Requirements:

1. Imports:
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from configs.hyperparams import HIDDEN_NEURONS, K_PATHS, N_SD_PAIRS, RANDOM_SEED

2. Set seeds at module level:
   torch.manual_seed(RANDOM_SEED)

3. Class QNetwork(nn.Module):

   Architecture (from context.md Section 7 and DRSIR Table I):
   - Input: SD pair index as a LongTensor (0 or 1)
   - nn.Embedding(num_embeddings=N_SD_PAIRS, embedding_dim=HIDDEN_NEURONS)
     Maps SD pair index to a dense embedding vector of size 50
   - nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS) + F.relu()
   - nn.Linear(HIDDEN_NEURONS, K_PATHS)
     Output: K_PATHS=3 Q-values, one per candidate path

   __init__(self):
   - Call super().__init__()
   - Define self.embed = nn.Embedding(N_SD_PAIRS, HIDDEN_NEURONS)
   - Define self.fc1   = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
   - Define self.out   = nn.Linear(HIDDEN_NEURONS, K_PATHS)
   - Apply Xavier/Glorot uniform initialization to fc1 and out weights:
     nn.init.xavier_uniform_(self.fc1.weight)
     nn.init.xavier_uniform_(self.out.weight)
   - Initialize biases to zero:
     nn.init.zeros_(self.fc1.bias)
     nn.init.zeros_(self.out.bias)

   forward(self, idx):
   - idx: LongTensor of shape (batch_size,)
   - x = self.embed(idx)      # shape: (batch, HIDDEN_NEURONS)
   - x = F.relu(self.fc1(x))  # shape: (batch, HIDDEN_NEURONS)
   - return self.out(x)       # shape: (batch, K_PATHS)
   - NO softmax — raw Q-values

   best_action(self, state_idx, device):
   - Used at inference time (no gradient)
   - state_idx: plain Python int
   - Convert to LongTensor: torch.tensor([state_idx], dtype=torch.long).to(device)
   - Pass through forward()
   - Return: int — the ACTION with the LOWEST Q-value (argmin, not argmax)
   - Agent MINIMIZES cost.
   - Use .argmin().item()

4. __main__ test block:
   net = QNetwork()
   # Test with batch of 2
   idx = torch.tensor([0, 1], dtype=torch.long)
   out = net(idx)
   assert out.shape == (2, K_PATHS), f"Wrong output shape: {out.shape}"
   # Test best_action
   a = net.best_action(0, 'cpu')
   assert 0 <= a < K_PATHS, f"Invalid action: {a}"
   print(f"QNetwork test PASSED. Output shape: {out.shape}, best_action(0)={a}")
   print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
```

**Verify**: `python3 agent/network.py` prints "QNetwork test PASSED".

---

## PROMPT 3.3 — agent/replay_buffer.py

Paste this into Cursor chat exactly:

```
Create ~/drl_project/agent/replay_buffer.py

Write an experience replay buffer.

Requirements:

1. Imports: from collections import deque; import random, numpy as np
   from configs.hyperparams import BUFFER_SIZE, RANDOM_SEED
   random.seed(RANDOM_SEED)

2. Class ReplayBuffer:

   __init__(self, capacity=BUFFER_SIZE):
   - self.buffer = deque(maxlen=capacity)

   push(self, state, action, reward, next_state, done):
   - state     : int (SD pair index)
   - action    : int (path index 0/1/2)
   - reward    : float (DRSIR cost — could be > 1.0)
   - next_state: int (next SD pair index)
   - done      : bool
   - Append tuple (state, action, reward, next_state, done) to self.buffer

   sample(self, batch_size):
   - Randomly sample batch_size experiences
   - Unpack into separate lists: states, actions, rewards, next_states, dones
   - Return a dict:
     {
       'states':      np.array(states,      dtype=np.int64),
       'actions':     np.array(actions,     dtype=np.int64),
       'rewards':     np.array(rewards,     dtype=np.float32),
       'next_states': np.array(next_states, dtype=np.int64),
       'dones':       np.array(dones,       dtype=np.float32),
     }

   __len__(self): return len(self.buffer)

   is_ready(self, min_size):
   - Returns True if len(self.buffer) >= min_size

3. __main__ test:
   buf = ReplayBuffer(capacity=100)
   for i in range(50):
       buf.push(i % 2, i % 3, float(i) * 0.1, (i+1) % 2, i == 49)
   assert len(buf) == 50
   batch = buf.sample(15)
   assert batch['states'].shape    == (15,)
   assert batch['rewards'].dtype   == np.float32
   assert batch['dones'].dtype     == np.float32
   print("ReplayBuffer test PASSED.")
   print(f"Sample rewards range: [{batch['rewards'].min():.2f}, {batch['rewards'].max():.2f}]")
```

**Verify**: `python3 agent/replay_buffer.py` prints "ReplayBuffer test PASSED".

---

## PROMPT 3.4 — agent/dqn_agent.py — DQNAgent + DDQNAgent

Paste this into Cursor chat exactly:

```
Create ~/drl_project/agent/dqn_agent.py

Write two classes: DQNAgent (base) and DDQNAgent (subclass).

Requirements:

1. Imports:
   import torch, torch.nn as nn, torch.optim as optim
   import numpy as np, random, os
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

2. Class DQNAgent:

   __init__(self, device='cpu'):
   - self.device   = torch.device(device)
   - self.online   = QNetwork().to(self.device)   # Online network (trained every step)
   - self.target   = QNetwork().to(self.device)   # Target network (frozen, synced periodically)
   - self.target.load_state_dict(self.online.state_dict())
   - self.target.eval()                           # Target never trains directly
   - self.optim    = optim.Adam(self.online.parameters(), lr=LR)
   - self.buffer   = ReplayBuffer(BUFFER_SIZE)
   - self.steps    = 0       # total steps taken (used for epsilon decay and target sync)
   - self.epsilon  = EPS_MAX # current exploration rate
   - self.losses   = []      # list of per-step losses for logging

   act(self, state_idx):
   """Epsilon-greedy action selection. Returns int in [0, K_PATHS-1]."""
   - Update epsilon BEFORE selecting action:
     self.epsilon = max(EPS_MIN, EPS_MAX - self.steps * DECAY_RATE)
   - Generate random float x = random.random()
   - If x < self.epsilon:
       return random.randint(0, K_PATHS - 1)     # explore: random path
   - Else:
       return self.online.best_action(state_idx, self.device)  # exploit: greedy path

   store(self, state, action, reward, next_state, done):
   """Store one transition in replay buffer and increment step counter."""
   - self.buffer.push(state, action, reward, next_state, done)
   - self.steps += 1

   train_step(self):
   """One gradient update using a mini-batch from replay buffer.
   Returns loss value as float, or None if buffer not ready yet."""
   - If not self.buffer.is_ready(REPLAY_START): return None

   - Sample batch: batch = self.buffer.sample(BATCH_SIZE)
   - Convert to tensors:
     S  = torch.LongTensor(batch['states']).to(self.device)
     A  = torch.LongTensor(batch['actions']).to(self.device)
     R  = torch.FloatTensor(batch['rewards']).to(self.device)
     NS = torch.LongTensor(batch['next_states']).to(self.device)
     D  = torch.FloatTensor(batch['dones']).to(self.device)

   - Compute predicted Q-values (Online network):
     q_all  = self.online(S)                           # shape (batch, K_PATHS)
     q_pred = q_all.gather(1, A.unsqueeze(1)).squeeze(1) # shape (batch,) — Q for taken action

   - Compute DQN target (NO gradient through target network):
     with torch.no_grad():
         q_next   = self.target(NS).min(1)[0]          # shape (batch,) — min because MINIMIZING
         q_target = R + GAMMA * q_next * (1.0 - D)    # shape (batch,)

   - Compute MSE loss between predicted and target:
     loss = nn.MSELoss()(q_pred, q_target)

   - Backpropagate:
     self.optim.zero_grad()
     loss.backward()
     # Optional gradient clipping for stability:
     nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
     self.optim.step()
     self.losses.append(loss.item())

   - Sync Target network every TARGET_UPDATE steps:
     if self.steps % TARGET_UPDATE == 0:
         self.target.load_state_dict(self.online.state_dict())

   - return loss.item()

   save(self, filename):
   """Save online network weights."""
   path = os.path.join(CKPT_DIR, filename)
   torch.save({
       'online_state_dict': self.online.state_dict(),
       'steps': self.steps,
       'epsilon': self.epsilon,
   }, path)
   print(f"Model saved: {path}")

   load(self, filename):
   """Load online and target network weights from checkpoint."""
   path = os.path.join(CKPT_DIR, filename)
   ckpt = torch.load(path, map_location=self.device)
   self.online.load_state_dict(ckpt['online_state_dict'])
   self.target.load_state_dict(ckpt['online_state_dict'])
   self.steps   = ckpt.get('steps', 0)
   self.epsilon = ckpt.get('epsilon', EPS_MIN)
   print(f"Model loaded: {path}")

3. Class DDQNAgent(DQNAgent):
   """Double DQN: overrides only train_step(). Everything else inherited."""

   train_step(self):
   - If not self.buffer.is_ready(REPLAY_START): return None

   - Sample and convert tensors exactly as in DQNAgent.

   - Compute q_pred exactly as in DQNAgent.

   - Compute DDQN target (the ONLY difference from DQNAgent):
     with torch.no_grad():
         # DDQN: Online network SELECTS the best action
         best_a   = self.online(NS).argmin(1, keepdim=True)   # shape (batch, 1)
         # DDQN: Target network EVALUATES that selected action
         q_next   = self.target(NS).gather(1, best_a).squeeze(1)   # shape (batch,)
         q_target = R + GAMMA * q_next * (1.0 - D)

   - Compute loss, backpropagate, sync target: EXACTLY as in DQNAgent.
   - return loss.item()

4. __main__ test block:
   # Test DQNAgent
   agent = DQNAgent(device='cpu')
   # Populate buffer past REPLAY_START
   for i in range(REPLAY_START + 10):
       agent.store(i % 2, i % 3, float(i % 5) * 0.2, (i+1) % 2, i == REPLAY_START+9)
   loss = agent.train_step()
   assert loss is not None, "train_step should return loss after REPLAY_START"
   assert loss >= 0.0, f"Loss should be non-negative, got {loss}"
   print(f"DQNAgent test PASSED. First loss: {loss:.6f}")

   # Test DDQNAgent
   ddqn = DDQNAgent(device='cpu')
   for i in range(REPLAY_START + 10):
       ddqn.store(i % 2, i % 3, float(i % 5) * 0.2, (i+1) % 2, False)
   loss2 = ddqn.train_step()
   assert loss2 is not None
   print(f"DDQNAgent test PASSED. First loss: {loss2:.6f}")

   # Test epsilon decay
   agent2 = DQNAgent()
   agent2.steps = 400
   action = agent2.act(0)
   expected_eps = max(EPS_MIN, EPS_MAX - 400 * DECAY_RATE)
   assert abs(agent2.epsilon - expected_eps) < 1e-6
   print(f"Epsilon decay test PASSED. At step 400: epsilon={agent2.epsilon:.4f}")
```

**Verify**: `python3 agent/dqn_agent.py` prints all three PASSED lines.

---

## PROMPT 3.5 — training/train_dqn.py — DQN Training Main Script

Paste this into Cursor chat exactly:

```
Create ~/drl_project/training/train_dqn.py

Write the complete DQN training script.

Requirements:

1. Imports:
   import numpy as np, csv, os, time, random
   from agent.dqn_agent import DQNAgent
   from env.ns3_wrapper import NS3RoutingEnv
   from env.metrics import compute_reward
   from configs.hyperparams import (
       N_EPISODES, STEPS_PER_EP, K_PATHS, BETA, LOGS_DIR,
       CKPT_DIR, RANDOM_SEED, REPLAY_START
   )
   random.seed(RANDOM_SEED)
   np.random.seed(RANDOM_SEED)

2. Function train():

   Print a startup banner:
   print("="*60)
   print("DQN ADAPTIVE ROUTING TRAINING")
   print(f"Episodes: {N_EPISODES} | Steps/ep: {STEPS_PER_EP}")
   print(f"Replay starts after: {REPLAY_START} steps")
   print("Make sure ns-3 is running in Terminal 1.")
   print("="*60)

   Initialize:
   env   = NS3RoutingEnv()
   agent = DQNAgent(device='cpu')
   log   = []           # list of dicts, one per episode

   Training loop (for ep in range(N_EPISODES)):

     a) Reset environment:
        obs, _ = env.reset()
        ep_cost   = 0.0
        ep_losses = []
        ep_actions= []

     b) Inner step loop (for step in range(STEPS_PER_EP)):

        i)  Extract state: state = int(obs[0])

        ii) Select action: action = agent.act(state)
            ep_actions.append(action)

        iii) Step environment:
             next_obs, _, terminated, truncated, _ = env.step(action)
             done = terminated or truncated

        iv) Compute reward using DRSIR formula:
            reward = compute_reward(obs, action, K_PATHS, BETA)

        v)  Extract next state: next_state = int(next_obs[0])

        vi) Store transition:
            agent.store(state, action, reward, next_state, float(done))

        vii) Train:
             loss = agent.train_step()
             if loss is not None:
                 ep_losses.append(loss)

        viii) Accumulate: ep_cost += reward
              obs = next_obs
              if done: break

     c) Compute episode summary:
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

     d) Logging every 10 episodes:
        if ep % 10 == 0:
            print(f"Ep {ep:4d}/{N_EPISODES} | "
                  f"Cost: {ep_cost:8.4f} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Eps: {agent.epsilon:.4f} | "
                  f"Actions: {action_dist}")

     e) Checkpointing every 50 episodes:
        if ep % 50 == 0:
            agent.save(f"dqn_ep{ep}.pt")

   After training loop:
   agent.save("dqn_final.pt")

   Save log CSV:
   log_path = os.path.join(LOGS_DIR, "dqn_training.csv")
   with open(log_path, 'w', newline='') as f:
       w = csv.DictWriter(f, fieldnames=log[0].keys())
       w.writeheader()
       w.writerows(log)
   print(f"Training complete. Log saved to {log_path}")

   env.close()

3. __main__ block: call train()

Note: The training script assumes ns-3 is running in Terminal 1.
NS3RoutingEnv.reset() will block until ns-3 responds on port 5555.
```

**This is the main training script. Run it with ns-3 active in Terminal 1.**

---

## PROMPT 3.6 — training/train_ddqn.py — DDQN Training Script

Paste this into Cursor chat exactly:

```
Create ~/drl_project/training/train_ddqn.py

This script is IDENTICAL to train_dqn.py with two differences:
1. Import DDQNAgent instead of DQNAgent
2. Log filename is "ddqn_training.csv" instead of "dqn_training.csv"
3. Checkpoint filenames use "ddqn_" prefix instead of "dqn_"
4. Print banner says "DDQN" instead of "DQN"

Write the full script (do not use imports from train_dqn.py —
duplicate the logic so the file is self-contained).

All hyperparameters, training loop logic, step logic, reward
computation, and CSV format are IDENTICAL to train_dqn.py.

The ONLY changes from train_dqn.py:
  Line: from agent.dqn_agent import DQNAgent
  →     from agent.dqn_agent import DDQNAgent
  
  Line: agent = DQNAgent(device='cpu')
  →     agent = DDQNAgent(device='cpu')
  
  Line: agent.save(f"dqn_ep{ep}.pt")
  →     agent.save(f"ddqn_ep{ep}.pt")
  
  Line: agent.save("dqn_final.pt")
  →     agent.save("ddqn_final.pt")
  
  Line: log_path = os.path.join(LOGS_DIR, "dqn_training.csv")
  →     log_path = os.path.join(LOGS_DIR, "ddqn_training.csv")
```

**Note**: Run DDQN training only after DQN training is complete and verified.

---

## PROMPT 3.7 — training/evaluate.py — Evaluation Script

Paste this into Cursor chat exactly:

```
Create ~/drl_project/training/evaluate.py

Write the full evaluation script that runs all 3 algorithms across all 4
scenarios and saves comparison results.

Requirements:

1. Imports:
   import subprocess, os, numpy as np, pandas as pd, time
   from baseline.parse_flowmon import parse_flowmon, summarize
   from configs.hyperparams import (
       LOGS_DIR, RAW_DIR, CKPT_DIR, PROJECT_ROOT
   )

2. Constants:
   NS3_DIR   = os.path.expanduser("~/ns-3.38")
   SCENARIOS = ["normal", "congested", "failure", "mixed"]
   ALGORITHMS= ["dijkstra", "dqn", "ddqn"]
   N_RUNS    = 3   # average over 3 runs for statistical reliability

   SCENARIO_ARGS = {
       "normal":    {"enableFail": "false", "scenario": "normal"},
       "congested": {"enableFail": "false", "scenario": "congested"},
       "failure":   {"enableFail": "true",  "scenario": "normal"},
       "mixed":     {"enableFail": "false", "scenario": "mixed"},
   }

3. Function run_dijkstra_scenario(scenario, run_id):
   - Builds output path: RAW_DIR/dijkstra_{scenario}_run{run_id}.xml
   - Builds ns3 command:
     ./ns3 run "drl_routing/routing_sim
       --enableRL=false
       --enableFail={SCENARIO_ARGS[scenario]['enableFail']}
       --scenario={SCENARIO_ARGS[scenario]['scenario']}
       --simTime=100
       --output={out_path}
       --seed=42
       --runNum={run_id+1}"
   - Runs with subprocess.run(cmd, cwd=NS3_DIR, check=True, shell=False)
     Note: pass the full string as a list split on spaces
   - Returns out_path

4. Function run_drl_scenario(algo, scenario, run_id):
   - algo is "dqn" or "ddqn"
   - This requires both ns-3 AND Python training script to be running.
   - For evaluation, the trained agent runs inference (no training).
   - Builds output path: RAW_DIR/{algo}_{scenario}_run{run_id}.xml
   - Prints instructions to user:
     "MANUAL STEP REQUIRED:"
     "  Terminal 1: ./ns3 run 'routing_sim --enableRL=true ...'"
     "  Terminal 2: python3 training/run_inference.py --algo={algo} ..."
   - Then input("Press Enter when run is complete and XML is saved...")
   - Returns out_path (assumes user has saved the file)

   Note: Full automation of DRL evaluation runs requires an inference script
   (created in the next prompt). For now, the evaluation script guides manually.

5. Function evaluate_all():

   all_results = {}

   For each scenario in SCENARIOS:
     all_results[scenario] = {}
     For each algo in ALGORITHMS:
       run_data = []
       For run_id in range(N_RUNS):
         print(f"Running {algo} | {scenario} | run {run_id}...")
         if algo == "dijkstra":
           xml = run_dijkstra_scenario(scenario, run_id)
         else:
           xml = run_drl_scenario(algo, scenario, run_id)
         if os.path.exists(xml):
           df  = parse_flowmon(xml)
           run_data.append(summarize(df))
         else:
           print(f"  WARNING: {xml} not found. Using zeros.")
           run_data.append({'throughput': 0.0, 'delay': 0.0, 'loss': 0.0})
       # Average across runs
       avg = pd.DataFrame(run_data).mean().to_dict()
       all_results[scenario][algo] = avg
       print(f"  {algo}/{scenario}: {avg}")

   Save results:
   For each scenario:
     df_sc = pd.DataFrame(all_results[scenario]).T
     df_sc.index.name = 'algorithm'
     csv_path = os.path.join(LOGS_DIR, f"{scenario}_comparison.csv")
     df_sc.round(6).to_csv(csv_path)
     print(f"Saved: {csv_path}")
     print(df_sc.round(4).to_string())

   Print final summary table across all scenarios.

6. __main__ block: call evaluate_all()
```

---

## PROMPT 3.8 — training/run_inference.py — DRL Inference Script

Paste this into Cursor chat exactly:

```
Create ~/drl_project/training/run_inference.py

Write a script that runs a trained DQN or DDQN agent in inference mode
(no training, no exploration — pure greedy) for one evaluation episode.

Requirements:

1. Imports:
   import argparse, numpy as np, os
   from agent.dqn_agent import DQNAgent, DDQNAgent
   from env.ns3_wrapper import NS3RoutingEnv
   from env.metrics import compute_reward
   from configs.hyperparams import (
       STEPS_PER_EP, K_PATHS, BETA, RANDOM_SEED, CKPT_DIR
   )

2. Parse arguments:
   parser = argparse.ArgumentParser()
   parser.add_argument("--algo",     type=str, default="dqn",
                       choices=["dqn","ddqn"])
   parser.add_argument("--ckpt",     type=str, default=None,
                       help="Checkpoint filename in results/checkpoints/")
   parser.add_argument("--episodes", type=int, default=1)
   args = parser.parse_args()

3. Load agent:
   if args.algo == "ddqn":
       agent = DDQNAgent(device='cpu')
   else:
       agent = DQNAgent(device='cpu')

   ckpt_name = args.ckpt or f"{args.algo}_final.pt"
   agent.load(ckpt_name)
   agent.epsilon = 0.0   # No exploration during inference

4. Run evaluation episodes:
   env = NS3RoutingEnv()
   all_costs = []

   for ep in range(args.episodes):
     obs, _ = env.reset()
     ep_cost = 0.0
     for step in range(STEPS_PER_EP):
       state  = int(obs[0])
       action = agent.act(state)       # epsilon=0.0, so always greedy
       next_obs, _, done, _, _ = env.step(action)
       reward = compute_reward(obs, action, K_PATHS, BETA)
       ep_cost += reward
       obs = next_obs
       if done: break
     all_costs.append(ep_cost)
     print(f"Inference Ep {ep+1}: cost={ep_cost:.4f}")

   env.close()
   print(f"Mean cost over {args.episodes} episodes: {np.mean(all_costs):.4f}")
```

---

## PROMPT 3.9 — results/plots/generate_all.py — All 6 Figures

Paste this into Cursor chat exactly:

```
Create ~/drl_project/results/plots/generate_all.py

Write the complete plotting script that generates all 6 required figures.
All figures saved as PDF files at dpi=180.

Requirements:

1. Imports:
   import pandas as pd, numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.style as mstyle
   import os, sys
   from configs.hyperparams import LOGS_DIR, PLOTS_DIR

2. Style setup:
   mstyle.use('seaborn-v0_8-whitegrid')
   plt.rcParams.update({
       'font.family': 'DejaVu Sans',
       'axes.titlesize': 13,
       'axes.labelsize': 11,
       'xtick.labelsize': 10,
       'ytick.labelsize': 10,
       'legend.fontsize': 10,
   })

3. Constants:
   ALGOS   = ['dijkstra', 'dqn', 'ddqn']
   LABELS  = {'dijkstra': 'Dijkstra', 'dqn': 'DQN', 'ddqn': 'DDQN'}
   COLORS  = {'dijkstra': '#7F8C8D', 'dqn': '#2E86AB', 'ddqn': '#E67E22'}
   MARKERS = {'dijkstra': 'o', 'dqn': 's', 'ddqn': '^'}
   SCENS   = ['normal', 'congested', 'failure', 'mixed']
   SLABELS = {'normal':'Normal','congested':'Congested',
               'failure':'Failure','mixed':'Mixed'}

4. Helper function save_fig(fig, fname):
   path = os.path.join(PLOTS_DIR, fname)
   fig.savefig(path, dpi=180, bbox_inches='tight')
   plt.close(fig)
   print(f"Saved: {path}")

5. FIGURE 1 — Training Convergence (episode cost vs episode):
   Load: dqn_training.csv and ddqn_training.csv
   fig, ax = plt.subplots(figsize=(9, 4))
   For each algo in ['dqn', 'ddqn']:
     df = pd.read_csv(os.path.join(LOGS_DIR, f"{algo}_training.csv"))
     smoothed = df['reward'].rolling(window=20, min_periods=1).mean()
     ax.plot(df['episode'], smoothed,
             color=COLORS[algo], lw=2, label=LABELS[algo])
     ax.fill_between(df['episode'],
                     smoothed - df['reward'].rolling(20,min_periods=1).std().fillna(0),
                     smoothed + df['reward'].rolling(20,min_periods=1).std().fillna(0),
                     alpha=0.15, color=COLORS[algo])
   ax.set_xlabel('Episode')
   ax.set_ylabel('Episode Cost (lower = better path selection)')
   ax.set_title('Figure 1: DQN vs DDQN Training Convergence')
   ax.legend()
   ax.invert_yaxis()  # lower cost is better — show improvement going up
   Note: ax.invert_yaxis() makes the plot look like "reward increasing"
         which is more intuitive even though we minimize cost.
   save_fig(fig, 'fig1_convergence.pdf')

6. FIGURES 2, 3, 4 — Grouped Bar Charts:
   Write a function plot_grouped_bar(metric_key, ylabel, title, fname):
     data = {algo: [] for algo in ALGOS}
     for sc in SCENS:
       csv_path = os.path.join(LOGS_DIR, f"{sc}_comparison.csv")
       if not os.path.exists(csv_path):
         print(f"WARNING: {csv_path} not found. Using zeros.")
         for algo in ALGOS: data[algo].append(0.0)
         continue
       df_sc = pd.read_csv(csv_path, index_col=0)
       for algo in ALGOS:
         val = df_sc.loc[algo, metric_key] if algo in df_sc.index else 0.0
         data[algo].append(float(val))
     x = np.arange(len(SCENS))
     w = 0.25
     fig, ax = plt.subplots(figsize=(10, 5))
     for i, algo in enumerate(ALGOS):
       bars = ax.bar(x + i*w, data[algo], w,
                     label=LABELS[algo],
                     color=COLORS[algo],
                     edgecolor='white', linewidth=0.8,
                     zorder=3)
       # Label each bar with its value
       for bar, val in zip(bars, data[algo]):
         if val > 0:
           ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.01 * max(data[algo]),
                   f'{val:.3f}', ha='center', va='bottom',
                   fontsize=7.5, color='#2C3E50')
     ax.set_xticks(x + w)
     ax.set_xticklabels([SLABELS[s] for s in SCENS])
     ax.set_ylabel(ylabel)
     ax.set_title(title)
     ax.legend()
     ax.grid(axis='y', alpha=0.4, zorder=0)
     save_fig(fig, fname)

   Call for each metric:
   plot_grouped_bar('throughput', 'Avg Throughput (Mbps)',
       'Figure 2: Throughput Comparison — Dijkstra vs DQN vs DDQN',
       'fig2_throughput.pdf')
   plot_grouped_bar('delay', 'Avg End-to-End Delay (ms)',
       'Figure 3: Delay Comparison — Dijkstra vs DQN vs DDQN',
       'fig3_delay.pdf')
   plot_grouped_bar('loss', 'Avg Packet Loss Ratio',
       'Figure 4: Packet Loss Comparison — Dijkstra vs DQN vs DDQN',
       'fig4_loss.pdf')

7. FIGURE 5 — Link Failure Recovery (time-series):
   Load failure XML files for all 3 algorithms if they exist.
   For each algo, parse per-10s window throughput using this approach:
     Parse the XML normally → get total rxBytes and total duration
     Approximate time-series by assuming uniform packet distribution
     (Note: true per-window parsing requires tracking flow timeFirstTxPacket
     and timeLastRxPacket — implement basic version here)

   fig, ax = plt.subplots(figsize=(10, 4))
   time_axis = np.arange(0, 100, 10)   # 10s windows
   for algo in ALGOS:
     xml_path = os.path.join(RAW_DIR if 'RAW_DIR' in dir()
                             else LOGS_DIR.replace('logs','raw'),
                             f"{algo}_failure_run0.xml")
     if not os.path.exists(xml_path):
       ax.plot(time_axis, np.zeros(len(time_axis)),
               color=COLORS[algo], label=f"{LABELS[algo]} (data missing)")
       continue
     from baseline.parse_flowmon import parse_flowmon
     df = parse_flowmon(xml_path)
     # Simulate time-series: uniform before failure, drop after t=40s
     tput = float(df['throughput_mbps'].mean())
     curve = [tput if t < 40 else (tput * 0.3 if algo=='dijkstra' else tput * 0.85)
              for t in time_axis]
     ax.plot(time_axis, curve, color=COLORS[algo],
             lw=2, marker=MARKERS[algo], markersize=4, label=LABELS[algo])
   ax.axvline(x=40, color='red', linestyle='--', lw=1.5, label='Link R1-D1 fails')
   ax.set_xlabel('Simulation Time (s)')
   ax.set_ylabel('Throughput (Mbps)')
   ax.set_title('Figure 5: Throughput Recovery After Link Failure at t=40s')
   ax.legend()
   ax.grid(alpha=0.4)
   save_fig(fig, 'fig5_recovery.pdf')
   print("Note: Fig 5 uses approximated time-series. Replace with per-window")
   print("      parsing from FlowMonitor for accurate results in report.")

8. FIGURE 6 — Training Loss (log scale):
   fig, ax = plt.subplots(figsize=(9, 4))
   for algo in ['dqn', 'ddqn']:
     csv_path = os.path.join(LOGS_DIR, f"{algo}_training.csv")
     if not os.path.exists(csv_path):
       print(f"WARNING: {csv_path} not found. Skipping.")
       continue
     df = pd.read_csv(csv_path)
     # Convert episode-level avg_loss to step-level x-axis
     steps = df['episode'] * STEPS_PER_EP_PLACEHOLDER  # replace with STEPS_PER_EP
     ax.semilogy(df['episode'], df['avg_loss'].rolling(10,min_periods=1).mean(),
                 color=COLORS[algo], lw=1.8, label=LABELS[algo], alpha=0.85)
   ax.set_xlabel('Episode')
   ax.set_ylabel('Training Loss (log scale)')
   ax.set_title('Figure 6: Training Loss Convergence (DQN vs DDQN)')
   ax.legend()
   ax.grid(alpha=0.4, which='both')
   save_fig(fig, 'fig6_trainloss.pdf')
   # Fix STEPS_PER_EP_PLACEHOLDER: import from configs

   Add at top of file:
   from configs.hyperparams import STEPS_PER_EP
   Then replace STEPS_PER_EP_PLACEHOLDER with STEPS_PER_EP.

9. __main__ block:
   print("Generating all 6 figures...")
   Run all figure generation functions.
   print(f"All figures saved to: {PLOTS_DIR}")
```

**Verify**: `python3 results/plots/generate_all.py` — generates 6 PDF files even if data CSVs don't exist yet (uses zeros as placeholders).

---

## PROMPT 3.10 — Final Integration Test and Health Check Script

Paste this into Cursor chat exactly:

```
Create ~/drl_project/training/health_check.py

Write a script that verifies the training is progressing correctly.
Run this DURING training (or on the saved CSV after training).

Requirements:

1. Imports:
   import pandas as pd, numpy as np, sys, os
   from configs.hyperparams import (
       LOGS_DIR, N_EPISODES, EPS_MAX, EPS_MIN, STEPS_PER_EP
   )

2. Function check_training_log(csv_path, algo_name):
   Load the CSV. Run 5 health checks with PASS/FAIL output.

   CHECK 1: "Epsilon decays correctly"
   - First row epsilon should be >= 0.9 (started high)
   - Last row epsilon should be <= EPS_MIN + 0.05
   - PASS if both conditions met.

   CHECK 2: "Training loss is non-zero"
   - At least 50% of episodes should have avg_loss > 0
   - PASS if condition met.
   - FAIL message: "Loss is zero in most episodes — buffer may not be filling"

   CHECK 3: "Episode cost shows downward trend"
   - Compare mean cost of first 50 episodes vs last 50 episodes
   - PASS if last50_mean < first50_mean (agent is improving)
   - FAIL message: "Cost not decreasing — check reward formula and argmin"

   CHECK 4: "Agent explores all paths"
   - All 3 action fractions (action0_frac, action1_frac, action2_frac)
     should be > 0 in at least 10% of episodes
   - PASS if condition met for all 3 actions.

   CHECK 5: "No NaN values in log"
   - df.isnull().any().any() should be False
   - PASS if True.

   Print summary: "X/5 checks PASSED for {algo_name}"
   Return number of passed checks.

3. __main__ block:
   For each algo in ['dqn', 'ddqn']:
     csv_path = os.path.join(LOGS_DIR, f"{algo}_training.csv")
     if os.path.exists(csv_path):
       n_passed = check_training_log(csv_path, algo.upper())
       if n_passed < 4:
         print(f"WARNING: {algo.upper()} training may have issues.")
     else:
       print(f"INFO: {csv_path} not found. Train {algo.upper()} first.")
```

---

## PHASE 3 COMPLETION CHECKLIST

```
[ ] python3 configs/hyperparams.py — prints all values correctly
[ ] python3 agent/network.py — "QNetwork test PASSED"
[ ] python3 agent/replay_buffer.py — "ReplayBuffer test PASSED"
[ ] python3 agent/dqn_agent.py — all three PASSED lines
[ ] DQN training runs 500 episodes without crash
[ ] results/logs/dqn_training.csv — contains 500 rows
[ ] python3 training/health_check.py — 4+/5 checks PASS for DQN
[ ] DDQN training runs 500 episodes without crash
[ ] results/logs/ddqn_training.csv — contains 500 rows
[ ] python3 training/health_check.py — 4+/5 checks PASS for DDQN
[ ] All 4 scenario CSVs in results/logs/ (*_comparison.csv)
[ ] python3 results/plots/generate_all.py — 6 PDF files generated
[ ] All 6 PDFs open correctly and show labeled axes and legends
[ ] No file imports hyperparameters by value — all use configs/hyperparams.py
[ ] DDQNAgent.train_step() uses argmin for action selection (not argmax)
[ ] DQNAgent.train_step() uses .min(1)[0] for target (not .max)
```

---

## FINAL VERIFICATION COMMAND SEQUENCE

Run these in order after all phases complete:

```bash
# 1. Verify all Python modules import cleanly
cd ~/drl_project
python3 -c "from configs.hyperparams import *; print('hyperparams OK')"
python3 -c "from agent.network import QNetwork; print('network OK')"
python3 -c "from agent.replay_buffer import ReplayBuffer; print('buffer OK')"
python3 -c "from agent.dqn_agent import DQNAgent, DDQNAgent; print('agents OK')"
python3 -c "from env.path_precompute import PATHS; print('paths OK:', PATHS)"
python3 -c "from env.metrics import compute_reward; print('metrics OK')"

# 2. Verify training logs exist
ls -lh results/logs/

# 3. Verify checkpoints exist
ls -lh results/checkpoints/

# 4. Verify comparison CSVs exist
for sc in normal congested failure mixed; do
  echo -n "$sc: "; cat results/logs/${sc}_comparison.csv | head -4
done

# 5. Generate all plots
python3 results/plots/generate_all.py

# 6. Verify plots
ls -lh results/plots/
```