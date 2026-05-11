# PROJECT CONTEXT — DRL-Based Adaptive Network Routing
# Read this file completely before writing any code.
# This is the single source of truth for the entire project.

---

## 0. What This Project Is

This is a semester-level Computer Networks project that implements a
**Deep Reinforcement Learning (DRL) based adaptive routing system**
inside the ns-3 network simulator. A single centralized DQN agent learns
to dynamically select optimal routing paths through a simulated 6-node
network. It is then upgraded to DDQN. Both are compared against static
Dijkstra routing across four network scenarios.

The design is directly inspired by:
> Casas-Velasco et al., "DRSIR: A Deep Reinforcement Learning Approach
> for Routing in Software-Defined Networking," IEEE TNSM 2021.

The key adaptation: DRSIR ran on Mininet with OpenFlow. This project
runs on ns-3 with Ipv4StaticRoutingHelper. Everything else follows
DRSIR's design as closely as possible.

---

## 1. Team & Repository Layout

```
Students : Muhammad Sabeeh (23K-0002) and Rayyan Merchant (23K-0073)
Course   : Computer Networks — FAST NUCES
```

### Directory Structure (must match exactly)

```
~/ns-3.38/scratch/drl_routing/
    routing_sim.cc          # Main ns-3 simulation file (C++)
    routing_env.h           # OpenGymEnv subclass header (C++)
    routing_env.cc          # OpenGymEnv subclass implementation (C++)

~/drl_project/
    configs/
        hyperparams.py      # ALL hyperparameters — import from here only
        topology.py         # Graph definition for NetworkX path computation
    agent/
        network.py          # QNetwork (PyTorch nn.Module)
        replay_buffer.py    # ReplayBuffer using collections.deque
        dqn_agent.py        # DQNAgent class + DDQNAgent subclass
    env/
        path_precompute.py  # NetworkX k-shortest paths, runs once at startup
        metrics.py          # DRSIR reward formula implementation
        ns3_wrapper.py      # Python-side Gym environment wrapper
    training/
        train_dqn.py        # DQN training main script
        train_ddqn.py       # DDQN training main script (imports DDQNAgent)
        evaluate.py         # Runs all 3 algorithms on all 4 scenarios
    baseline/
        parse_flowmon.py    # FlowMonitor XML parser → pandas DataFrame
        run_baseline.py     # Runs Dijkstra baseline and saves results
    results/
        logs/               # Training CSVs: dqn_training.csv, ddqn_training.csv
        plots/              # Output figures: fig1_convergence.pdf … fig6_loss.pdf
        raw/                # FlowMonitor XML files from ns-3
        checkpoints/        # Model .pt files: dqn_ep0.pt, dqn_final.pt, etc.
```

---

## 2. Technology Stack

| Layer | Technology | Version | Notes |
|---|---|---|---|
| Simulation | ns-3 | 3.38 | C++, compiled with ./ns3 build |
| RL Bridge | ns3-gym (opengym) | latest | ZeroMQ + protobuf, port 5555 |
| DRL Framework | PyTorch | 2.x | CPU only, no CUDA needed |
| Path Computation | NetworkX | 3.x | Runs once at startup, static |
| Data | pandas + numpy | latest | CSV logs, metric arrays |
| Plotting | matplotlib + seaborn | latest | Save as PDF at 180 dpi |
| Python | Python | 3.10 or 3.11 | Must be consistent across venv |

---

## 3. Network Topology (6 Nodes, 10 Links)

### Node Index Table
```
Index  Name  Role
  0     S1   Source 1  — primary traffic generator
  1     S2   Source 2  — secondary traffic generator
  2     R1   Interior router
  3     R2   Interior router (central hub, most connected)
  4     R3   Interior router
  5     D1   Destination — PacketSink receives all flows
```

### Link Table (C++ array order = index used in devs[])
```
devs[] index | u  | v  | Bandwidth | Delay  | Strategic role
      0       | 0  | 2  | 10 Mbps   | 2 ms   | S1-R1 fast entry
      1       | 0  | 3  | 7 Mbps    | 4 ms   | S1-R2 medium entry
      2       | 1  | 3  | 7 Mbps    | 4 ms   | S2-R2 medium entry
      3       | 1  | 4  | 10 Mbps   | 2 ms   | S2-R3 fast entry
      4       | 2  | 3  | 5 Mbps    | 5 ms   | R1-R2 interior bottleneck
      5       | 3  | 4  | 5 Mbps    | 5 ms   | R2-R3 interior bottleneck
      6       | 2  | 5  | 8 Mbps    | 8 ms   | R1-D1 primary exit
      7       | 3  | 5  | 6 Mbps    | 6 ms   | R2-D1 secondary exit
      8       | 4  | 5  | 4 Mbps    | 10 ms  | R3-D1 slow exit
      9       | 2  | 4  | 3 Mbps    | 12 ms  | R1-R3 low-cap cross-link
```

### IP Subnet Assignment (assigned in devs[] order)
```
devs[0]  → 10.0.1.0/24   (S1–R1)
devs[1]  → 10.1.1.0/24   (S1–R2)
devs[2]  → 10.2.1.0/24   (S2–R2)
devs[3]  → 10.3.1.0/24   (S2–R3)
devs[4]  → 10.4.1.0/24   (R1–R2)
devs[5]  → 10.5.1.0/24   (R2–R3)
devs[6]  → 10.6.1.0/24   (R1–D1)
devs[7]  → 10.7.1.0/24   (R2–D1)
devs[8]  → 10.8.1.0/24   (R3–D1)
devs[9]  → 10.9.1.0/24   (R1–R3)
```

**CRITICAL**: D1's IP address must NEVER be hardcoded. Always resolve it
dynamically after addr.Assign() using:
```cpp
Ipv4Address d1_ip = nodes.Get(5)->GetObject<Ipv4>()
                        ->GetAddress(1, 0).GetLocal();
```

---

## 4. Candidate Paths (Precomputed, Static)

NetworkX computes these once using `shortest_simple_paths` with `weight='delay'`.
They never change during training.

```
SD Pair (0,5)  =  S1 → D1
  Action 0  →  Path [0, 2, 5]      =  S1 - R1 - D1        (fastest, 2 hops)
  Action 1  →  Path [0, 3, 5]      =  S1 - R2 - D1        (medium, 2 hops)
  Action 2  →  Path [0, 2, 3, 5]   =  S1 - R1 - R2 - D1   (slowest, 3 hops)

SD Pair (1,5)  =  S2 → D1
  Action 0  →  Path [1, 4, 5]      =  S2 - R3 - D1        (direct, slow exit)
  Action 1  →  Path [1, 3, 5]      =  S2 - R2 - D1        (medium, better BW)
  Action 2  →  Path [1, 3, 2, 5]   =  S2 - R2 - R1 - D1   (3 hops, bottleneck)
```

**Dijkstra always selects Action 0 for both SD pairs** (lowest hop/delay
without congestion awareness). The DQN agent must learn to select Actions
1 or 2 under congestion.

---

## 5. MDP Formulation (The Core Design)

### State
- A single integer: the SD pair index being routed this step
- 0 = S1→D1,  1 = S2→D1
- This integer is the ONLY input to the neural network
- It is fed through an nn.Embedding layer (not one-hot)

### Action
- A single integer in [0, k-1] where k=3
- Selects one of the 3 precomputed candidate paths for the current SD pair

### Observation Vector (10 floats, sent from ns-3 to Python)
```
obs[0]    = float(sd_pair_index)     # 0.0 or 1.0
obs[1]    = path0_bw_raw             # available BW on path 0 (Mbps)
obs[2]    = path0_delay_raw          # total delay on path 0 (ms)
obs[3]    = path0_loss_raw           # packet loss ratio on path 0 [0,1]
obs[4]    = path1_bw_raw             # path 1 metrics
obs[5]    = path1_delay_raw
obs[6]    = path1_loss_raw
obs[7]    = path2_bw_raw             # path 2 metrics
obs[8]    = path2_delay_raw
obs[9]    = path2_loss_raw
```

**CRITICAL separation**:
- `obs[0]` → cast to int → fed to `nn.Embedding` → neural network input
- `obs[1:10]` → used ONLY for reward computation → NEVER fed to neural network

### Reward Function (DRSIR Equation 5 — Agent MINIMIZES this)
```
path_metrics = obs[1:].reshape(k=3, 3)   # shape (3, 3): [bw, delay, loss] per path

For each column c in {bw, delay, loss}:
    norm[:, c] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    # if max == min, set norm[:, c] = 0.5

bw_n    = norm[action, 0]
delay_n = norm[action, 1]
loss_n  = norm[action, 2]

R = beta1 * (1.0 / (bw_n + 1e-6)) + beta2 * delay_n + beta3 * loss_n

where beta1 = beta2 = beta3 = 1.0  (from configs/hyperparams.py BETA)
```

**The agent MINIMIZES R. Use argmin not argmax. Use .min() not .max() in targets.**

### Path-Level Metric Aggregation (DRSIR Equations 2-4)
```
bw_path   = min(bw_link)     for all links i on path   [bottleneck bandwidth]
delay_path = sum(delay_link) for all links i on path   [total propagation delay]
loss_path  = 1 - product(1 - loss_link) for all links  [combined loss]
```

### Episode Definition
```
sim_time   = 100 simulated seconds per episode
tmon       = 5 simulated seconds per decision step
steps_per_ep = sim_time / tmon = 20 steps per episode
n_episodes = 500 total training episodes
```

---

## 6. Hyperparameters (All from configs/hyperparams.py)

```python
# configs/hyperparams.py
HIDDEN_NEURONS = 50          # neurons in hidden layer (DRSIR Table I)
GAMMA          = 0.1         # discount factor (DRSIR Table I)
REPLAY_START   = 200         # steps before training starts
EPS_MAX        = 1.0         # starting epsilon (full exploration)
EPS_MIN        = 0.05        # minimum epsilon (5% exploration floor)
DECAY_RATE     = 1.0 / 400.0 # epsilon decay per step (DRSIR Table I)
BATCH_SIZE     = 15          # replay buffer sample size (DRSIR Table I)
TARGET_UPDATE  = 100         # sync Target NN every N steps (DRSIR Table I)
BUFFER_SIZE    = 10_000      # replay buffer capacity
LR             = 0.001       # Adam optimizer learning rate
K_PATHS        = 3           # candidate paths per SD pair
T_MON          = 5.0         # simulation seconds per RL step
SIM_TIME       = 100.0       # total simulation time per episode
N_EPISODES     = 500         # total training episodes
STEPS_PER_EP   = int(SIM_TIME / T_MON)   # = 20
BETA           = (1.0, 1.0, 1.0)         # reward weights (β1, β2, β3)
OBS_SIZE       = 1 + K_PATHS * 3         # = 10
N_SD_PAIRS     = 2                        # S1→D1 and S2→D1
ZMQ_PORT       = 5555                     # ns3-gym communication port
```

**RULE**: Every Python file imports these constants from configs/hyperparams.py.
No file may hardcode any of these values inline.

---

## 7. Neural Network Architecture

```
Input:  SD pair index (integer 0 or 1) → cast to LongTensor
        ↓
nn.Embedding(num_embeddings=N_SD_PAIRS=2, embedding_dim=HIDDEN_NEURONS=50)
        ↓  shape: (batch, 50)
nn.Linear(50, 50) + F.relu()
        ↓  shape: (batch, 50)
nn.Linear(50, K_PATHS=3)
        ↓  shape: (batch, 3)   ← one Q-value per candidate path
Output: Q-values for each path (agent picks path with LOWEST Q-value = lowest cost)
```

**Weight initialization**: `nn.init.xavier_uniform_` on both linear layers
(Glorot uniform, from DRSIR).

**Optimizer**: `torch.optim.Adam(lr=LR)`.

**Loss**: `nn.MSELoss()` between predicted Q and target Q.

---

## 8. DQN vs DDQN — Exact Formulas

### DQN Target (used in DQNAgent.train_step)
```python
with torch.no_grad():
    q_next   = self.target(NS).min(1)[0]          # Target net selects AND evaluates
    q_target = R + GAMMA * q_next * (1 - D)
```

### DDQN Target (used in DDQNAgent.train_step — only this changes)
```python
with torch.no_grad():
    best_a   = self.online(NS).argmin(1, keepdim=True)   # Online net SELECTS
    q_next   = self.target(NS).gather(1, best_a).squeeze(1)  # Target net EVALUATES
    q_target = R + GAMMA * q_next * (1 - D)
```

DDQNAgent is a subclass of DQNAgent. It overrides ONLY train_step().
Everything else (act, store, save, load, network architecture, buffer) is inherited.

---

## 9. Evaluation Scenarios

Four scenarios, three algorithms each, three runs each (average results):

| Scenario | S1 Rate | S2 Rate | Extra Condition | Key Claim |
|---|---|---|---|---|
| normal | 4 Mbps | 3 Mbps | none | Establishes baseline |
| congested | 9 Mbps | 8 Mbps | none | DRL outperforms Dijkstra |
| failure | 4 Mbps | 3 Mbps | R1-D1 fails at t=40s | DRL recovers faster |
| mixed | varies | varies | OnOff: high-low-high | DRL adapts dynamically |

For the mixed scenario:
- t=0–30s: 9 Mbps (high)
- t=30–60s: 2 Mbps (low)
- t=60–100s: 9 Mbps (high)

---

## 10. Metrics Collected Per Run

From FlowMonitor XML (via parse_flowmon.py):
```
throughput_mbps  = (rxBytes * 8) / (duration_seconds * 1e6)
avg_delay_ms     = delaySum_ns / rxPackets / 1e6
loss_ratio       = (txPackets - rxPackets) / txPackets
```

From training logs (CSV written during training):
```
episode          = episode index (0 to N_EPISODES-1)
reward           = total episode cost (sum of step rewards)
avg_loss         = mean training loss in this episode
epsilon          = epsilon value at end of episode
```

---

## 11. Required Output Files

### Training
```
results/logs/dqn_training.csv    # columns: episode, reward, avg_loss, epsilon
results/logs/ddqn_training.csv   # same columns
results/checkpoints/dqn_ep0.pt   results/checkpoints/dqn_ep250.pt
results/checkpoints/dqn_final.pt results/checkpoints/ddqn_final.pt
```

### Evaluation
```
results/logs/normal_comparison.csv     # index=algo, cols=throughput,delay,loss
results/logs/congested_comparison.csv
results/logs/failure_comparison.csv
results/logs/mixed_comparison.csv
results/raw/{algo}_{scenario}_run{N}.xml   # raw FlowMonitor per run
```

### Figures (all saved as PDF at dpi=180)
```
results/plots/fig1_convergence.pdf   # episode cost vs episode (DQN + DDQN)
results/plots/fig2_throughput.pdf    # grouped bar, 4 scenarios, 3 algorithms
results/plots/fig3_delay.pdf         # same layout, delay metric
results/plots/fig4_loss.pdf          # same layout, loss metric
results/plots/fig5_recovery.pdf      # time-series throughput, failure scenario
results/plots/fig6_trainloss.pdf     # training loss vs steps (DQN + DDQN)
```

---

## 12. Critical Rules — Read Before Writing Any Code

1. **Agent MINIMIZES reward** (it is a cost function). Use `argmin` not `argmax`
   in `act()`. Use `.min()` not `.max()` in DQN target. Use `.argmin()` in DDQN.

2. **Neural network input is ONLY obs[0] cast to int**. Never feed obs[1:] to the
   network. obs[1:] is used only in the reward function.

3. **Never hardcode any IP address** in ns-3 C++ code. Always resolve D1's IP
   dynamically with `GetObject<Ipv4>()->GetAddress(1,0).GetLocal()` after
   `addr.Assign()` has been called.

4. **Never hardcode any hyperparameter** in Python files. Always import from
   `configs/hyperparams.py`.

5. **All code blocks in ns-3 use `using namespace ns3;`**. Never use the ns3::
   prefix inline when this is declared.

6. **Episode termination**: the simulation runs for exactly SIM_TIME=100s.
   `GetGameOver()` in the C++ env returns true when
   `Simulator::Now().GetSeconds() >= SIM_TIME`.

7. **Step interval**: the ns-3 env advances the simulation by T_MON=5 simulated
   seconds per `step()` call. This is implemented with `Simulator::Schedule`.

8. **devs[6]** = the R1-D1 link (index 6 in the link array). This is the link
   that fails in the failure scenario.

9. **DDQNAgent inherits from DQNAgent** and overrides ONLY `train_step()`.
   The training script for DDQN is identical to DQN's except it instantiates
   `DDQNAgent()` instead of `DQNAgent()`.

10. **The ZeroMQ port** is 5555. Both the C++ simulation and the Python wrapper
    must use the same port. It is defined as ZMQ_PORT in hyperparams.py and
    passed to ns3-gym on both sides.

11. **Link failure implementation**: use `RateErrorModel` with `ErrorRate=1.0`
    on `devs[6].Get(1)` (the receiving device of R1→D1). Schedule this at
    `Seconds(failTime)` using `Simulator::Schedule`.

12. **FlowMonitor delaySum** is returned as a string like `"12345678ns"`. Always
    strip the `"ns"` suffix and convert to float before dividing.

13. **Routing table update**: use `Ipv4StaticRoutingHelper` to manually install
    routes. When the agent selects a path (e.g., [0, 2, 5] for S1-R1-D1),
    set the next-hop for node 0's routing table to route D1's IP via the
    interface that connects to node 2.

14. **Random seed**: always set `ns3::RngSeedManager::SetSeed(42)` and
    `ns3::RngSeedManager::SetRun(1)` for reproducibility. In Python, set
    `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`.

---

## 13. ns3-gym Bridge — How It Works

The bridge uses the OpenGym module from `contrib/opengym`.

### C++ Side (routing_env.cc)
Subclass `OpenGymEnv` and implement:
```
GetObservationSpace()  → returns Box(low=0, high=1, shape=(10,), dtype=float32)
GetActionSpace()       → returns Discrete(K=3)
GetObservation()       → returns current 10-float obs vector
GetReward()            → returns 0.0f (reward computed Python-side)
GetGameOver()          → returns true when sim time >= SIM_TIME
GetExtraInfo()         → returns empty string
ExecuteActions(action) → calls UpdateRouting(pathIdx) to update routing tables
ScheduleNextStep()     → calls Simulator::Schedule(Seconds(T_MON), &Step, this)
```

### Python Side (env/ns3_wrapper.py)
Uses `ns3gym.Ns3Env` as base:
```python
env = Ns3Env(port=ZMQ_PORT, stepTime=T_MON, startSim=True,
             simSeed=42, simArgs={...}, debug=False)
obs  = env.reset()    # returns obs vector as numpy array (10,)
obs, _, done, _ = env.step(action)   # action is int 0/1/2
```

**The reward is NOT computed in C++.** `GetReward()` in C++ returns `0.0f`.
The actual DRSIR reward is computed in Python using `metrics.compute_reward(obs, action)`.

---

## 14. Phase Summary

```
Phase 1  ns-3 C++ simulation + Dijkstra baseline + FlowMonitor metrics
         Files: routing_sim.cc, baseline/parse_flowmon.py, baseline/run_baseline.py
         Done when: 3 scenario XML files generated, metrics parsed correctly

Phase 2  ns3-gym integration (C++ env + Python wrapper + reward)
         Files: routing_env.h, routing_env.cc, env/path_precompute.py,
                env/metrics.py, env/ns3_wrapper.py
         Done when: 50-episode random action loop runs, rewards vary per action

Phase 3  DRL agent + training + DDQN + evaluation + plotting
         Files: agent/network.py, agent/replay_buffer.py, agent/dqn_agent.py,
                training/train_dqn.py, training/train_ddqn.py,
                training/evaluate.py, results/plots/generate_all.py
         Done when: all 6 figures generated, all comparison CSVs saved
```

---

## 15. Running Commands Reference

```bash
# Build ns-3 after any C++ change
cd ~/ns-3.38 && ./ns3 build

# Run baseline (Dijkstra, no RL)
./ns3 run "drl_routing/routing_sim --simTime=100 --scenario=normal \
           --algo=dijkstra --output=results/raw/dijkstra_normal_run0.xml"

# Run with RL agent (Terminal 1: ns-3, Terminal 2: Python)
# Terminal 1:
./ns3 run "drl_routing/routing_sim --simTime=100 --rl=true --port=5555"
# Terminal 2:
cd ~/drl_project && python3 training/train_dqn.py

# Run evaluation after training
python3 training/evaluate.py

# Generate all plots
python3 results/plots/generate_all.py
```

---

*End of context.md — this file must be loaded into the AI IDE context
(via .cursorrules or equivalent) before working on any file in this project.*