# 🚀 DRL-Based Adaptive Network Routing

| | |
|---|---|
| [<img src="https://img.shields.io/badge/▶%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=googledrive" />](https://drive.google.com/file/d/1LsBZg2RAi3vqA09-DrNPHtNm2ZFrE2fE/view?usp=sharing) | [<img src="https://img.shields.io/badge/📄%20Read%20Report-Google%20Drive-red?style=for-the-badge&logo=googledrive" />](https://drive.google.com/file/d/1DyuBlhz-sqmWcdMqXINm4TwUa7ERW263/view?usp=sharing) |


**Authors:** Muhammad Sabeeh (23K-0002), Rayyan Merchant (23K-0073)

A Deep Reinforcement Learning (DRL) system that dynamically routes network traffic using DQN and DDQN algorithms, built on **ns-3.35** + **ns3-gym** + **PyTorch**. The agent learns to minimize delay, packet loss, and maximize throughput — outperforming traditional Dijkstra routing.

---

## 📁 Project Structure

```text
CN Project/
├── agent/                      # PyTorch DRL agents
│   ├── network.py              # QNetwork architecture (Embedding → FC → 3 Q-values)
│   ├── dqn_agent.py            # DQNAgent + DDQNAgent classes
│   └── replay_buffer.py        # Experience Replay Buffer
├── baseline/                   # Dijkstra baseline
│   ├── run_baseline.py         # Runs 3 scenarios × 3 runs via ns-3
│   └── parse_flowmon.py        # Parses FlowMonitor XML → DataFrame
├── configs/
│   └── hyperparams.py          # Single source of truth for ALL hyperparameters
├── env/                        # RL environment
│   ├── ns3_wrapper.py          # Gym wrapper around ns3-gym ZMQ interface
│   └── metrics.py              # DRSIR reward function (cost minimization)
├── training/                   # Training & evaluation scripts
│   ├── train_dqn.py            # DQN training (500 episodes)
│   ├── train_ddqn.py           # DDQN training (500 episodes)
│   ├── run_inference.py        # Run trained agent in greedy mode
│   ├── evaluate.py             # Evaluate all algorithms × all scenarios
│   └── health_check.py         # 5-point training verification
├── results/                    # Generated outputs
│   ├── checkpoints/            # Saved model weights (.pt files)
│   ├── logs/                   # Training CSVs + comparison CSVs
│   ├── plots/                  # PDF figures + generate_all.py
│   └── raw/                    # Raw FlowMonitor XML files
├── routing_sim.cc              # ns-3 C++ simulation (topology + traffic + opengym hooks)
├── routing_env.cc              # ns-3 C++ RL environment (obs/action/reward interface)
├── routing_env.h               # Header for RoutingEnv class
└── README.md                   # This file
```

---

## 🏗️ Network Topology

```
         S1(0)──────R1(2)──────D1(5)
          │  \       │  \      / |
          │   \      │   \    /  |
          │    R2(3)─┘    R3(4) |
          │   /      │   /      |
         S2(1)──────R2(3)───────┘
```

- **6 nodes**: S1, S2 (sources), R1, R2, R3 (routers), D1 (destination)
- **10 point-to-point links** with varying bandwidth (3–10 Mbps) and delay (2–12 ms)
- **3 candidate paths** per source-destination pair
- **Link failure**: R1–D1 fails at t=40s in failure scenario

---

## 🛠️ Environment Setup (Ubuntu 22.04 WSL)

### Prerequisites Already Installed
The WSL environment is pre-configured with:
- **ns-3.35** at `~/ns-allinone-3.35/ns-3.35/` (WAF-based build)
- **ns3-gym** (opengym) in `contrib/opengym/` (WAF-compatible `app` branch)
- **Python 3.10** with: `torch`, `gym`, `ns3gym`, `zmq`, `protobuf`, `pandas`, `matplotlib`, `numpy`
- **C++ files** compiled in `scratch/drl_routing/`

### If Setting Up Fresh

```bash
# 1. Install system dependencies
sudo apt update && sudo apt install -y gcc g++ python3 python3-pip \
    libzmq5-dev libprotobuf-dev protobuf-compiler

# 2. Download and extract ns-3.35
cd ~
wget https://www.nsnam.org/releases/ns-allinone-3.35.tar.bz2
tar xf ns-allinone-3.35.tar.bz2

# 3. Clone ns3-gym (WAF-compatible branch)
cd ~/ns-allinone-3.35/ns-3.35/contrib
git clone https://github.com/tkn-tub/ns3-gym.git opengym
cd opengym && git checkout app

# 4. Copy C++ simulation files
mkdir -p ~/ns-allinone-3.35/ns-3.35/scratch/drl_routing
cp routing_sim.cc routing_env.cc routing_env.h ~/ns-allinone-3.35/ns-3.35/scratch/drl_routing/

# 5. Configure and build ns-3
cd ~/ns-allinone-3.35/ns-3.35
./waf configure --build-profile=optimized --disable-examples --disable-tests --disable-python
./waf build -j4

# 6. Install Python dependencies
pip install torch numpy pandas matplotlib gym zmq protobuf
cd ~/ns-allinone-3.35/ns-3.35/contrib/opengym/model/ns3gym
pip install -e .

# 7. Patch ns3gym for NumPy 2.0 compatibility
# In ns3gym/ns3env.py, replace:
#   np.float → np.float64
#   np.int   → np.int64
#   np.uint  → np.uint64

# 8. Copy Python project
cp -r "CN Project/" ~/drl_project/
```

---

## 📊 Understanding the Results

### Training Logs (`results/logs/`)
| Column | Description |
|--------|-------------|
| `episode` | Episode number (0–499) |
| `reward` | Total DRSIR cost for the episode (lower = better) |
| `avg_loss` | Average MSE loss for the episode |
| `epsilon` | Exploration rate (1.0 → 0.05) |
| `action{0,1,2}_frac` | Fraction of steps using each path |

### Key Hyperparameters (`configs/hyperparams.py`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_EPISODES` | 500 | Training episodes |
| `STEPS_PER_EP` | 20 | Steps per episode (100s ÷ 5s) |
| `GAMMA` | 0.1 | Discount factor (near-sighted) |
| `EPS_MAX/MIN` | 1.0/0.05 | Epsilon-greedy range |
| `REPLAY_START` | 200 | Steps before training begins |
| `BATCH_SIZE` | 15 | Replay buffer mini-batch |
| `K_PATHS` | 3 | Candidate paths per SD pair |
| `HIDDEN_NEURONS` | 50 | Network hidden layer size |

### Health Check Criteria
| Check | What It Verifies |
|-------|-----------------|
| 1. Epsilon decay | Started at 1.0, ended at 0.05 |
| 2. Loss non-zero | ≥30% episodes have training loss |
| 3. Cost trend | Later episodes cost less than early ones |
| 4. Path exploration | Agent uses all 3 paths |
| 5. No NaN | No corrupted values in logs |

---

## 🔧 How It Works (Architecture)

```
┌─────────────────────────────────────────────────────────┐
│                     ns-3 Simulator                       │
│  routing_sim.cc → topology, traffic, FlowMonitor        │
│  routing_env.cc → RoutingEnv (obs/action/reward)        │
│         ↕ ZeroMQ (port 5555) via ns3-gym                │
├─────────────────────────────────────────────────────────┤
│                    Python Agent                          │
│  ns3_wrapper.py → Gym interface                         │
│  metrics.py → DRSIR reward computation                  │
│  dqn_agent.py → DQN/DDQN with experience replay        │
│  network.py → QNetwork (Embedding → FC → 3 Q-values)   │
└─────────────────────────────────────────────────────────┘
```

1. **ns-3** simulates the network, generates traffic, measures throughput/delay/loss
2. **ns3-gym** exposes observations (per-path BW, delay, loss) and accepts routing actions via ZMQ
3. **Python agent** observes network state, selects a path (ε-greedy), receives DRSIR cost
4. **DQN/DDQN** learns to minimize cost using experience replay and target networks

### DQN vs DDQN
- **DQN**: Target uses `Q_target(s').min()` directly → can overestimate
- **DDQN**: Online network selects action (`argmin`), target network evaluates it → more stable

---

## 💡 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Address already in use (ZMQ port 5555)` | `killall -9 drl_routing` in WSL |
| `ns3gym import error` | `cd ~/ns-allinone-3.35/ns-3.35/contrib/opengym/model/ns3gym && pip install -e .` |
| `np.float deprecated` | Patch `ns3env.py`: `np.float` → `np.float64` |
| `gymnasium not found` | Use `import gym` (not `gymnasium`) — ns3gym uses old gym |
| Build fails on Python bindings | Add `--disable-python` to waf configure |
| ns-3 runs but Python doesn't connect | Make sure ns-3 has `--enableRL=true` |
| Training loss is 0 for early episodes | Normal — buffer needs 200 steps (10 episodes) to warm up |

---

## 📝 Presentation Talking Points

1. **Problem**: Static routing (Dijkstra) can't adapt to congestion or link failures
2. **Solution**: DRL agent learns optimal routing through trial-and-error
3. **Architecture**: ns-3 (C++) ↔ ZMQ ↔ Python (PyTorch DQN/DDQN)
4. **Results**: Agent trains for 500 episodes, epsilon decays from 1.0→0.05
5. **Baseline**: 3 scenarios tested — Normal (0% loss), Congested (10% loss), Failure (15% loss)
6. **Health**: All 10/10 health checks pass for both DQN and DDQN

Happy Routing! 🚀
