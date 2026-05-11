# MASTER DEBUGGING PROMPT
# Paste this entire prompt into Cursor (or your AI IDE) in one go.
# Make sure context.md is loaded as .cursorrules before running this.
# The AI will read every file, audit every layer, and produce a single
# structured report of all errors, contradictions, and fixes required.

---

You are performing a full pre-execution code audit of this project.
Your job is NOT to run anything. Your job is to READ every file listed
below, then systematically check every item in every audit category,
and produce a single structured report at the end.

Do NOT fix anything yet. First read everything, then report everything.
After the report is complete, I will tell you which fixes to apply.

---

## STEP 1 — READ ALL FILES IN THIS EXACT ORDER

Read each file completely before moving to Step 2. Do not skim.

C++ files (in ~/ns-3.38/scratch/drl_routing/):
1. routing_env.h
2. routing_env.cc
3. routing_sim.cc

Python files (in ~/drl_project/):
4.  configs/hyperparams.py
5.  configs/topology.py               (if it exists)
6.  agent/network.py
7.  agent/replay_buffer.py
8.  agent/dqn_agent.py
9.  env/path_precompute.py
10. env/metrics.py
11. env/ns3_wrapper.py
12. env/test_random_actions.py
13. training/train_dqn.py
14. training/train_ddqn.py
15. training/run_inference.py
16. training/evaluate.py
17. training/health_check.py
18. baseline/parse_flowmon.py
19. baseline/run_baseline.py
20. baseline/validate_metrics.py
21. results/plots/generate_all.py

Confirm you have read all files before proceeding. If any file is
missing, list it as MISSING in the report.

---

## STEP 2 — AUDIT CATEGORIES

Work through every category below. For every item, check the actual
code in the files you just read. Report every issue found, no matter
how small. Use the exact format described in Step 3.

---

### CATEGORY A — Critical Logic Errors (project-breaking)

A1. ARGMIN vs ARGMAX — THE MOST IMPORTANT CHECK
    This agent MINIMIZES a cost function. Verify every occurrence:

    In agent/network.py QNetwork.best_action():
    - Must use .argmin().item() NOT .argmax()
    - Any use of argmax here is a fatal bug

    In agent/dqn_agent.py DQNAgent.train_step() DQN target:
    - Must use self.target(NS).min(1)[0]  NOT .max(1)[0]
    - q_target = R + GAMMA * q_next * (1 - D)  — verify this exactly

    In agent/dqn_agent.py DDQNAgent.train_step() DDQN target:
    - Online net must use .argmin(1, keepdim=True) NOT .argmax
    - Target net evaluates: self.target(NS).gather(1, best_a).squeeze(1)
    - This is the ONLY difference from DQNAgent.train_step()
    - Verify DDQNAgent does NOT copy the DQN target formula

    In training/train_dqn.py and train_ddqn.py:
    - action = agent.act(state) — verify act() is called correctly
    - reward = compute_reward(...) — verify this is called AFTER env.step()
      using the PREVIOUS obs, not next_obs

    In training/run_inference.py:
    - agent.epsilon = 0.0 must be set BEFORE the episode loop
    - act() with epsilon=0.0 must always return greedy (lowest Q) action

A2. REWARD COMPUTATION — OBS SPLIT
    The observation vector has 10 elements.
    obs[0]   = SD pair index → cast to int → fed to nn.Embedding
    obs[1:10] = path metrics → used ONLY in compute_reward()

    In training/train_dqn.py and train_ddqn.py:
    - Verify: state = int(obs[0])  — correct cast
    - Verify: reward = compute_reward(obs, action, ...)
              uses the FULL obs (not just obs[1:])
              because compute_reward() does the slicing internally
    - Verify: agent.store(state, ...) stores the INTEGER state,
              not the full obs array
    - Verify: next_state = int(next_obs[0]) — same cast on next_obs
    - FATAL if: reward = compute_reward(next_obs, ...) — must use
                current obs, not next_obs

    In env/metrics.py compute_reward():
    - Verify: path_metrics = obs[1:].reshape(k, 3)
              This must slice from index 1, not index 0
    - Verify: norm[action, 0] is bandwidth (higher raw = better)
    - Verify: norm[action, 1] is delay (lower raw = better)
    - Verify: norm[action, 2] is loss (lower raw = better)
    - Verify reward formula: beta1*(1/(bw_n+eps)) + beta2*delay_n + beta3*loss_n
              NOT: beta1*bw_n + beta2*delay_n + beta3*loss_n
              (the BW term must be INVERTED — this is the most common mistake)

A3. EPSILON DECAY ORDER
    In agent/dqn_agent.py DQNAgent.act():
    - Epsilon MUST be updated BEFORE the action is selected
    - Correct: self.epsilon = max(EPS_MIN, EPS_MAX - self.steps * DECAY_RATE)
               then use self.epsilon to decide explore/exploit
    - Wrong: updating epsilon after action selection
    - Verify self.steps is incremented in store(), NOT in act()
      (incrementing in act() would decay epsilon even during warmup
       before training starts)

A4. TARGET NETWORK SYNC
    In DQNAgent.train_step():
    - Target sync must happen AFTER the gradient update (after optim.step())
      NOT before
    - Sync condition: if self.steps % TARGET_UPDATE == 0
    - Uses self.steps (total steps), NOT episode count
    - Verify: self.target.load_state_dict(self.online.state_dict())
              NOT: self.online.load_state_dict(self.target.state_dict())
              (direction matters — copy Online → Target, never reverse)

A5. DONE FLAG HANDLING
    In DQNAgent.train_step():
    - q_target = R + GAMMA * q_next * (1.0 - D)
    - D must be float32 (0.0 or 1.0), NOT bool
    - When done=True: (1-D)=0, so future reward is zeroed — correct
    - Verify ReplayBuffer stores done as float, not bool

A6. DDQN IS SUBCLASS CHECK
    Verify DDQNAgent(DQNAgent) — inherits from DQNAgent
    Verify DDQNAgent ONLY overrides train_step()
    Verify DDQNAgent does NOT redefine: __init__, act, store, save, load
    If any of those are redefined in DDQNAgent, flag as error

---

### CATEGORY B — ns-3 C++ Specific Errors

B1. IP ADDRESS RESOLUTION
    In routing_sim.cc:
    - Verify d1_ip is resolved with GetObject<Ipv4>()->GetAddress(1,0).GetLocal()
    - Verify this line comes AFTER Ipv4GlobalRoutingHelper::PopulateRoutingTables()
    - Verify d1_ip is NOT hardcoded anywhere (no "10.x.x.x" literals for dest)
    - Verify the port variable is used consistently in both OnOff and PacketSink

B2. LINK ARRAY ORDER
    In routing_sim.cc, verify the LinkDef array has EXACTLY these 10 entries
    in this EXACT order (order determines devs[] index used in failure logic):
    Index 0: (0,2) S1-R1  10Mbps 2ms
    Index 1: (0,3) S1-R2  7Mbps  4ms
    Index 2: (1,3) S2-R2  7Mbps  4ms
    Index 3: (1,4) S2-R3  10Mbps 2ms
    Index 4: (2,3) R1-R2  5Mbps  5ms  ← bottleneck
    Index 5: (3,4) R2-R3  5Mbps  5ms  ← bottleneck
    Index 6: (2,5) R1-D1  8Mbps  8ms  ← THIS IS THE FAILURE LINK
    Index 7: (3,5) R2-D1  6Mbps  6ms
    Index 8: (4,5) R3-D1  4Mbps  10ms
    Index 9: (2,4) R1-R3  3Mbps  12ms
    Any deviation from this order means the failure scenario hits the wrong link.

B3. LINK FAILURE IMPLEMENTATION
    In routing_sim.cc:
    - Verify failure is applied to devs[6].Get(1) — the D1-side device
      NOT devs[6].Get(0) — that would be the R1-side device
    - Verify RateErrorModel ErrorRate is 1.0 (drops ALL packets)
    - Verify the failure is scheduled with Simulator::Schedule(Seconds(failTime), ...)
      NOT called directly (direct call would fail before simulation starts)
    - Verify failTime default is 40.0 (not 0.0)

B4. FLOWMONITOR SERIALIZATION
    In routing_sim.cc:
    - Verify fm->SerializeToXmlFile(outFile, true, true) is called AFTER
      Simulator::Run() completes, NOT before
    - Verify Simulator::Destroy() is called AFTER SerializeToXmlFile()
      NOT before (destroying before serializing loses all data)

B5. RANDOM SEED
    In routing_sim.cc:
    - Verify RngSeedManager::SetSeed(seed) AND RngSeedManager::SetRun(runNum)
      are both called after CommandLine::Parse() and before any node creation

B6. NS3-GYM PORT
    In routing_sim.cc (RL mode):
    - Verify OpenGymInterface is created with port 5555
    - Verify this matches ZMQ_PORT = 5555 in hyperparams.py

B7. ROUTING ENV GETREWARD
    In routing_env.cc GetReward():
    - Verify it returns 0.0f (the literal float zero)
    - The actual reward is computed Python-side in metrics.py
    - If GetReward() returns any non-zero computation, it will confuse
      the Python agent which ignores the C++ reward entirely

B8. EXECUTIVEACTIONS SD PAIR ALTERNATION
    In routing_env.cc ExecuteActions():
    - Verify m_currentSD is toggled: m_currentSD = (m_currentSD + 1) % 2
    - This ensures the agent alternates routing decisions for S1→D1 and S2→D1
    - Verify m_currentSD starts at 0 in constructor

B9. OBSERVATION VECTOR SIZE
    In routing_env.cc GetObservation():
    - Verify the returned container has exactly 10 float values:
      obs[0] = (float)m_currentSD
      obs[1..9] = path metrics (3 metrics × 3 paths = 9 values)
    - Verify GetObservationSpace() declares shape {10}, dtype float32

B10. SIMULATOR SCHEDULE STEP INTERVAL
    In routing_env.cc or routing_sim.cc:
    - The recurring step must be scheduled with Seconds(5.0) — the tmon value
    - Verify the step scheduling creates a loop (each step schedules the next)
      NOT a one-time schedule that stops after the first step

---

### CATEGORY C — Python Architecture Errors

C1. HYPERPARAMS IMPORT DISCIPLINE
    Read every Python file. For each file verify:
    - No numeric hyperparameter is hardcoded inline
    - Specifically check for these literals that must NOT appear inline:
      50 (HIDDEN_NEURONS), 0.1 (GAMMA), 200 (REPLAY_START),
      1.0 (EPS_MAX), 0.05 (EPS_MIN), 0.0025 (DECAY_RATE),
      15 (BATCH_SIZE), 100 (TARGET_UPDATE), 10000 (BUFFER_SIZE),
      0.001 (LR), 3 (K_PATHS), 5.0 (T_MON), 100.0 (SIM_TIME),
      500 (N_EPISODES), 20 (STEPS_PER_EP), 5555 (ZMQ_PORT)
    - Exception: hyperparams.py itself may define these
    - Flag every file that has any of these literals hardcoded

C2. QNETWORK ARCHITECTURE
    In agent/network.py:
    - Verify nn.Embedding(N_SD_PAIRS, HIDDEN_NEURONS) — NOT nn.Embedding(10, ...)
    - Verify N_SD_PAIRS = 2 is imported from hyperparams, not hardcoded
    - Verify forward() takes LongTensor (int indices), NOT FloatTensor
    - Verify NO softmax in forward() — raw Q-values only
    - Verify xavier_uniform_ applied to fc1.weight AND out.weight
    - Verify bias initialized to zeros for both layers

C3. REPLAY BUFFER DTYPE
    In agent/replay_buffer.py sample():
    - states      must be np.int64    (fed to nn.Embedding as LongTensor)
    - actions     must be np.int64
    - rewards     must be np.float32  (MSE loss operates on float32)
    - next_states must be np.int64
    - dones       must be np.float32  (multiplied in target computation)
    - Any wrong dtype will cause a silent type error in PyTorch

C4. TRAINING LOOP VARIABLE FLOW
    In training/train_dqn.py and train_ddqn.py, verify this EXACT flow
    for each step:
    Step 1: state      = int(obs[0])
    Step 2: action     = agent.act(state)
    Step 3: next_obs, _, terminated, truncated, _ = env.step(action)
    Step 4: done       = terminated or truncated
    Step 5: reward     = compute_reward(obs, action, K_PATHS, BETA)
                         ← uses OBS (before step), NOT next_obs
    Step 6: next_state = int(next_obs[0])
    Step 7: agent.store(state, action, reward, next_state, float(done))
    Step 8: loss       = agent.train_step()
    Step 9: obs        = next_obs
                         ← update obs LAST, after reward is computed

    Any deviation from this order (especially computing reward from
    next_obs instead of obs) is a correctness bug.

C5. GYMNASIUM VS GYM API
    In env/ns3_wrapper.py:
    - If using gymnasium: env.step() returns 5 values:
      (obs, reward, terminated, truncated, info)
    - If using gym (old): env.step() returns 4 values:
      (obs, reward, done, info)
    - Verify the unpacking in the wrapper matches the installed version
    - Verify reset() returns (obs, info) for gymnasium or just obs for gym
    - Check which is actually installed: import gymnasium vs import gym
    - Verify train_dqn.py unpacking matches ns3_wrapper.py's return signature

C6. NS3_WRAPPER REWARD PASS-THROUGH
    In env/ns3_wrapper.py step():
    - The reward from ns-3 (always 0.0) must be DISCARDED
    - The wrapper must return reward=0.0 (or any constant) from step()
    - The training script ignores this returned reward entirely
    - The real reward comes from metrics.compute_reward(obs, action)
    - Verify no file uses the reward returned by env.step() for training

C7. PATH PRECOMPUTATION CORRECTNESS
    In env/path_precompute.py:
    - Verify PATHS[(0,5)] contains exactly 3 paths
    - Verify PATHS[(0,5)][0] = [0, 2, 5]    (S1-R1-D1)
    - Verify PATHS[(0,5)][1] = [0, 3, 5]    (S1-R2-D1)
    - Verify PATHS[(0,5)][2] starts with 0 and ends with 5
    - Verify PATHS[(1,5)] contains exactly 3 paths
    - Verify PATHS[(1,5)][0] = [1, 4, 5]    (S2-R3-D1)
    - Verify the graph edges match the topology in context.md exactly:
      edge (0,2) bw=10 delay=2, edge (0,3) bw=7 delay=4,
      edge (1,3) bw=7 delay=4,  edge (1,4) bw=10 delay=2,
      edge (2,3) bw=5 delay=5,  edge (3,4) bw=5 delay=5,
      edge (2,5) bw=8 delay=8,  edge (3,5) bw=6 delay=6,
      edge (4,5) bw=4 delay=10, edge (2,4) bw=3 delay=12

C8. METRICS MIN-MAX EDGE CASE
    In env/metrics.py minmax_normalize():
    - When all k paths have identical value for a metric:
      (vmax - vmin) < 1e-8 → must set normalized to 0.5 for ALL paths
      NOT to 0.0 (which would make the BW term 1/0.5 = 2.0 — acceptable)
      NOT to 1.0 (which would make delay/loss term all 1.0 — wrong)
    - Verify the edge case is handled with explicit 0.5 assignment

C9. DDQN TRAIN_STEP ISOLATION
    In agent/dqn_agent.py DDQNAgent.train_step():
    - Verify it has its OWN complete implementation — not calling super().train_step()
    - The parent's train_step() uses DQN target; calling it from DDQN would
      silently use DQN logic instead of DDQN logic
    - The correct pattern is to duplicate the full train_step() body and
      change ONLY the target computation lines

C10. INFERENCE EPSILON
    In training/run_inference.py:
    - Verify agent.epsilon = 0.0 is set after agent.load() is called
    - Setting it before load() would be overwritten by the loaded value
    - Verify agent.load() does NOT restore epsilon to EPS_MIN from checkpoint
      OR verify that after load(), epsilon is explicitly reset to 0.0

---

### CATEGORY D — Cross-File Consistency Errors

D1. OBSERVATION SIZE AGREEMENT
    Check that OBS_SIZE = 10 is consistent across ALL these locations:
    - configs/hyperparams.py: OBS_SIZE = 1 + K_PATHS * 3  (= 1 + 3*3 = 10)
    - routing_env.h: obs vector declared as 10 floats
    - routing_env.cc GetObservationSpace(): shape {10}
    - routing_env.cc GetObservation(): appends exactly 10 values
    - env/ns3_wrapper.py observation_space shape: (OBS_SIZE,) = (10,)
    - env/metrics.py: obs[1:].reshape(K_PATHS, 3) = obs[1:10].reshape(3,3)
    Any mismatch causes a silent shape error or index out of bounds.

D2. K_PATHS AGREEMENT
    Check K_PATHS = 3 is consistent across:
    - configs/hyperparams.py: K_PATHS = 3
    - routing_env.h constructor parameter: kPaths = 3
    - routing_sim.cc: passes 3 to RoutingEnv constructor
    - agent/network.py: output layer size = K_PATHS = 3
    - agent/replay_buffer.py: action range [0, K_PATHS-1]
    - env/path_precompute.py: k=K_PATHS=3 paths per SD pair
    - env/metrics.py: path_metrics shape = (K_PATHS, 3) = (3, 3)
    - env/ns3_wrapper.py: action_space = Discrete(K_PATHS) = Discrete(3)

D3. ZMQ PORT AGREEMENT
    ZMQ_PORT = 5555 must match in:
    - configs/hyperparams.py: ZMQ_PORT = 5555
    - routing_sim.cc: OpenGymInterface port argument = 5555
    - env/ns3_wrapper.py: Ns3Env port argument = ZMQ_PORT or 5555
    Any mismatch causes a connection refused error.

D4. T_MON AGREEMENT
    T_MON = 5.0 (simulated seconds per step) must match in:
    - configs/hyperparams.py: T_MON = 5.0
    - routing_sim.cc: Simulator::Schedule(Seconds(5.0), ...)
    - routing_env.cc: m_tmon = 5.0 (from constructor parameter)
    - routing_sim.cc: passes T_MON value (5.0) to RoutingEnv constructor
    - env/ns3_wrapper.py: stepTime=T_MON argument to Ns3Env

D5. STEPS_PER_EP AGREEMENT
    STEPS_PER_EP = 20 (= SIM_TIME/T_MON = 100/5) must match in:
    - configs/hyperparams.py: STEPS_PER_EP = int(SIM_TIME / T_MON) = 20
    - training/train_dqn.py inner loop: for step in range(STEPS_PER_EP)
    - training/train_ddqn.py: same
    - training/health_check.py: references to episode length
    - results/plots/generate_all.py: x-axis computation for fig6

D6. N_SD_PAIRS AGREEMENT
    N_SD_PAIRS = 2 must match in:
    - configs/hyperparams.py: N_SD_PAIRS = 2
    - agent/network.py: nn.Embedding(N_SD_PAIRS, ...) — MUST NOT be hardcoded
    - env/path_precompute.py: exactly 2 entries in PATHS dict
    - env/path_precompute.py: SD_PAIRS dict has exactly 2 keys (0 and 1)
    - routing_env.cc: m_currentSD alternates between 0 and 1 only

D7. HYPERPARAMS IMPORT COMPLETENESS
    In training/train_dqn.py, verify these are ALL imported from hyperparams:
    N_EPISODES, STEPS_PER_EP, K_PATHS, BETA, LOGS_DIR, CKPT_DIR,
    RANDOM_SEED, REPLAY_START
    Report any that are missing from the import.

    In agent/dqn_agent.py, verify these are ALL imported:
    GAMMA, REPLAY_START, EPS_MAX, EPS_MIN, DECAY_RATE,
    BATCH_SIZE, TARGET_UPDATE, BUFFER_SIZE, LR, K_PATHS

    In env/metrics.py, verify these are imported:
    K_PATHS, BETA

D8. CSV COLUMN NAME AGREEMENT
    In baseline/parse_flowmon.py, verify the returned DataFrame has exactly:
    columns = ['flow_id', 'throughput_mbps', 'avg_delay_ms', 'loss_ratio']

    In baseline/parse_flowmon.py summarize(), verify the returned dict has:
    keys = ['throughput', 'delay', 'loss']  (shortened names for comparison)

    In training/evaluate.py, verify it accesses df_sc columns using the
    summarize() output keys: 'throughput', 'delay', 'loss'
    NOT the raw DataFrame column names.

    In results/plots/generate_all.py, verify it accesses comparison CSVs
    using: 'throughput', 'delay', 'loss' — matching evaluate.py output.

D9. CHECKPOINT FILENAME AGREEMENT
    In agent/dqn_agent.py save(filename):
    - Saves to: os.path.join(CKPT_DIR, filename)
    In training/train_dqn.py:
    - Saves: f"dqn_ep{ep}.pt" at ep=0,50,100,... and "dqn_final.pt"
    In training/train_ddqn.py:
    - Saves: f"ddqn_ep{ep}.pt" and "ddqn_final.pt"
    In training/run_inference.py:
    - Loads: f"{args.algo}_final.pt" by default
    Verify all filenames are consistent between save and load calls.

D10. BETA USAGE AGREEMENT
    BETA = (1.0, 1.0, 1.0) from hyperparams.py.
    In env/metrics.py compute_reward(): uses beta parameter
    In training/train_dqn.py: passes BETA to compute_reward()
    In training/run_inference.py: passes BETA to compute_reward()
    Verify beta is never passed as positional (1.0, 1.0, 1.0) inline.

---

### CATEGORY E — FlowMonitor XML Parsing Errors

E1. DELAYSUM PARSING
    In baseline/parse_flowmon.py:
    - The delaySum attribute looks like: "+123456789.0ns" or "0ns"
    - Must strip "ns" AND "+" before converting to float
    - Correct: float(flow.get("delaySum","0ns").replace("ns","").replace("+",""))
    - Wrong: float(flow.get("delaySum","0ns").replace("ns",""))
      (the "+" sign will cause a ValueError on some XML outputs)
    - Verify: avg_delay_ms = delay_sum_ns / rx_packets / 1e6
      NOT: delay_sum_ns / 1e6 (forgetting to divide by packet count)

E2. THROUGHPUT CALCULATION
    In baseline/parse_flowmon.py:
    - throughput_mbps = (rxBytes * 8) / (duration * 1e6)
    - duration should be 99.0 (simTime=100 minus 1s app start delay)
      NOT 100.0 (off by 1 second causes slight underestimate)
    - Verify rxBytes is cast to int or float before multiplication
    - Verify the result is in Mbps (not bps or Kbps)

E3. LOSS RATIO ZERO DIVISION
    In baseline/parse_flowmon.py:
    - loss_ratio = (txPackets - rxPackets) / txPackets
    - Must check txPackets > 0 before dividing
    - If txPackets == 0: loss_ratio = 0.0 (not NaN or exception)

E4. FLOWMON ITERATION
    In baseline/parse_flowmon.py:
    - Verify iteration uses root.iter("Flow") or finds the FlowStats element
    - Some ns-3 versions nest flows under <FlowStats>
    - Verify the XML tag name matches what ns-3 3.38 actually outputs:
      It should be "Flow" as child of "FlowStats"

---

### CATEGORY F — Plot and Evaluation Errors

F1. GENERATE_ALL.PY MISSING IMPORT
    In results/plots/generate_all.py:
    - Verify STEPS_PER_EP is imported from configs.hyperparams
    - Verify RAW_DIR is imported from configs.hyperparams
    - Check for any STEPS_PER_EP_PLACEHOLDER leftover from the prompt
      (should have been replaced with the actual import)

F2. FIGURE 5 PLACEHOLDER
    In results/plots/generate_all.py figure 5 (failure recovery):
    - The prompt included a simplified approximation using if/else
    - Verify this is clearly commented as an approximation
    - Verify it does not crash when XML files are missing
      (should plot zeros or skip gracefully)

F3. COMPARISON CSV INDEX
    In training/evaluate.py:
    - df_sc saved with .to_csv() must use index=True (algo name is the index)
    - In generate_all.py loading with pd.read_csv(..., index_col=0)
    - Mismatch here causes KeyError when accessing df_sc.loc[algo, metric]

F4. BAR CHART ALIGNMENT
    In results/plots/generate_all.py grouped bar charts:
    - x = np.arange(len(SCENS)) — 4 positions
    - width = 0.25 — must fit 3 bars without overlap
    - Bars at: x + 0*w, x + 1*w, x + 2*w
    - x-tick at: x + w (center of 3 bars = x + 1*w)
    - Verify set_xticks uses x + w, not x + 1.5*w or x

---

### CATEGORY G — Environment and Dependency Errors

G1. NS3GYM IMPORT
    In env/ns3_wrapper.py:
    - Verify: from ns3gym import ns3env  OR  import ns3gym
    - Verify the Ns3Env class is instantiated correctly:
      The startSim parameter should be False (user starts ns-3 manually)
      NOT True (auto-start would fail because ns-3 binary path is complex)

G2. GYMNASIUM VS GYM
    Check which library is used:
    - If "import gymnasium as gym" — step returns 5 values, reset returns 2
    - If "import gym" — step returns 4 values, reset returns 1
    - Verify ALL files using the env are consistent (all use same version)
    - Verify requirements match: pip install gymnasium OR pip install gym

G3. TORCH TENSOR TYPES IN TRAIN_STEP
    In agent/dqn_agent.py train_step():
    - S, NS must be: torch.LongTensor (for nn.Embedding)
      NOT: torch.FloatTensor
    - A must be: torch.LongTensor (for .gather())
    - R, D must be: torch.FloatTensor (for arithmetic)
    - Verify gather() call: q_all.gather(1, A.unsqueeze(1))
      A must be LongTensor here — gather requires Long index

G4. GRADIENT CLIPPING PARAMETER
    In agent/dqn_agent.py train_step():
    - If gradient clipping is used: nn.utils.clip_grad_norm_(params, max_norm=10.0)
    - max_norm=10.0 is appropriate for this problem
    - Verify clipping happens AFTER loss.backward() and BEFORE optim.step()
    - Wrong order (clipping before backward, or after step) has no effect

---

### CATEGORY H — Missing Files and Structural Issues

H1. Check for __init__.py files:
    The following directories need __init__.py to be importable as packages:
    - configs/__init__.py
    - agent/__init__.py
    - env/__init__.py
    - training/__init__.py
    - baseline/__init__.py
    If any are missing, cross-module imports (e.g., from agent.network import ...)
    will fail with ModuleNotFoundError.

H2. Check for missing files listed in context.md Section 14:
    List any files from the directory structure that do not exist.

H3. Check for duplicate function definitions:
    In agent/dqn_agent.py — verify train_step() is not defined twice
    In routing_env.cc — verify GetObservation() is not defined twice

H4. Check for circular imports:
    - configs/hyperparams.py must NOT import from agent/, env/, or training/
    - agent/network.py imports only from configs/ and torch
    - env/metrics.py imports only from configs/ and numpy
    - Training scripts may import from all others — that is fine

---

## STEP 3 — REPORT FORMAT

After completing all checks above, produce a report in this exact structure:

```
═══════════════════════════════════════════════════════
         FULL PROJECT CODE AUDIT REPORT
═══════════════════════════════════════════════════════

MISSING FILES
─────────────
List each missing file with its expected path.
If none: "All files present."

═══════════════════════════════════════════════════════
CATEGORY A — Critical Logic Errors
═══════════════════════════════════════════════════════

[A1] ARGMIN vs ARGMAX
  File: agent/network.py, line X
  Status: PASS / FAIL
  Issue (if FAIL): [exact description]
  Fix (if FAIL): [exact code change needed]

[A2] REWARD COMPUTATION — OBS SPLIT
  ...

[continue for each sub-item]

═══════════════════════════════════════════════════════
CATEGORY B — ns-3 C++ Specific Errors
═══════════════════════════════════════════════════════
[B1] through [B10] — same format

═══════════════════════════════════════════════════════
[continue all categories C through H]

═══════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════

CRITICAL ERRORS (project will not run):         X found
  [list each by item code e.g. A1, B3, C4]

CORRECTNESS ERRORS (wrong results silently):    X found
  [list each]

MINOR ERRORS (style, missing optimizations):    X found
  [list each]

TOTAL ISSUES: X
TOTAL PASS:   X out of Y checks
```

After the report, do NOT apply any fixes yet.
Wait for me to review the report and tell you which fixes to apply.
When I say "fix [item code]", apply only that specific fix.
When I say "fix all critical", fix all items listed under CRITICAL ERRORS.