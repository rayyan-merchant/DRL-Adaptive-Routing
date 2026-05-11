# PHASE 2 PROMPTS — ns3-gym Integration (C++ Environment + Python Wrapper + Reward)
# Use these prompts sequentially after Phase 1 is fully complete and validated.
# Always have context.md loaded before starting.
# Phase 2 touches BOTH the C++ side (new files) and the Python side (new files).
# Run `./ns3 build` after every C++ prompt.

---

## PHASE 2 OVERVIEW

Phase 2 converts the static ns-3 simulation into a Gym-compatible RL environment.

Three things are built in sequence:
1. C++ OpenGymEnv subclass (routing_env.h + routing_env.cc)
2. Modified routing_sim.cc with ns3-gym hooks
3. Python side: path precomputation, reward function, ns3 wrapper

The communication works like this:
```
[Python training script]
    ↕  Gym API (reset, step, obs, done)
[env/ns3_wrapper.py]
    ↕  ZeroMQ socket port 5555
[routing_sim.cc + routing_env.cc]
    ↕  Simulator::Schedule loop
[ns-3 topology + FlowMonitor]
```

---

## PROMPT 2.1 — routing_env.h Header File

Open `~/ns-3.38/scratch/drl_routing/` in Cursor.

Paste this into Cursor chat exactly:

```
Create ~/ns-3.38/scratch/drl_routing/routing_env.h

Write a complete C++ header file for a custom ns-3 OpenGymEnv subclass.

Requirements:

1. Include guard: #ifndef ROUTING_ENV_H / #define ROUTING_ENV_H / #endif

2. Includes needed:
   #include "ns3/opengym-module.h"
   #include "ns3/node-container.h"
   #include "ns3/net-device-container.h"
   #include "ns3/ipv4-address.h"
   #include "ns3/flow-monitor-helper.h"
   #include "ns3/flow-monitor.h"
   #include <vector>
   #include <map>
   #include <string>

3. using namespace ns3;

4. Declare class RoutingEnv that inherits publicly from OpenGymEnv.

5. Public section:
   - TypeId static method: static TypeId GetTypeId();
   - Constructor: RoutingEnv(NodeContainer nodes,
                             NetDeviceContainer devs[],
                             Ipv4Address d1ip,
                             Ptr<FlowMonitor> flowmon,
                             double tmon,
                             double simTime,
                             uint32_t kPaths);
   - Destructor: virtual ~RoutingEnv();

   - These five override methods (all pure virtual in OpenGymEnv):
     Ptr<OpenGymSpace> GetObservationSpace() override;
     Ptr<OpenGymSpace> GetActionSpace() override;
     Ptr<OpenGymDataContainer> GetObservation() override;
     float GetReward() override;
     bool GetGameOver() override;
     std::string GetExtraInfo() override;
     bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

   - Public method to trigger the next step:
     void ScheduleNextStep();

   - Public method called by ExecuteActions:
     void UpdateRouting(uint32_t pathIdx);

6. Private section:
   - Member variables:
     NodeContainer   m_nodes;
     NetDeviceContainer m_devs[10];  // all 10 P2P link device containers
     Ipv4Address     m_d1ip;
     Ptr<FlowMonitor> m_flowmon;
     double          m_tmon;         // step interval in sim-seconds
     double          m_simTime;      // total sim time
     uint32_t        m_kPaths;       // number of candidate paths (3)
     uint32_t        m_currentSD;    // current SD pair being routed: 0 or 1
     uint32_t        m_stepCount;    // counts steps taken in this episode
     bool            m_gameOver;     // true when sim time reached

   - Private helper:
     std::vector<float> ComputePathMetrics();
     // Returns 9 floats: [bw0,del0,loss0, bw1,del1,loss1, bw2,del2,loss2]
     // for the current SD pair (m_currentSD)

   - Private: the precomputed paths stored as 2D array
     // Outer: SD pair index (0 or 1)
     // Middle: path index (0, 1, 2)
     // Inner: sequence of node indices on the path
     std::vector< std::vector< std::vector<uint32_t> > > m_paths;

     - Initialise m_paths in constructor:
       SD 0 (S1→D1): path0=[0,2,5], path1=[0,3,5], path2=[0,2,3,5]
       SD 1 (S2→D1): path0=[1,4,5], path1=[1,3,5], path2=[1,3,2,5]
```

**Expected result**: A compilable header file. `./ns3 build` may warn about missing .cc — that is fine.

---

## PROMPT 2.2 — routing_env.cc Part A (Constructor and Space Methods)

Paste this into Cursor chat exactly:

```
Create ~/ns-3.38/scratch/drl_routing/routing_env.cc — Part A.

Write the first half of the implementation of RoutingEnv.

1. Includes:
   #include "routing_env.h"
   #include "ns3/log.h"
   #include "ns3/ipv4-static-routing-helper.h"
   #include "ns3/ipv4-routing-table-entry.h"
   #include "ns3/simulator.h"
   #include "ns3/opengym-module.h"
   #include <algorithm>
   #include <numeric>

2. NS_LOG_COMPONENT_DEFINE("RoutingEnv");

3. TypeId RoutingEnv::GetTypeId():
   Return a TypeId named "RoutingEnv" with base OpenGymEnv and no attributes.

4. Constructor implementation:
   RoutingEnv::RoutingEnv(NodeContainer nodes,
                          NetDeviceContainer devs[],
                          Ipv4Address d1ip,
                          Ptr<FlowMonitor> flowmon,
                          double tmon,
                          double simTime,
                          uint32_t kPaths)
   - Initialise all member variables from parameters
   - Copy all 10 devs[] entries into m_devs[10] with a loop
   - Set m_currentSD = 0
   - Set m_stepCount = 0
   - Set m_gameOver = false
   - Initialise m_paths with the static path definitions:
     SD 0 paths: {0,2,5}, {0,3,5}, {0,2,3,5}
     SD 1 paths: {1,4,5}, {1,3,5}, {1,3,2,5}

5. Destructor: default empty body.

6. GetObservationSpace():
   - Create a Box space with:
     low  = 0.0 (std::vector<float> of size 10, all zeros)
     high = 1.0 (std::vector<float> of size 10, all ones)
     shape = {10} (std::vector<uint32_t>)
     dtype = "float32"
   - Use: OpenGymBoxSpace(low_vec, high_vec, shape_vec, dtype_str)
   - Return Ptr<OpenGymSpace> to it

7. GetActionSpace():
   - Create a Discrete space with n = m_kPaths (= 3)
   - Use: OpenGymDiscreteSpace(n)
   - Return Ptr<OpenGymSpace> to it

Both methods should match the observation_space and action_space
defined in context.md Section 5.
```

**Expected result**: Constructor and space methods compile. `./ns3 build`.

---

## PROMPT 2.3 — routing_env.cc Part B (GetObservation and ComputePathMetrics)

Paste this into Cursor chat exactly:

```
Continue routing_env.cc — Part B.

Write these two methods for RoutingEnv:

1. ComputePathMetrics() — private helper, returns std::vector<float> of 9 values.

   This method reads current per-link metrics from FlowMonitor and aggregates
   them to path-level metrics for each of the k=3 candidate paths for the
   current SD pair (m_currentSD).

   a) Call m_flowmon->CheckForLostPackets() to update statistics.

   b) Get FlowMonitor statistics map:
      auto stats = m_flowmon->GetFlowStats();

   c) Build per-link metrics. For each link index i from 0 to 9:
      Compute available bandwidth and delay from the stats map.
      Since FlowMonitor doesn't give per-link stats directly,
      approximate as follows:
        - For each flow in stats:
            double tx_bytes = (double)it->second.txBytes;
            double rx_bytes = (double)it->second.rxBytes;
            double elapsed  = Simulator::Now().GetSeconds();
            double throughput_bps = (elapsed > 0) ? (tx_bytes * 8.0 / elapsed) : 0.0;
        - Store a simple map: link_load[i] = throughput_bps / link_capacity_bps[i]
          where link_capacity_bps is the bandwidth from linkDefs (hardcode the
          capacities in Mbps from context.md: {10,7,7,10,5,5,8,6,4,3}).
          Clamp to [0.0, 1.0].
        - available_bw_mbps[i] = (1.0 - link_load[i]) * link_capacity_mbps[i]
        - delay_ms[i] = fixed propagation delay from linkDefs:
          {2,4,4,2,5,5,8,6,10,12} (in ms)
        - loss_ratio[i] = 0.0 for now (set to link_load[i] * 0.1 as a proxy)

   d) For each path p in [0, 1, 2] for current SD pair m_currentSD:
      Get path nodes from m_paths[m_currentSD][p].
      Build the list of link indices on that path:
        For consecutive node pairs (u, v) in the path, find the link index
        by checking which devs[i] connects u and v.
        Do this with a nested loop over linkDefs.
      Compute path-level metrics using DRSIR aggregation:
        bw_path   = min of available_bw_mbps[link] for links on path
        delay_path = sum of delay_ms[link] for links on path
        loss_path  = 1.0 - product of (1.0 - loss_ratio[link]) for links on path

   e) Return vector of 9 floats: [bw0, delay0, loss0, bw1, delay1, loss1, bw2, delay2, loss2]
      Normalize each metric to [0, 1] range using min-max across the 3 paths:
        For bw: normalize so higher BW = higher value (inverted later in Python reward)
        For delay and loss: normalize so lower = better (lower value)
        Use: (val - min) / (max - min + 1e-6)
        If all 3 values equal, set all to 0.5.

2. GetObservation() method:
   a) Call std::vector<float> metrics = ComputePathMetrics();
   b) Build obs vector of 10 floats:
      obs[0]   = (float)m_currentSD
      obs[1..9] = metrics[0..8]
   c) Create Ptr<OpenGymBoxContainer<float>> container:
      auto box = CreateObject<OpenGymBoxContainer<float>>();
      box->SetShape({10});
      for each value in obs: box->AddValue(v);
   d) Return box (as Ptr<OpenGymDataContainer>)

3. GetReward():
   // Reward is computed Python-side from the observation vector.
   // C++ always returns 0.0.
   return 0.0f;

4. GetGameOver():
   m_gameOver = (Simulator::Now().GetSeconds() >= m_simTime - 0.1);
   return m_gameOver;

5. GetExtraInfo():
   std::ostringstream ss;
   ss << "step=" << m_stepCount << ",sd=" << m_currentSD;
   return ss.str();
```

**Expected result**: Observation and reward methods complete. `./ns3 build`.

---

## PROMPT 2.4 — routing_env.cc Part C (ExecuteActions and UpdateRouting)

Paste this into Cursor chat exactly:

```
Continue routing_env.cc — Part C.

Write these two methods:

1. ExecuteActions(Ptr<OpenGymDataContainer> action):
   a) Cast the action container:
      Ptr<OpenGymDiscreteContainer> disc =
          DynamicCast<OpenGymDiscreteContainer>(action);
      uint32_t pathIdx = disc->GetValue();
   b) Clamp pathIdx: if (pathIdx >= m_kPaths) pathIdx = 0;
   c) Call UpdateRouting(pathIdx);
   d) Alternate SD pair for next step:
      m_currentSD = (m_currentSD + 1) % 2;
      // We alternate between routing S1→D1 and S2→D1 each step
   e) Increment m_stepCount++
   f) Return true (action executed successfully)

2. UpdateRouting(uint32_t pathIdx):
   This method installs the chosen path into the static routing tables.

   a) Get the selected path:
      auto& path = m_paths[m_currentSD][pathIdx];
      // path is a vector of node indices, e.g., [0, 2, 5] for S1-R1-D1

   b) Determine source node (SD pair 0 = node 0, SD pair 1 = node 1):
      uint32_t srcNode = m_currentSD; // 0 for S1, 1 for S2
      uint32_t dstNode = 5;           // always D1

   c) For each intermediate hop in the path (from src to one before dst):
      Loop i from 0 to path.size()-2:
        uint32_t curNode  = path[i];
        uint32_t nextNode = path[i+1];

        // Get Ipv4StaticRouting for curNode
        Ptr<Ipv4StaticRouting> staticRouting =
            Ipv4RoutingHelper::GetRouting<Ipv4StaticRouting>(
                m_nodes.Get(curNode)->GetObject<Ipv4>()->GetRoutingProtocol());

        // Find the interface index on curNode that connects to nextNode
        // by finding which devs[j] connects curNode and nextNode
        uint32_t iface = FindInterface(curNode, nextNode);

        // Get D1's IP from m_d1ip
        // Add a host route: D1's IP via interface iface, metric 1
        // Remove old route first to avoid duplicates:
        // Iterate existing routes and remove any for m_d1ip destination
        RemoveHostRoute(staticRouting, m_d1ip);

        // Add new route
        staticRouting->AddHostRouteTo(m_d1ip, iface, 1);

        break; // Only set the next-hop from the source node
               // (the intermediate routers keep global routing)

   Note: We only update the routing decision at the SOURCE node (S1 or S2).
   The intermediate routers (R1, R2, R3) use global routing for forwarding.
   This is a simplification that works for this topology.

3. Write two private helper methods:

   uint32_t FindInterface(uint32_t fromNode, uint32_t toNode):
   - Loops over link indices 0-9 in the linkDefs equivalent
   - Hardcode the link endpoints as a static array inside this method:
     static uint32_t linkU[] = {0,0,1,1,2,3,2,3,4,2};
     static uint32_t linkV[] = {2,3,3,4,3,4,5,5,5,4};
   - For link i: if (linkU[i]==fromNode && linkV[i]==toNode) or vice versa:
       // Get the device index on fromNode's side
       // devs[i].Get(0) is the fromNode device if linkU[i]==fromNode
       // devs[i].Get(1) is the fromNode device if linkV[i]==fromNode
       Ptr<NetDevice> dev = ...;
       return m_nodes.Get(fromNode)->GetObject<Ipv4>()
                     ->GetInterfaceForDevice(dev);
   - Return 1 as fallback if not found (shouldn't happen)

   void RemoveHostRoute(Ptr<Ipv4StaticRouting> sr, Ipv4Address dest):
   - Iterate all routes in sr using sr->GetNRoutes()
   - For each route index i: sr->GetRoute(i)
   - If route destination matches dest: sr->RemoveRoute(i); return;
   - (Remove only the first match)

4. ScheduleNextStep():
   Simulator::Schedule(Seconds(m_tmon),
                       &RoutingEnv::NotifySimulationEnd, this);
   // Or use the opengym Notify method:
   Notify();
   // Actually: call this->Notify() which triggers the ZMQ exchange
```

**Expected result**: Complete routing_env.cc. `./ns3 build` should compile.

---

## PROMPT 2.5 — Modify routing_sim.cc to Add ns3-gym Hooks

Paste this into Cursor chat exactly:

```
Modify ~/ns-3.38/scratch/drl_routing/routing_sim.cc to add ns3-gym support
when the --enableRL=true flag is set.

The file currently has Phase 1 code. Add ns3-gym support without breaking
the existing Dijkstra/FlowMonitor baseline code.

Changes to make:

1. Add these includes at the top (after existing includes):
   #include "routing_env.h"
   #include "ns3/opengym-module.h"

2. After the FlowMonitor setup line and BEFORE Simulator::Stop():
   Add an if(enableRL) block that:

   a) Creates the custom environment:
      Ptr<RoutingEnv> routingEnv = CreateObject<RoutingEnv>(
          nodes, devs, d1_ip, flowMon,
          5.0,     // tmon: 5 simulated seconds per step
          simTime, // total simulation time
          3        // kPaths
      );

   b) Creates the OpenGymInterface on port 5555:
      Ptr<OpenGymInterface> openGym =
          CreateObject<OpenGymInterface>(5555);
      openGym->SetGetActionSpaceCb(
          MakeCallback(&RoutingEnv::GetActionSpace, routingEnv));
      openGym->SetGetObservationSpaceCb(
          MakeCallback(&RoutingEnv::GetObservationSpace, routingEnv));
      openGym->SetGetObservationCb(
          MakeCallback(&RoutingEnv::GetObservation, routingEnv));
      openGym->SetGetRewardCb(
          MakeCallback(&RoutingEnv::GetReward, routingEnv));
      openGym->SetGetGameOverCb(
          MakeCallback(&RoutingEnv::GetGameOver, routingEnv));
      openGym->SetGetExtraInfoCb(
          MakeCallback(&RoutingEnv::GetExtraInfo, routingEnv));
      openGym->SetExecuteActionsCb(
          MakeCallback(&RoutingEnv::ExecuteActions, routingEnv));

   c) Trigger the first observation:
      openGym->NotifyCurrentState();

   d) Schedule the first step:
      Simulator::Schedule(Seconds(5.0),
          &RoutingEnv::ScheduleNextStep, routingEnv);

3. The existing Simulator::Stop() and Simulator::Run() lines stay unchanged.

4. After Simulator::Destroy(), add:
   if(enableRL) {
       openGym->NotifySimulationEnd();
   }

IMPORTANT: The existing Dijkstra baseline code (traffic, FlowMonitor, etc.)
must not be changed. The RL hooks are ADDITIVE, inside the if(enableRL) block.
```

**Expected result**: routing_sim.cc now supports both baseline (enableRL=false) and RL (enableRL=true) modes. `./ns3 build`.

---

## PROMPT 2.6 — Python: Path Precomputation

Switch to Python side. Open `~/drl_project/` in Cursor.

Paste this into Cursor chat exactly:

```
Create ~/drl_project/env/path_precompute.py

Write a Python module that uses NetworkX to precompute the k=3 candidate
paths for each SD pair. This runs ONCE at startup and is imported by other modules.

Requirements:

1. Imports: networkx as nx, sys, os
   from configs.hyperparams import K_PATHS

2. Function build_graph():
   Creates and returns a NetworkX undirected Graph G with 6 nodes (0-5).
   Add edges with the EXACT attributes from context.md Section 3:
   G.add_edge(0, 2, bw=10, delay=2,  capacity_bps=10e6)
   G.add_edge(0, 3, bw=7,  delay=4,  capacity_bps=7e6)
   G.add_edge(1, 3, bw=7,  delay=4,  capacity_bps=7e6)
   G.add_edge(1, 4, bw=10, delay=2,  capacity_bps=10e6)
   G.add_edge(2, 3, bw=5,  delay=5,  capacity_bps=5e6)
   G.add_edge(3, 4, bw=5,  delay=5,  capacity_bps=5e6)
   G.add_edge(2, 5, bw=8,  delay=8,  capacity_bps=8e6)
   G.add_edge(3, 5, bw=6,  delay=6,  capacity_bps=6e6)
   G.add_edge(4, 5, bw=4,  delay=10, capacity_bps=4e6)
   G.add_edge(2, 4, bw=3,  delay=12, capacity_bps=3e6)

3. Function get_k_paths(G, source, target, k=K_PATHS):
   Uses nx.shortest_simple_paths(G, source, target, weight='delay')
   Returns a list of the first k paths, each path being a list of node indices.
   Each path is converted to a plain list (not generator).
   Handle StopIteration gracefully — if fewer than k paths exist, return what's available.

4. Module-level constants (computed once on import):
   G = build_graph()

   PATHS = {
       (0, 5): get_k_paths(G, 0, 5),   # S1 → D1
       (1, 5): get_k_paths(G, 1, 5),   # S2 → D1
   }

   # SD_PAIRS maps SD pair index → (source_node, dest_node)
   SD_PAIRS = {
       0: (0, 5),   # index 0 = S1→D1
       1: (1, 5),   # index 1 = S2→D1
   }

5. Function get_path(sd_idx, action):
   Returns PATHS[SD_PAIRS[sd_idx]][action]
   This is the path selected by the agent.

6. Function get_link_indices_on_path(path):
   Given a path as a list of node indices (e.g., [0, 2, 5]):
   Returns a list of link indices (into the devs[] array) for each edge.
   Use the LINK_ENDPOINTS constant defined below:
   LINK_ENDPOINTS = [
       (0,2),(0,3),(1,3),(1,4),(2,3),(3,4),(2,5),(3,5),(4,5),(2,4)
   ]
   For each consecutive pair (u, v) in path, find the index i where
   LINK_ENDPOINTS[i] == (u,v) or (v,u).

7. __main__ block: print PATHS nicely for verification.
   Expected output example:
   SD (0,5) S1->D1:
     Action 0: [0, 2, 5]   # S1-R1-D1
     Action 1: [0, 3, 5]   # S1-R2-D1
     Action 2: [0, 2, 3, 5] # S1-R1-R2-D1
```

**Expected result**: Run `python3 env/path_precompute.py` and verify the output matches the expected paths in context.md.

---

## PROMPT 2.7 — Python: Reward Function (DRSIR Eq. 5)

Paste this into Cursor chat exactly:

```
Create ~/drl_project/env/metrics.py

Write the DRSIR reward computation module.

Requirements:

1. Imports: numpy as np
   from configs.hyperparams import K_PATHS, BETA

2. Function extract_path_metrics(obs, k=K_PATHS):
   Input: obs — numpy array of shape (10,)
   Output: numpy array of shape (k, 3) — columns are [bw, delay, loss]

   obs[0] is SD pair index (ignored here).
   obs[1:] has k*3 = 9 values: [bw0, delay0, loss0, bw1, delay1, loss1, ...]
   Reshape obs[1:] to shape (k, 3) and return it.

3. Function minmax_normalize(path_metrics):
   Input: numpy array of shape (k, 3)
   Output: numpy array of shape (k, 3) with each column normalized to [0,1]

   For each column c in range(3):
     col = path_metrics[:, c]
     vmin, vmax = col.min(), col.max()
     if (vmax - vmin) < 1e-8:
         normalized[:, c] = 0.5    # all paths equal on this metric
     else:
         normalized[:, c] = (col - vmin) / (vmax - vmin)
   Return normalized array.

   Note: This normalization is applied ACROSS the k paths at each decision step,
   not across episodes or time. It is recomputed fresh each call.

4. Function compute_reward(obs, action, k=K_PATHS, beta=BETA):
   """
   Compute the DRSIR cost for the chosen action.
   The agent MINIMIZES this value.

   Args:
       obs    : numpy array (10,) — full observation from ns-3
       action : int in [0, k-1] — path chosen by agent
       k      : number of candidate paths (default K_PATHS=3)
       beta   : tuple (beta1, beta2, beta3) for reward weighting

   Returns:
       float — the DRSIR cost R (lower = better path chosen)
   """
   path_metrics = extract_path_metrics(obs, k)          # shape (k, 3)
   norm         = minmax_normalize(path_metrics)         # shape (k, 3)

   bw_n    = norm[action, 0]    # normalized bandwidth for chosen path
   delay_n = norm[action, 1]    # normalized delay for chosen path
   loss_n  = norm[action, 2]    # normalized loss for chosen path

   # DRSIR Equation 5:
   # R = beta1 * (1/bw_norm) + beta2 * delay_norm + beta3 * loss_norm
   # Inversely proportional to BW (more BW = lower cost)
   # Directly proportional to delay and loss (higher = higher cost)
   reward = beta[0] * (1.0 / (bw_n + 1e-6)) + \
            beta[1] * delay_n + \
            beta[2] * loss_n

   return float(reward)

5. Function batch_rewards(obs_batch, actions_batch, k=K_PATHS, beta=BETA):
   Vectorized version for computing rewards over a batch.
   Input: obs_batch shape (N, 10), actions_batch shape (N,)
   Output: numpy array shape (N,) of reward values.
   Loop over batch and call compute_reward for each.

6. Sanity test in __main__:
   # Test with known values
   obs_test = np.array([0.0,  # SD pair 0
                         1.0, 0.1, 0.0,   # path0: high BW, low delay, no loss
                         0.3, 0.5, 0.2,   # path1: medium BW, medium delay
                         0.1, 0.9, 0.8])  # path2: low BW, high delay, high loss
   r0 = compute_reward(obs_test, action=0)  # should be lowest cost
   r2 = compute_reward(obs_test, action=2)  # should be highest cost
   assert r0 < r2, f"Expected r0 < r2, got r0={r0:.4f} r2={r2:.4f}"
   print(f"Sanity check PASSED: r0={r0:.4f} < r2={r2:.4f}")
```

**Expected result**: `python3 env/metrics.py` prints "Sanity check PASSED".

---

## PROMPT 2.8 — Python: ns3 Environment Wrapper

Paste this into Cursor chat exactly:

```
Create ~/drl_project/env/ns3_wrapper.py

Write the Python-side Gym environment wrapper for ns3-gym.

Requirements:

1. Imports:
   import numpy as np
   import gymnasium as gym
   from gymnasium import spaces
   try:
       from ns3gym import ns3env
   except ImportError:
       raise ImportError("ns3gym not installed. Run: pip install ns3gym")
   from configs.hyperparams import K_PATHS, OBS_SIZE, ZMQ_PORT, T_MON

2. Class NS3RoutingEnv(gym.Env):

   Attributes:
   - observation_space = spaces.Box(low=0.0, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
     Note: use np.inf as high because raw metric values are not bounded to [0,1]
     The normalization is done in metrics.py, not here.
   - action_space = spaces.Discrete(K_PATHS)

   __init__(self, port=ZMQ_PORT, sim_seed=42, debug=False):
   - Store port, sim_seed, debug
   - Set self.env = None  (will be set in reset())
   - Set self._obs = np.zeros(OBS_SIZE, dtype=np.float32)

   reset(self, seed=None, options=None):
   - If self.env is not None: self.env.close()
   - Create a new ns3env.Ns3Env:
     self.env = ns3env.Ns3Env(
         port=self.port,
         stepTime=T_MON,
         startSim=False,    # ns-3 sim must be started manually in Terminal 1
         simSeed=self.sim_seed,
         simArgs={},
         debug=self.debug
     )
   - obs = self.env.reset()
   - If obs is None: raise RuntimeError("ns-3 environment reset returned None")
   - self._obs = np.array(obs, dtype=np.float32)
   - Return self._obs, {}   # gymnasium API returns (obs, info)

   step(self, action):
   - Assert action in [0, K_PATHS-1]
   - Call: obs, reward, done, info = self.env.step(int(action))
   - Note: reward from ns-3 is always 0.0 — we ignore it
   - self._obs = np.array(obs, dtype=np.float32) if obs is not None else self._obs
   - Return: self._obs, 0.0, bool(done), False, {}
     (obs, reward=0.0, terminated, truncated, info)
     The real reward is computed by the training script using metrics.py

   close(self):
   - If self.env is not None: self.env.close()

   @property
   def current_obs(self): return self._obs

3. Add a __main__ block for random action testing:
   """Quick smoke test — run ns-3 in Terminal 1 first:
   cd ~/ns-3.38 && ./ns3 run 'drl_routing/routing_sim --enableRL=true --simTime=100'
   Then run this script in Terminal 2."""
   import random
   env = NS3RoutingEnv(port=ZMQ_PORT)
   obs, _ = env.reset()
   print(f"Initial obs: {obs}")
   for step in range(20):
       action = random.randint(0, K_PATHS-1)
       obs, _, done, _, _ = env.step(action)
       print(f"Step {step:2d} | action={action} | obs[0]={obs[0]:.1f} | done={done}")
       if done: break
   env.close()
   print("Smoke test complete.")
```

**Expected result**: Wrapper class with correct Gym API. Smoke test runs when ns-3 is active in Terminal 1.

---

## PROMPT 2.9 — Integration Test: Random Action Loop

Paste this into Cursor chat exactly:

```
Create ~/drl_project/env/test_random_actions.py

Write a complete integration test script that:

1. Imports:
   import random, numpy as np, time
   from env.ns3_wrapper import NS3RoutingEnv
   from env.metrics import compute_reward
   from configs.hyperparams import K_PATHS, N_EPISODES

2. Defines N_TEST_EPISODES = 5  (short test)
   Defines STEPS_PER_EP = 20

3. Prints instructions at the start:
   print("="*60)
   print("INTEGRATION TEST: Random Action Loop")
   print("Make sure ns-3 is running in Terminal 1:")
   print("  cd ~/ns-3.38")
   print("  ./ns3 run 'drl_routing/routing_sim --enableRL=true --simTime=100'")
   print("="*60)
   input("Press Enter when ns-3 is running...")

4. Creates NS3RoutingEnv and runs N_TEST_EPISODES episodes:
   env = NS3RoutingEnv()
   reward_history = []
   reward_per_action = {0: [], 1: [], 2: []}

   For each episode:
     obs, _ = env.reset()
     ep_reward = 0.0
     for step in range(STEPS_PER_EP):
       action = random.randint(0, K_PATHS-1)
       next_obs, _, done, _, _ = env.step(action)
       reward = compute_reward(obs, action)
       ep_reward += reward
       reward_per_action[action].append(reward)
       obs = next_obs
       if done: break
     reward_history.append(ep_reward)
     print(f"Episode {ep+1}/{N_TEST_EPISODES} | Total cost: {ep_reward:.4f}")

5. After all episodes, print:
   - Mean reward per action (should differ — proves env is responsive)
   - "TEST PASSED" if all episodes completed without exception
   - "TEST FAILED" otherwise

6. Assertions (gates):
   assert len(reward_history) == N_TEST_EPISODES, "Not all episodes completed"
   rewards_flat = [r for v in reward_per_action.values() for r in v]
   assert max(rewards_flat) != min(rewards_flat), \
       "All rewards identical — environment is not responsive"
   print("GATE: Rewards vary across actions — environment is working correctly.")

7. env.close()

IMPORTANT: This script must be run MANUALLY with ns-3 in Terminal 1.
Do not run it automatically as part of any training pipeline.
```

**Expected result**: All episodes complete, rewards vary. This is the Phase 2 gate condition.

---

## PHASE 2 COMPLETION CHECKLIST

Before moving to Phase 3, verify manually:

```
[ ] routing_env.h compiles without errors
[ ] routing_env.cc compiles without errors
[ ] routing_sim.cc --enableRL=true starts and waits on ZMQ port 5555
[ ] python3 env/path_precompute.py prints correct paths from context.md
[ ] python3 env/metrics.py prints "Sanity check PASSED"
[ ] Integration test (test_random_actions.py) prints "TEST PASSED"
[ ] Rewards differ between actions (not all identical)
[ ] ns-3 routing table visibly changes — verify with:
    NS_LOG="Ipv4StaticRouting=info" ./ns3 run "routing_sim --enableRL=true ..."
    Different actions should produce different routing log entries
[ ] No hardcoded IPs in routing_env.cc or routing_sim.cc
[ ] GetReward() in C++ always returns 0.0f
[ ] Reward computation is entirely in Python (env/metrics.py)
```