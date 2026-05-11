# PHASE 1 PROMPTS — ns-3 Baseline Simulation (C++)
# Use these prompts sequentially in Cursor or your AI IDE.
# Always have context.md loaded before starting.
# Each prompt builds on the previous one. Do not skip steps.
# Run `./ns3 build` after every prompt and fix any compiler errors before continuing.

---

## HOW TO USE THESE PROMPTS

1. Open Cursor (or your AI IDE).
2. Open the folder `~/ns-3.38/scratch/drl_routing/` as your workspace.
3. Create a new file called `.cursorrules` in that folder and paste the
   contents of context.md into it. This gives the AI full project context.
4. Open `routing_sim.cc` as a blank file.
5. Run each prompt below IN ORDER inside the Cursor chat/composer.
6. After each prompt: read the generated code, verify it matches the
   expected output described, then run `./ns3 build` to confirm it compiles.
7. Fix any errors before moving to the next prompt.

---

## PROMPT 1.1 — File Header, Includes, Namespace

Paste this into Cursor chat exactly:

```
Create the file ~/ns-3.38/scratch/drl_routing/routing_sim.cc.

Write the opening section of an ns-3 3.38 C++ simulation file with:
1. A file header comment block that includes:
   - Project: DRL-Based Adaptive Network Routing
   - File: routing_sim.cc
   - Purpose: ns-3 baseline simulation with Dijkstra routing and FlowMonitor
   - Authors: Muhammad Sabeeh (23K-0002), Rayyan Merchant (23K-0073)
   - Note: This file is used for Phase 1 (baseline). Phase 2 adds ns3-gym hooks.

2. These exact #include directives in this order:
   #include <ns3/core-module.h>
   #include <ns3/network-module.h>
   #include <ns3/internet-module.h>
   #include <ns3/point-to-point-module.h>
   #include <ns3/applications-module.h>
   #include <ns3/flow-monitor-module.h>
   #include <ns3/ipv4-static-routing-helper.h>
   #include <ns3/ipv4-global-routing-helper.h>
   #include <sstream>
   #include <string>

3. The line: using namespace ns3;

4. The main() function signature:
   int main(int argc, char *argv[])

5. Inside main(), declare these CommandLine variables with defaults:
   double simTime      = 100.0;   // total simulation time in seconds
   bool   enableRL     = false;   // true when running with DRL agent
   bool   enableFail   = false;   // true for link failure scenario
   double failTime     = 40.0;    // time when R1-D1 link fails
   std::string scenario = "normal";  // normal / congested / failure / mixed
   std::string outFile  = "baseline.xml";  // FlowMonitor output filename
   uint32_t    seed     = 42;     // random seed for reproducibility
   uint32_t    runNum   = 1;      // run number for seed variation

6. Parse all of the above with CommandLine.

7. Set the random seed immediately after parsing:
   RngSeedManager::SetSeed(seed);
   RngSeedManager::SetRun(runNum);

8. Add a comment: // === TOPOLOGY SETUP STARTS HERE ===
   Do not write any topology code yet. Just the comment.

Do NOT close the main() function. More code will be added in the next prompt.
```

**Expected result**: A compilable file up to the comment line. Run `./ns3 build` — should succeed.

---

## PROMPT 1.2 — Node and Link Creation

Paste this into Cursor chat exactly:

```
Continue routing_sim.cc after the comment // === TOPOLOGY SETUP STARTS HERE ===

Write C++ ns-3 code to:

1. Create exactly 6 nodes using NodeContainer named 'nodes':
   nodes.Create(6);
   Add a comment above it: // Nodes: S1=0, S2=1, R1=2, R2=3, R3=4, D1=5

2. Define a struct called LinkDef inside main() with members:
   uint32_t u, v;
   const char* bw;
   const char* delay;

3. Declare an array of LinkDef named 'linkDefs' with exactly 10 entries
   in this EXACT order (order matters — index used later for failure):
   {0, 2, "10Mbps", "2ms"},    // index 0: S1-R1
   {0, 3, "7Mbps",  "4ms"},    // index 1: S1-R2
   {1, 3, "7Mbps",  "4ms"},    // index 2: S2-R2
   {1, 4, "10Mbps", "2ms"},    // index 3: S2-R3
   {2, 3, "5Mbps",  "5ms"},    // index 4: R1-R2 bottleneck
   {3, 4, "5Mbps",  "5ms"},    // index 5: R2-R3 bottleneck
   {2, 5, "8Mbps",  "8ms"},    // index 6: R1-D1 PRIMARY EXIT (fails in failure scenario)
   {3, 5, "6Mbps",  "6ms"},    // index 7: R2-D1
   {4, 5, "4Mbps",  "10ms"},   // index 8: R3-D1
   {2, 4, "3Mbps",  "12ms"},   // index 9: R1-R3 low-cap

4. Declare: NetDeviceContainer devs[10];

5. Write a PointToPointHelper named p2p. Then loop i from 0 to 9:
   - Set DataRate attribute from linkDefs[i].bw
   - Set Delay attribute from linkDefs[i].delay
   - Install on nodes.Get(linkDefs[i].u) and nodes.Get(linkDefs[i].v)
   - Store result in devs[i]

6. Add comment: // === INTERNET STACK AND IP ADDRESSING ===
   Do not write IP code yet.
```

**Expected result**: Node creation and all 10 links defined. `./ns3 build` should compile.

---

## PROMPT 1.3 — Internet Stack, IP Addressing, Routing

Paste this into Cursor chat exactly:

```
Continue routing_sim.cc after: // === INTERNET STACK AND IP ADDRESSING ===

Write C++ ns-3 code to:

1. Install the Internet stack on all nodes:
   InternetStackHelper inet;
   inet.Install(nodes);

2. Assign IP addresses to all 10 links using a loop.
   Use Ipv4AddressHelper named 'addr'.
   For each link index i from 0 to 9:
   - Build the subnet string: "10." + std::to_string(i) + ".1.0"
   - Do this using an std::ostringstream named 'subnet'
   - Call addr.SetBase(subnet.str().c_str(), "255.255.255.0")
   - Call addr.Assign(devs[i])
   - You do NOT need to store the returned Ipv4InterfaceContainer

3. After the loop, populate routing tables using Dijkstra:
   Ipv4GlobalRoutingHelper::PopulateRoutingTables();

4. Resolve D1's IP address DYNAMICALLY (never hardcode):
   Ipv4Address d1_ip = nodes.Get(5)->GetObject<Ipv4>()
                           ->GetAddress(1, 0).GetLocal();
   Add a comment explaining: D1 is node index 5, interface 1 is the first
   assigned interface (interface 0 is loopback).

5. Add comment: // === TRAFFIC GENERATION ===
   Do not write traffic code yet.

IMPORTANT: The dynamic IP resolution line MUST come after
Ipv4GlobalRoutingHelper::PopulateRoutingTables(), not before.
```

**Expected result**: Full internet stack, IPs, Dijkstra routing, D1 IP resolved. Compiles with `./ns3 build`.

---

## PROMPT 1.4 — Traffic Generation (Normal and Congested Scenarios)

Paste this into Cursor chat exactly:

```
Continue routing_sim.cc after: // === TRAFFIC GENERATION ===

Write C++ ns-3 code to:

1. Define the sink port: uint16_t port = 9;

2. Install a PacketSinkHelper on node 5 (D1):
   - Factory string: "ns3::UdpSocketFactory"
   - Bind address: InetSocketAddress(Ipv4Address::GetAny(), port)
   - Install on nodes.Get(5)
   - Application starts at t=0.0s and stops at t=simTime

3. Determine traffic rates based on the 'scenario' variable:
   Use an if-else block:
   - if scenario == "congested": s1Rate = "9Mbps", s2Rate = "8Mbps"
   - else (normal, failure, mixed all start with normal rate):
     s1Rate = "4Mbps", s2Rate = "3Mbps"
   Declare these as std::string variables.

4. Create an OnOffHelper for S1 → D1:
   - Factory: "ns3::UdpSocketFactory"
   - Remote address: InetSocketAddress(d1_ip, port)
   - DataRate: s1Rate
   - PacketSize: 1024 bytes
   - OnTime: "ns3::ConstantRandomVariable[Constant=1]"
   - OffTime: "ns3::ConstantRandomVariable[Constant=0]"
   - Install on nodes.Get(0) — this is S1
   - Start at t=1.0s, Stop at t=simTime-0.1s

5. Create a SECOND OnOffHelper for S2 → D1:
   - Same settings as above but DataRate = s2Rate
   - Install on nodes.Get(1) — this is S2
   - Same start/stop times

6. For the "mixed" scenario, add an additional simulation event:
   After t=30s: reduce traffic by scheduling a new lower-rate app
   For simplicity, just add a comment here:
   // TODO Phase 1 extension: implement mixed scenario using
   // Simulator::Schedule to change DataRate at t=30s and t=60s
   // For now, mixed uses the same rate as normal.

7. Add comment: // === LINK FAILURE SCENARIO ===
   Do not write failure code yet.

IMPORTANT: Use 'OnOffHelper onoff1' and 'OnOffHelper onoff2' as variable names.
Do not reuse the same helper variable for both flows.
```

**Expected result**: Two UDP flows installed. `./ns3 build` compiles.

---

## PROMPT 1.5 — Link Failure, FlowMonitor, Simulator Run

Paste this into Cursor chat exactly:

```
Continue routing_sim.cc after: // === LINK FAILURE SCENARIO ===

Write C++ ns-3 code to:

1. Link failure logic (only when enableFail == true):
   Write an if(enableFail) block that:
   a) Creates a Ptr<RateErrorModel> named 'failModel'
   b) Sets its ErrorRate attribute to DoubleValue(1.0) — drops ALL packets
   c) Sets its ErrorUnit to "ERROR_UNIT_PACKET" using StringValue
   d) Schedules a Simulator event at Seconds(failTime) that calls a lambda:
      [&]() {
          devs[6].Get(1)->SetAttribute("ReceiveErrorModel",
                                        PointerValue(failModel));
      }
      devs[6] is the R1-D1 link. Get(1) is the receiving device (D1 side).
   e) Add a comment: // devs[6] = R1-D1 link (index 6 in linkDefs array)

2. FlowMonitor setup:
   FlowMonitorHelper fmHelper;
   Ptr<FlowMonitor> flowMon = fmHelper.InstallAll();

3. Start and run the simulation:
   Simulator::Stop(Seconds(simTime));
   Simulator::Run();

4. After Simulator::Run(), serialize FlowMonitor results:
   flowMon->SerializeToXmlFile(outFile, true, true);
   Add comment: // true, true = enableHistograms, enableProbes

5. Destroy the simulator:
   Simulator::Destroy();

6. Print completion message:
   std::cout << "Simulation complete. Output: " << outFile << std::endl;

7. Return 0 and close the main() function with }.

The complete main() function is now done for Phase 1.
Make sure all braces are balanced. Do NOT add any ns3-gym code yet.
```

**Expected result**: Complete routing_sim.cc for Phase 1. `./ns3 build` must compile cleanly.

---

## PROMPT 1.6 — Build Verification and First Run

Paste this into Cursor chat exactly:

```
The routing_sim.cc file is now complete for Phase 1.
Help me verify it works correctly.

1. Write the exact shell commands to:
   a) Build the project: cd ~/ns-3.38 && ./ns3 build
   b) Run a normal traffic baseline (100s, no failure):
      ./ns3 run "drl_routing/routing_sim \
        --simTime=100 --enableRL=false --enableFail=false \
        --scenario=normal --output=normal.xml --seed=42 --runNum=1"
   c) Run a congested traffic scenario:
      ./ns3 run "drl_routing/routing_sim \
        --simTime=100 --enableRL=false --enableFail=false \
        --scenario=congested --output=congested.xml --seed=42 --runNum=1"
   d) Run a link failure scenario:
      ./ns3 run "drl_routing/routing_sim \
        --simTime=100 --enableRL=false --enableFail=true \
        --scenario=normal --output=failure.xml --seed=42 --runNum=1"

2. After running, the XML files appear in ~/ns-3.38/. Move them:
   mkdir -p ~/drl_project/results/raw
   mv normal.xml    ~/drl_project/results/raw/dijkstra_normal_run0.xml
   mv congested.xml ~/drl_project/results/raw/dijkstra_congested_run0.xml
   mv failure.xml   ~/drl_project/results/raw/dijkstra_failure_run0.xml

3. Verify the XML is non-empty:
   wc -l ~/drl_project/results/raw/dijkstra_normal_run0.xml
   grep -c "Flow flowId" ~/drl_project/results/raw/dijkstra_normal_run0.xml
   The second command should print 2 (one line per flow: S1→D1 and S2→D1).
```

**Expected result**: 3 XML files generated, each with 2 flows, non-empty.

---

## PROMPT 1.7 — FlowMonitor XML Parser (Python)

Switch context to the Python side. Open `~/drl_project/` in Cursor.

Paste this into Cursor chat exactly:

```
Create ~/drl_project/baseline/parse_flowmon.py

Write a Python module that parses ns-3 FlowMonitor XML output files.

Requirements:

1. Import: xml.etree.ElementTree as ET, pandas as pd, numpy as np, os, sys

2. Write a function parse_flowmon(xml_file, duration=99.0) that:
   a) Parses the XML file using ET.parse(xml_file).getroot()
   b) Iterates over all elements matching the tag "Flow"
   c) For each Flow element, extracts:
      - flow_id   : flow.get("flowId")
      - tx        : int(flow.get("txPackets", 0))
      - rx        : int(flow.get("rxPackets", 0))
      - loss_ratio: (tx - rx) / tx if tx > 0 else 0.0
      - delay_sum : the "delaySum" attribute — it looks like "12345678ns"
                    Strip the "ns" suffix, convert to float nanoseconds,
                    then convert to milliseconds: ns / 1e6
                    avg_delay_ms = delay_sum_ms / rx if rx > 0 else 0.0
      - rx_bytes  : int(flow.get("rxBytes", 0))
      - throughput: (rx_bytes * 8) / (duration * 1e6)  — result in Mbps
   d) Append a dict per flow to a list 'rows'
   e) Return pd.DataFrame(rows) with columns:
      [flow_id, throughput_mbps, avg_delay_ms, loss_ratio]

3. Write a function summarize(df) that returns a dict with keys:
   throughput, delay, loss — each being the mean across all flows.

4. Write a __main__ block:
   - Accept one command-line argument: the XML file path
   - Call parse_flowmon(sys.argv[1])
   - Print the DataFrame using df.to_string(index=False)
   - Print summarize(df)
   - Save the DataFrame to a CSV file with the same name as the XML
     but with .csv extension in results/logs/

5. Handle the edge case where delaySum is "0ns" or missing.

IMPORTANT: The delaySum field looks like "+123456789.0ns" — strip
everything after the numeric part using a regex or by splitting on 'ns'.
Use: float(flow.get("delaySum", "0ns").replace("ns", "").replace("+", ""))
```

**Expected result**: Parser that produces a DataFrame with correct columns.

---

## PROMPT 1.8 — Baseline Runner and Validation Script

Paste this into Cursor chat exactly:

```
Create ~/drl_project/baseline/run_baseline.py

Write a Python script that:

1. Imports: subprocess, os, sys, pandas as pd
   from baseline.parse_flowmon import parse_flowmon, summarize

2. Defines SCENARIOS = ["normal", "congested", "failure"]
   Defines N_RUNS = 3

3. Defines NS3_DIR = os.path.expanduser("~/ns-3.38")
   Defines OUT_DIR = os.path.join(os.path.dirname(__file__), "../results/raw")
   Creates OUT_DIR if it doesn't exist.

4. Defines a function run_ns3(scenario, run_id):
   - Builds the output filename: f"dijkstra_{scenario}_run{run_id}.xml"
   - Full output path: os.path.join(OUT_DIR, outname)
   - Builds the ns3 command list:
     ["./ns3", "run",
      f"drl_routing/routing_sim --simTime=100 --enableRL=false "
      f"--enableFail={'true' if scenario=='failure' else 'false'} "
      f"--scenario={scenario} "
      f"--output={out_path} "
      f"--seed=42 --runNum={run_id+1}"]
   - Runs it with subprocess.run(cmd, cwd=NS3_DIR, check=True)
   - Returns out_path

5. Defines run_all_baseline():
   For each scenario and run_id in range(N_RUNS):
     - Print: f"Running Dijkstra {scenario} run {run_id}..."
     - Call run_ns3(scenario, run_id)
     - Parse the resulting XML with parse_flowmon(out_path)
     - Append summarize(df) to a list for this scenario

   After all runs for a scenario:
     - Compute mean across N_RUNS runs using pd.DataFrame(run_summaries).mean()
     - Print results
     - Save to results/logs/dijkstra_{scenario}_summary.csv

6. Call run_all_baseline() in __main__ block.

7. Also create a function validate_metrics() that:
   - Checks normal.xml: throughput should be > 0 Mbps
   - Checks congested.xml: loss_ratio should be > 0 (congestion causes loss)
   - Checks failure.xml: loss_ratio should be > normal scenario's loss
   - Prints PASS or FAIL for each check with the actual values

Call validate_metrics() at the end of run_all_baseline() if all XMLs exist.
```

**Expected result**: Script that runs all baseline scenarios and validates metrics.

---

## PROMPT 1.9 — Metric Sanity Check (5 Validation Tests)

Paste this into Cursor chat exactly:

```
Create ~/drl_project/baseline/validate_metrics.py

Write a Python script with 5 explicit validation tests.
Each test modifies a simulation parameter, runs ns-3, parses
the output, and asserts the expected directional change.

Import: subprocess, os, pandas as pd, sys
from baseline.parse_flowmon import parse_flowmon, summarize

Define NS3_DIR and OUT_DIR as in run_baseline.py.

Define run_quick(label, extra_args, scenario="normal"):
  - Runs ns-3 with simTime=30 (shorter for speed)
  - Returns summarize(parse_flowmon(output_xml))

Tests to implement:

TEST 1: "BW Sensitivity"
  Normal: OnOff at 4 Mbps. Then run with 1 Mbps.
  Since we can't change rate via cmdline yet, skip this — just document it.
  Print: "[TEST 1] BW Sensitivity: Run manually — change DataRate in routing_sim.cc"

TEST 2: "Congestion Increases Loss"
  Run normal scenario (4 Mbps) → get loss_ratio_normal
  Run congested scenario (9 Mbps) → get loss_ratio_congested
  Assert: loss_ratio_congested > loss_ratio_normal
  Print PASS or FAIL with actual values.

TEST 3: "Congestion Increases Delay"
  Use same runs as TEST 2.
  Assert: avg_delay_ms_congested > avg_delay_ms_normal
  Print PASS or FAIL with actual values.

TEST 4: "Link Failure Reduces Throughput During Failure Window"
  Run failure scenario → get throughput_failure
  Run normal scenario → get throughput_normal
  Assert: throughput_failure < throughput_normal
  Print PASS or FAIL.

TEST 5: "FlowMonitor Has Exactly 2 Flows"
  Parse normal.xml → count rows in DataFrame
  Assert: len(df) == 2 (one flow for S1→D1, one for S2→D1)
  Print PASS or FAIL with actual count.

Print a final summary: X/5 tests passed.
Exit with code 1 if any test fails, 0 if all pass.

Run the tests using XMLs already generated by run_baseline.py.
Do not re-run ns-3 unless the XML files don't exist.
```

**PHASE 1 COMPLETE when**: All 5 validation tests print PASS.

---

## PHASE 1 COMPLETION CHECKLIST

Before moving to Phase 2, verify manually:

```
[ ] ./ns3 build completes with 0 errors and 0 warnings (warnings OK)
[ ] routing_sim.cc is ~180-220 lines long
[ ] dijkstra_normal_run0.xml exists and has 2 <Flow> elements
[ ] dijkstra_congested_run0.xml exists and has higher loss than normal
[ ] dijkstra_failure_run0.xml exists and has lower throughput than normal
[ ] parse_flowmon.py returns a DataFrame with 2 rows and correct columns
[ ] All 5 validation tests print PASS
[ ] No hardcoded IP addresses anywhere in routing_sim.cc
[ ] RngSeedManager::SetSeed(42) is called in main()
```

If any item fails, fix it before starting Phase 2. The Phase 2 prompts
build directly on the routing_sim.cc structure created here.