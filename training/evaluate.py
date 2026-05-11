import subprocess
import os
import numpy as np
import pandas as pd
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.parse_flowmon import parse_flowmon, summarize
from configs.hyperparams import (
    LOGS_DIR, RAW_DIR, CKPT_DIR, PROJECT_ROOT
)

NS3_DIR   = os.path.expanduser("~/projects/ns-3-dev")
SCENARIOS = ["normal", "congested", "failure", "mixed"]
ALGORITHMS= ["dijkstra", "dqn", "ddqn"]
N_RUNS    = 3   # average over 3 runs for statistical reliability

SCENARIO_ARGS = {
    "normal":    {"enableFail": "false", "scenario": "normal"},
    "congested": {"enableFail": "false", "scenario": "congested"},
    "failure":   {"enableFail": "true",  "scenario": "normal"},
    "mixed":     {"enableFail": "false", "scenario": "mixed"},
}

def run_dijkstra_scenario(scenario, run_id):
    """Run a single Dijkstra (baseline) scenario and return the output XML path."""
    out_path = os.path.join(RAW_DIR, f"dijkstra_{scenario}_run{run_id}.xml")
    args = SCENARIO_ARGS[scenario]
    
    cmd = [
        "./ns3", "run",
        f"scratch/drl_routing/routing_sim "
        f"--enableRL=false "
        f"--enableFail={args['enableFail']} "
        f"--scenario={args['scenario']} "
        f"--simTime=100 "
        f"--output={out_path} "
        f"--seed=42 "
        f"--runNum={run_id+1}"
    ]
    
    try:
        result = subprocess.run(
            cmd, cwd=NS3_DIR, check=True,
            capture_output=True, text=True, timeout=120
        )
        print(f"  ✓ {scenario} run{run_id} complete")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {scenario} run{run_id} failed: {e.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  ✗ {scenario} run{run_id} timed out")
    
    return out_path

def run_drl_scenario(algo, scenario, run_id):
    """
    DRL evaluation requires both ns-3 and Python agent running.
    For now, guides user through manual steps.
    """
    out_path = os.path.join(RAW_DIR, f"{algo}_{scenario}_run{run_id}.xml")
    
    print(f"\n  MANUAL STEP REQUIRED for {algo.upper()} | {scenario} | run {run_id}:")
    print(f"    Terminal 1: cd {NS3_DIR} && ./ns3 run "
          f"'scratch/drl_routing/routing_sim --enableRL=true "
          f"--enableFail={SCENARIO_ARGS[scenario]['enableFail']} "
          f"--scenario={SCENARIO_ARGS[scenario]['scenario']} "
          f"--simTime=100 --output={out_path}'")
    print(f"    Terminal 2: cd {PROJECT_ROOT} && python3 training/run_inference.py "
          f"--algo={algo} --episodes=1")
    
    response = input("  Press Enter when run is complete (or 's' to skip)... ")
    if response.strip().lower() == 's':
        print(f"  Skipped {algo}/{scenario}/run{run_id}")
    
    return out_path

def evaluate_all():
    """Run all algorithms across all scenarios and generate comparison CSVs."""
    all_results = {}
    
    for scenario in SCENARIOS:
        all_results[scenario] = {}
        for algo in ALGORITHMS:
            run_data = []
            for run_id in range(N_RUNS):
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
    
    # Save results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for scenario in SCENARIOS:
        df_sc = pd.DataFrame(all_results[scenario]).T
        df_sc.index.name = 'algorithm'
        csv_path = os.path.join(LOGS_DIR, f"{scenario}_comparison.csv")
        df_sc.round(6).to_csv(csv_path)
        print(f"\nSaved: {csv_path}")
        print(df_sc.round(4).to_string())
    
    # Final summary table
    print("\n" + "="*60)
    print("SUMMARY: Average Throughput (Mbps) across all scenarios")
    print("="*60)
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {algo: all_results[scenario][algo]['throughput']
                            for algo in ALGORITHMS}
    summary_df = pd.DataFrame(summary).round(4)
    summary_df.index.name = 'algorithm'
    print(summary_df.to_string())

if __name__ == "__main__":
    evaluate_all()
