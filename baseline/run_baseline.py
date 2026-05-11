import subprocess
import os
import sys
import pandas as pd
from parse_flowmon import parse_flowmon, summarize

SCENARIOS = ["normal", "congested", "failure"]
N_RUNS = 3

NS3_DIR = os.path.expanduser("~/projects/ns-3-dev")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/raw")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/logs"), exist_ok=True)

def run_ns3(scenario, run_id):
    outname = f"dijkstra_{scenario}_run{run_id}.xml"
    out_path = os.path.abspath(os.path.join(OUT_DIR, outname))
    
    # Updated to use ./ns3 run for modern ns-3
    args = (
        f"routing_sim "
        f"--simTime=100 "
        f"--enableRL=false "
        f"--enableFail={'true' if scenario=='failure' else 'false'} "
        f"--scenario={scenario} "
        f"--output={out_path} "
        f"--seed=42 "
        f"--runNum={run_id+1}"
    )
    
    cmd = ["./ns3", "run", args]
    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=NS3_DIR, check=True)
    return out_path

def run_all_baseline():
    for scenario in SCENARIOS:
        run_summaries = []
        for run_id in range(N_RUNS):
            print(f"Running Dijkstra {scenario} run {run_id}...")
            out_path = run_ns3(scenario, run_id)
            df = parse_flowmon(out_path)
            run_summaries.append(summarize(df))
            
        summary_df = pd.DataFrame(run_summaries).mean()
        print(f"Results for {scenario}:")
        print(summary_df)
        
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/logs", f"dijkstra_{scenario}_summary.csv")
        summary_df.to_csv(log_path)

def validate_metrics():
    print("\n--- Validation ---")
    try:
        normal_df = parse_flowmon(os.path.join(OUT_DIR, "dijkstra_normal_run0.xml"))
        normal_res = summarize(normal_df)
        
        congested_df = parse_flowmon(os.path.join(OUT_DIR, "dijkstra_congested_run0.xml"))
        congested_res = summarize(congested_df)
        
        failure_df = parse_flowmon(os.path.join(OUT_DIR, "dijkstra_failure_run0.xml"))
        failure_res = summarize(failure_df)
        
        print("Normal throughput > 0:", "PASS" if normal_res["throughput"] > 0 else f"FAIL ({normal_res['throughput']})")
        print("Congested loss > 0:", "PASS" if congested_res["loss"] > 0 else f"FAIL ({congested_res['loss']})")
        print("Failure loss > Normal loss:", "PASS" if failure_res["loss"] > normal_res["loss"] else f"FAIL ({failure_res['loss']} <= {normal_res['loss']})")
        
    except FileNotFoundError as e:
        print(f"Validation skipped: {e}")

if __name__ == "__main__":
    run_all_baseline()
    validate_metrics()