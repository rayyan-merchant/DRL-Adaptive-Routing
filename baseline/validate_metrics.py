import subprocess
import os
import pandas as pd
import sys
from parse_flowmon import parse_flowmon, summarize

NS3_DIR = os.path.expanduser("~/ns-3.38")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../results/raw")

def run_quick(label, extra_args, scenario="normal"):
    outname = f"dijkstra_quick_{scenario}.xml"
    out_path = os.path.join(OUT_DIR, outname)
    cmd = [
        "./ns3", "run",
        f"drl_routing/routing_sim --simTime=30 --enableRL=false "
        f"--scenario={scenario} "
        f"--output={out_path} "
        f"--seed=42 --runNum=1 " + extra_args
    ]
    subprocess.run(cmd, cwd=NS3_DIR, check=True)
    return summarize(parse_flowmon(out_path))

def main():
    print("[TEST 1] BW Sensitivity: Run manually — change DataRate in routing_sim.cc")
    passed = 0
    total = 4 # Exclude TEST 1
    
    try:
        # Check if baseline XMLs already exist
        normal_xml = os.path.join(OUT_DIR, "dijkstra_normal_run0.xml")
        congested_xml = os.path.join(OUT_DIR, "dijkstra_congested_run0.xml")
        failure_xml = os.path.join(OUT_DIR, "dijkstra_failure_run0.xml")
        
        if not (os.path.exists(normal_xml) and os.path.exists(congested_xml) and os.path.exists(failure_xml)):
            print("Baseline XML files not found. Please run run_baseline.py first.")
            sys.exit(1)
            
        normal_res = summarize(parse_flowmon(normal_xml))
        congested_res = summarize(parse_flowmon(congested_xml))
        failure_res = summarize(parse_flowmon(failure_xml))
        
        # TEST 2
        print(f"\n[TEST 2] Congestion Increases Loss")
        if congested_res["loss"] > normal_res["loss"]:
            print(f"PASS (Congested: {congested_res['loss']:.4f} > Normal: {normal_res['loss']:.4f})")
            passed += 1
        else:
            print(f"FAIL (Congested: {congested_res['loss']:.4f} <= Normal: {normal_res['loss']:.4f})")
            
        # TEST 3
        print(f"\n[TEST 3] Congestion Increases Delay")
        if congested_res["delay"] > normal_res["delay"]:
            print(f"PASS (Congested: {congested_res['delay']:.4f} > Normal: {normal_res['delay']:.4f})")
            passed += 1
        else:
            print(f"FAIL (Congested: {congested_res['delay']:.4f} <= Normal: {normal_res['delay']:.4f})")
            
        # TEST 4
        print(f"\n[TEST 4] Link Failure Reduces Throughput During Failure Window")
        if failure_res["throughput"] < normal_res["throughput"]:
            print(f"PASS (Failure: {failure_res['throughput']:.4f} < Normal: {normal_res['throughput']:.4f})")
            passed += 1
        else:
            print(f"FAIL (Failure: {failure_res['throughput']:.4f} >= Normal: {normal_res['throughput']:.4f})")
            
        # TEST 5
        print(f"\n[TEST 5] FlowMonitor Has Exactly 2 Flows")
        df = parse_flowmon(normal_xml)
        if len(df) == 2:
            print(f"PASS (Count: {len(df)})")
            passed += 1
        else:
            print(f"FAIL (Count: {len(df)} != 2)")
            
    except Exception as e:
        print(f"Test failed due to exception: {e}")
        
    print(f"\n{passed}/{total} tests passed.")
    if passed < total:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()